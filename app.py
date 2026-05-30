from __future__ import annotations

import csv
import io
import math
import tempfile
from html import escape
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "best.pt"

CLASS_COLORS = {
    "car": (56, 189, 248),
    "person": (34, 197, 94),
    "tree": (250, 204, 21),
    "free_space": (167, 139, 250),
}

SUPPORTED_IMAGES = ["jpg", "jpeg", "png", "webp"]
SUPPORTED_VIDEOS = ["mp4", "mov", "avi", "mkv", "webm"]

TARGET_CLASSES = ["person", "car", "tree", "free_space"]


st.set_page_config(
    page_title="Custom YOLO Predictor",
    page_icon="🔨",
    layout="wide",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
          background:
            radial-gradient(circle at 10% 10%, rgba(56, 189, 248, 0.28), transparent 30%),
            radial-gradient(circle at 85% 5%, rgba(167, 139, 250, 0.28), transparent 28%),
            radial-gradient(circle at 45% 100%, rgba(34, 197, 94, 0.14), transparent 35%),
            #070816;
          color: #f8fafc;
        }

        [data-testid="stHeader"] {
          background: rgba(7, 8, 22, 0);
        }

        .block-container {
          padding-top: 2rem;
          padding-bottom: 4rem;
          max-width: 1380px;
        }

        .glass {
          border: 1px solid rgba(255, 255, 255, 0.16);
          background: linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.045));
          box-shadow: 0 30px 90px rgba(0, 0, 0, 0.42);
          backdrop-filter: blur(20px);
          border-radius: 28px;
          padding: 26px;
        }

        .hero h1 {
          margin: 18px 0 12px;
          font-size: clamp(36px, 5vw, 68px);
          line-height: 0.95;
          letter-spacing: -0.06em;
          color: #f8fafc;
        }

        .hero h1 span {
          display: block;
          color: transparent;
          background: linear-gradient(90deg, #67e8f9, #c4b5fd, #86efac);
          -webkit-background-clip: text;
          background-clip: text;
        }

        .badge {
          display: inline-flex;
          padding: 8px 13px;
          border-radius: 999px;
          color: #bae6fd;
          background: rgba(56, 189, 248, 0.12);
          border: 1px solid rgba(56, 189, 248, 0.24);
          font-size: 13px;
          font-weight: 900;
        }

        .hero p {
          color: #94a3b8;
          max-width: 850px;
          line-height: 1.7;
          font-size: 17px;
        }

        .model-card {
          min-height: 120px;
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 20px;
          border-radius: 24px;
          background: rgba(2, 6, 23, 0.38);
          border: 1px solid rgba(255,255,255,0.1);
        }

        .status-dot {
          width: 18px;
          height: 18px;
          border-radius: 999px;
          background: #22c55e;
          box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.65);
          animation: pulse 1.4s infinite;
        }

        @keyframes pulse {
          70% { box-shadow: 0 0 0 16px rgba(34, 197, 94, 0); }
          100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
        }

        .metric-card {
          border: 1px solid rgba(255,255,255,0.14);
          background: rgba(15, 23, 42, 0.55);
          border-radius: 24px;
          padding: 18px;
          min-height: 112px;
        }

        .metric-card small {
          display: block;
          color: #94a3b8;
          font-weight: 900;
          margin-bottom: 8px;
        }

        .metric-card strong {
          display: block;
          font-size: 42px;
          line-height: 1;
          letter-spacing: -0.055em;
        }

        .metric-card.car strong { color: #38bdf8; }
        .metric-card.person strong { color: #22c55e; }
        .metric-card.tree strong { color: #facc15; }
        .metric-card.free strong { color: #a78bfa; }
        .metric-card.total strong { color: #fb7185; }

        .panel {
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(15, 23, 42, 0.46);
          border-radius: 24px;
          padding: 18px;
          margin-top: 12px;
        }

        .panel h3 {
          margin-top: 0;
          color: #f8fafc;
        }

        .panel p,
        .panel li,
        .panel small {
          color: #cbd5e1;
          line-height: 1.6;
        }

        .risk-low {
          color: #22c55e;
          font-weight: 950;
        }

        .risk-medium {
          color: #facc15;
          font-weight: 950;
        }

        .risk-high {
          color: #fb7185;
          font-weight: 950;
        }

        .detection-row {
          padding: 12px;
          border-radius: 16px;
          background: rgba(2, 6, 23, 0.42);
          border: 1px solid rgba(255,255,255,0.08);
          margin-bottom: 10px;
        }

        .detection-row small {
          color: #94a3b8;
        }

        .feature-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 10px;
          margin-top: 12px;
        }

        .feature-chip {
          padding: 10px 12px;
          border-radius: 14px;
          background: rgba(2, 6, 23, 0.42);
          border: 1px solid rgba(255,255,255,0.08);
        }

        .feature-chip small {
          display: block;
          color: #94a3b8;
          font-weight: 900;
          margin-bottom: 4px;
        }

        .feature-chip strong {
          color: #f8fafc;
        }

        div[data-testid="stFileUploader"] {
          border: 1px dashed rgba(148, 163, 184, 0.55);
          background: rgba(15, 23, 42, 0.44);
          border-radius: 22px;
          padding: 10px;
        }

        .stButton > button {
          min-height: 52px;
          border: 0;
          border-radius: 18px;
          color: #020617;
          background: linear-gradient(135deg, #67e8f9, #a7f3d0);
          font-weight: 950;
          box-shadow: 0 14px 32px rgba(56, 189, 248, 0.2);
        }

        .stDownloadButton > button {
          min-height: 48px;
          border: 0;
          border-radius: 18px;
          color: #020617;
          background: linear-gradient(135deg, #facc15, #a7f3d0);
          font-weight: 950;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Place your trained best.pt inside the models/ folder."
        )

    return YOLO(str(MODEL_PATH))


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb_image(frame: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def get_color(label: str) -> tuple[int, int, int]:
    rgb = CLASS_COLORS.get(label, (255, 255, 255))
    return rgb[2], rgb[1], rgb[0]


def parse_yolo_result(result: Any) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = str(result.names[class_id])

        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height

        detections.append(
            {
                "class_id": class_id,
                "label": label,
                "confidence": confidence,
                "box": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "width": round(width, 2),
                    "height": round(height, 2),
                    "area": round(area, 2),
                },
            }
        )

    return detections


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    output = frame_bgr.copy()

    # Smaller visual style
    box_thickness = 1
    font_scale = 0.42
    font_thickness = 1
    label_padding_x = 5
    label_padding_y = 4

    for detection in detections:
        label = detection["label"]
        confidence = detection["confidence"]
        box = detection["box"]

        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])

        color = get_color(label)
        text = f"{label} {confidence * 100:.1f}%"

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            box_thickness,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            font_thickness,
        )

        text_w, text_h = text_size

        label_w = text_w + label_padding_x * 2
        label_h = text_h + label_padding_y * 2 + baseline

        y_label_top = max(0, y1 - label_h)
        y_label_bottom = y1

        # If box is too close to the top, draw label inside the box
        if y_label_top <= 0:
            y_label_top = y1
            y_label_bottom = min(output.shape[0], y1 + label_h)

        cv2.rectangle(
            output,
            (x1, y_label_top),
            (min(x1 + label_w, output.shape[1] - 1), y_label_bottom),
            color,
            -1,
        )

        text_x = x1 + label_padding_x
        text_y = y_label_bottom - label_padding_y - baseline

        cv2.putText(
            output,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (2, 6, 23),
            font_thickness,
            cv2.LINE_AA,
        )

    return output


def summarize(detections: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "person": 0,
        "car": 0,
        "tree": 0,
        "free_space": 0,
    }

    for detection in detections:
        label = detection["label"]

        if label in summary:
            summary[label] += 1

    return summary


def extract_scene_features(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    image_area = max(1.0, float(image_width * image_height))

    counts = {
        "car": 0.0,
        "person": 0.0,
        "tree": 0.0,
        "free_space": 0.0,
    }

    areas = {
        "car": 0.0,
        "person": 0.0,
        "tree": 0.0,
        "free_space": 0.0,
    }

    confidence_sum = 0.0
    center_activity_score = 0.0
    largest_object_area = 0.0

    center_x = image_width / 2
    center_y = image_height / 2
    max_distance = max(1.0, math.sqrt(center_x**2 + center_y**2))

    for detection in detections:
        label = detection["label"]
        confidence = float(detection["confidence"])
        box = detection["box"]

        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])

        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        box_area = box_width * box_height

        if label in counts:
            counts[label] += 1.0
            areas[label] += box_area

        confidence_sum += confidence
        largest_object_area = max(largest_object_area, box_area)

        object_center_x = (x1 + x2) / 2
        object_center_y = (y1 + y2) / 2

        distance = math.sqrt(
            (object_center_x - center_x) ** 2 + (object_center_y - center_y) ** 2
        )
        center_weight = 1.0 - min(distance / max_distance, 1.0)
        center_activity_score += center_weight

    total_objects = float(len(detections))
    total_detected_area = sum(areas.values())
    avg_confidence = confidence_sum / total_objects if total_objects else 0.0

    car_person_ratio = counts["car"] / max(1.0, counts["person"])
    green_space_ratio = areas["tree"] / max(1.0, areas["free_space"])

    return {
        "car_count": counts["car"],
        "person_count": counts["person"],
        "tree_count": counts["tree"],
        "free_space_count": counts["free_space"],
        "total_objects": total_objects,
        "avg_confidence": avg_confidence,
        "object_density_per_megapixel": total_objects / (image_area / 1_000_000.0),
        "detected_area_percent": (total_detected_area / image_area) * 100.0,
        "car_area_percent": (areas["car"] / image_area) * 100.0,
        "person_area_percent": (areas["person"] / image_area) * 100.0,
        "tree_area_percent": (areas["tree"] / image_area) * 100.0,
        "free_space_area_percent": (areas["free_space"] / image_area) * 100.0,
        "largest_object_area_percent": (largest_object_area / image_area) * 100.0,
        "center_activity_score": center_activity_score,
        "car_person_ratio": car_person_ratio,
        "green_space_ratio": green_space_ratio,
    }


def classify_scene_from_features(features: dict[str, float]) -> dict[str, str]:
    cars = features["car_count"]
    persons = features["person_count"]
    trees = features["tree_count"]
    free_spaces = features["free_space_count"]
    detected_area = features["detected_area_percent"]
    avg_confidence = features["avg_confidence"]

    if features["total_objects"] == 0:
        scene_type = "empty_or_uncertain_scene"
        explanation = "No target objects were detected."
    elif cars >= 5 and free_spaces == 0:
        scene_type = "crowded_parking_scene"
        explanation = "Many cars were detected and no free-space region was found."
    elif cars >= 3:
        scene_type = "vehicle_dominated_scene"
        explanation = "The scene is mainly dominated by vehicles."
    elif persons >= 3:
        scene_type = "human_activity_scene"
        explanation = "Several persons were detected, suggesting human activity."
    elif trees >= 3 and cars == 0:
        scene_type = "green_area_scene"
        explanation = "Tree detections dominate the scene."
    elif free_spaces >= 1 and cars <= 2:
        scene_type = "open_space_scene"
        explanation = "Free-space detections suggest available open area."
    elif detected_area > 40:
        scene_type = "dense_object_scene"
        explanation = "Detected objects cover a large part of the image."
    else:
        scene_type = "mixed_scene"
        explanation = "The scene contains a mixed distribution of detected classes."

    if avg_confidence < 0.35 and features["total_objects"] > 0:
        reliability = "low"
    elif avg_confidence < 0.65:
        reliability = "medium"
    else:
        reliability = "high"

    return {
        "scene_type": scene_type,
        "explanation": explanation,
        "reliability": reliability,
    }


def calculate_anomaly_score(features: dict[str, float]) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []

    if features["avg_confidence"] < 0.35 and features["total_objects"] > 0:
        score += 2
        reasons.append("Low average confidence may indicate uncertain detections.")

    if features["car_count"] >= 10:
        score += 2
        reasons.append("High vehicle count may indicate congestion.")

    if features["person_count"] >= 5:
        score += 2
        reasons.append("High person count indicates increased human activity.")

    if features["detected_area_percent"] > 60:
        score += 2
        reasons.append("Detected objects cover a large part of the image.")

    if features["free_space_count"] == 0 and features["car_count"] >= 5:
        score += 1
        reasons.append("No free-space region was detected while several cars are present.")

    if features["center_activity_score"] >= 6:
        score += 1
        reasons.append("Many detections are concentrated near the image center.")

    if score >= 5:
        level = "high"
    elif score >= 2:
        level = "medium"
    else:
        level = "low"

    if not reasons:
        reasons.append("No unusual pattern was detected from the available features.")

    return {
        "score": score,
        "level": level,
        "reasons": reasons,
    }


def normalize_features_for_scene_map(features: dict[str, float]) -> list[float]:
    return [
        min(features["car_count"] / 10.0, 1.0),
        min(features["person_count"] / 8.0, 1.0),
        min(features["tree_count"] / 8.0, 1.0),
        min(features["free_space_count"] / 4.0, 1.0),
        min(features["detected_area_percent"] / 70.0, 1.0),
        min(features["center_activity_score"] / 8.0, 1.0),
        features["avg_confidence"],
    ]


def match_som_style_scene_pattern(features: dict[str, float]) -> dict[str, Any]:
    vector = normalize_features_for_scene_map(features)

    prototypes = [
        {
            "name": "open_low_activity",
            "grid": (0, 0),
            "vector": [0.0, 0.0, 0.1, 0.8, 0.15, 0.1, 0.75],
            "description": "Open scene with low activity and visible available space.",
        },
        {
            "name": "vehicle_cluster",
            "grid": (1, 0),
            "vector": [0.8, 0.1, 0.1, 0.1, 0.35, 0.5, 0.75],
            "description": "Vehicle-dominated scene with multiple cars.",
        },
        {
            "name": "human_activity",
            "grid": (2, 0),
            "vector": [0.2, 0.8, 0.1, 0.1, 0.25, 0.6, 0.75],
            "description": "Scene with noticeable human activity.",
        },
        {
            "name": "green_area",
            "grid": (0, 1),
            "vector": [0.0, 0.0, 0.9, 0.2, 0.45, 0.3, 0.75],
            "description": "Tree or vegetation dominated scene.",
        },
        {
            "name": "mixed_environment",
            "grid": (1, 1),
            "vector": [0.4, 0.3, 0.3, 0.3, 0.35, 0.4, 0.7],
            "description": "Mixed outdoor scene with multiple object types.",
        },
        {
            "name": "crowded_or_complex",
            "grid": (2, 1),
            "vector": [0.9, 0.6, 0.2, 0.0, 0.65, 0.8, 0.65],
            "description": "Crowded or complex scene with limited free space.",
        },
        {
            "name": "uncertain_sparse",
            "grid": (0, 2),
            "vector": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.25],
            "description": "Sparse or uncertain scene with low detection confidence.",
        },
        {
            "name": "dense_large_objects",
            "grid": (1, 2),
            "vector": [0.5, 0.2, 0.4, 0.2, 0.85, 0.6, 0.7],
            "description": "Scene where detected objects cover a large image area.",
        },
        {
            "name": "central_activity",
            "grid": (2, 2),
            "vector": [0.4, 0.4, 0.2, 0.1, 0.35, 0.95, 0.7],
            "description": "Scene with objects concentrated near the image center.",
        },
    ]

    best = None
    best_distance = float("inf")

    for prototype in prototypes:
        distance = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(vector, prototype["vector"]))
        )

        if distance < best_distance:
            best_distance = distance
            best = prototype

    confidence = max(0.0, 1.0 - best_distance / math.sqrt(len(vector)))

    return {
        "pattern": best["name"],
        "grid": best["grid"],
        "description": best["description"],
        "similarity": confidence,
        "distance": best_distance,
    }


def generate_ai_summary(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    media_type: str = "image",
) -> dict[str, Any]:
    summary = summarize(detections)
    features = extract_scene_features(detections, image_width, image_height)
    scene = classify_scene_from_features(features)
    anomaly = calculate_anomaly_score(features)
    scene_map = match_som_style_scene_pattern(features)

    total = int(features["total_objects"])
    avg_confidence = features["avg_confidence"]
    occupied_percent = features["detected_area_percent"]

    observations: list[str] = []

    if total == 0:
        observations.append(
            "No target objects were detected. The scene may be empty, unclear, or below the selected confidence threshold."
        )
    else:
        observations.append(
            f"The model detected {total} target object{'s' if total != 1 else ''} in this {media_type}."
        )

    if summary["car"] > 0:
        observations.append(
            f"{summary['car']} car{'s were' if summary['car'] != 1 else ' was'} detected, suggesting vehicle presence or parking activity."
        )

    if summary["person"] > 0:
        observations.append(
            f"{summary['person']} person{'s were' if summary['person'] != 1 else ' was'} detected, indicating human activity."
        )

    if summary["tree"] > 0:
        observations.append(
            f"{summary['tree']} tree region{'s were' if summary['tree'] != 1 else ' was'} detected, suggesting visible vegetation or green infrastructure."
        )

    if summary["free_space"] > 0:
        observations.append(
            f"{summary['free_space']} free-space region{'s were' if summary['free_space'] != 1 else ' was'} detected, suggesting available open area."
        )

    observations.append(
        f"The detected bounding boxes cover approximately {occupied_percent:.1f}% of the image area."
    )

    if avg_confidence >= 0.75:
        observations.append("The average detection confidence is high.")
    elif avg_confidence >= 0.45:
        observations.append("The average detection confidence is moderate.")
    elif avg_confidence > 0:
        observations.append(
            "The average detection confidence is low, so the result should be reviewed carefully."
        )
    else:
        observations.append("No confidence score is available because no objects were detected.")

    observations.append(
        f"The SOM-style scene map matched this image to the '{scene_map['pattern']}' pattern."
    )

    if anomaly["level"] == "high":
        recommendation = (
            "Manual review is strongly recommended because the scene appears complex or unusual."
        )
    elif anomaly["level"] == "medium":
        recommendation = (
            "Manual review is recommended, especially if this image is used for monitoring or decision support."
        )
    else:
        recommendation = (
            "The scene appears relatively simple based on the current detections."
        )

    if total == 0:
        recommendation = (
            "Try lowering the confidence threshold or uploading a clearer image if objects are expected."
        )

    return {
        "summary": summary,
        "features": features,
        "scene": scene,
        "anomaly": anomaly,
        "scene_map": scene_map,
        "observations": observations,
        "recommendation": recommendation,
    }


def detections_to_csv(detections: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(
        [
            "index",
            "class_id",
            "label",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "width",
            "height",
            "area",
        ]
    )

    for index, detection in enumerate(detections, start=1):
        box = detection["box"]
        writer.writerow(
            [
                index,
                detection["class_id"],
                detection["label"],
                round(detection["confidence"], 5),
                box["x1"],
                box["y1"],
                box["x2"],
                box["y2"],
                box["width"],
                box["height"],
                box["area"],
            ]
        )

    return buffer.getvalue().encode("utf-8")


def render_hero(model: YOLO) -> None:
    names = ", ".join(str(name) for name in model.names.values())

    left, right = st.columns([2.2, 1])

    with left:
        st.markdown(
            """
            <section class="hero glass">
              <div class="badge">🔨🤖🔧 Custom Trained YOLO Predictor</div>
              <h1>
                Predict, visualize
                <span>and understand scenes</span>
              </h1>
              <p>
                Upload an image or video. Your trained <b>best.pt</b> YOLO11 model runs inside Streamlit,
                then this dashboard visualizes detections, bounding boxes, AI scene summaries,
                feature analytics, anomaly scores, and SOM-style scene patterns.
              </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
            <div class="model-card">
              <div class="status-dot"></div>
              <div>
                <strong>Model ready</strong><br>
                <small>Classes: {escape(names)}</small>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_metrics(summary: dict[str, int]) -> None:
    total = sum(summary.values())

    cols = st.columns(5)

    cards = [
        (cols[0], "person", "Persons", summary.get("person", 0)),
        (cols[1], "car", "Cars", summary.get("car", 0)),
        (cols[2], "tree", "Trees", summary.get("tree", 0)),
        (cols[3], "free", "Free Space", summary.get("free_space", 0)),
        (cols[4], "total", "Total", total),
    ]

    for column, css_class, label, value in cards:
        with column:
            st.markdown(
                f"""
                <div class="metric-card {css_class}">
                  <small>{escape(label)}</small>
                  <strong>{value}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_ai_summary(ai_summary: dict[str, Any]) -> None:
    anomaly = ai_summary["anomaly"]
    scene = ai_summary["scene"]
    scene_map = ai_summary["scene_map"]
    features = ai_summary["features"]

    risk_level = anomaly["level"]
    risk_class = {
        "low": "risk-low",
        "medium": "risk-medium",
        "high": "risk-high",
    }.get(risk_level, "risk-medium")

    observations_html = "".join(
        f"<li>{escape(observation)}</li>" for observation in ai_summary["observations"]
    )

    reasons_html = "".join(
        f"<li>{escape(reason)}</li>" for reason in anomaly["reasons"]
    )

    st.markdown(
        f"""
        <div class="panel">
          <h3>🤖 AI Scene Intelligence</h3>

          <p>
            <strong>Scene type:</strong> {escape(scene["scene_type"])}<br>
            <strong>Scene explanation:</strong> {escape(scene["explanation"])}<br>
            <strong>Reliability:</strong> {escape(scene["reliability"]).upper()}
          </p>

          <p>
            <strong>Anomaly / complexity level:</strong>
            <span class="{risk_class}">{escape(risk_level).upper()}</span>
            &nbsp; | &nbsp;
            <strong>Anomaly score:</strong> {anomaly["score"]}
          </p>

          <p>
            <strong>SOM-style pattern:</strong> {escape(scene_map["pattern"])}<br>
            <strong>Pattern grid position:</strong> {scene_map["grid"]}<br>
            <strong>Pattern similarity:</strong> {scene_map["similarity"] * 100:.1f}%<br>
            <strong>Pattern meaning:</strong> {escape(scene_map["description"])}
          </p>

          <ul>{observations_html}</ul>

          <p><strong>Anomaly reasons:</strong></p>
          <ul>{reasons_html}</ul>

          <p><strong>Recommendation:</strong> {escape(ai_summary["recommendation"])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_feature_panel(features)


def render_feature_panel(features: dict[str, float]) -> None:
    chips = [
        ("Avg confidence", f"{features['avg_confidence'] * 100:.1f}%"),
        ("Detected area", f"{features['detected_area_percent']:.1f}%"),
        ("Object density", f"{features['object_density_per_megapixel']:.2f}/MP"),
        ("Center activity", f"{features['center_activity_score']:.2f}"),
        ("Car area", f"{features['car_area_percent']:.1f}%"),
        ("Person area", f"{features['person_area_percent']:.1f}%"),
        ("Tree area", f"{features['tree_area_percent']:.1f}%"),
        ("Free-space area", f"{features['free_space_area_percent']:.1f}%"),
        ("Largest object", f"{features['largest_object_area_percent']:.1f}%"),
        ("Car/person ratio", f"{features['car_person_ratio']:.2f}"),
    ]

    chips_html = ""

    for label, value in chips:
        chips_html += f"""
        <div class="feature-chip">
          <small>{escape(label)}</small>
          <strong>{escape(value)}</strong>
        </div>
        """

    st.markdown(
        f"""
        <div class="panel">
          <h3>📊 Extracted Scene Features</h3>
          <div class="feature-grid">
            {chips_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detections(detections: list[dict[str, Any]]) -> None:
    st.markdown("### Detections")

    if not detections:
        st.markdown(
            """
            <div class="panel">
              No detections. Try lowering confidence or use a clearer image.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    rows_html = ""

    for index, detection in enumerate(detections, start=1):
        box = detection["box"]

        rows_html += f"""
        <div class="detection-row">
          <strong>#{index} {escape(detection["label"])}</strong>
          <span style="float:right;">{detection["confidence"] * 100:.1f}%</span><br>
          <small>
            x1: {box["x1"]}, y1: {box["y1"]}<br>
            width: {box["width"]}, height: {box["height"]}<br>
            area: {box["area"]}
          </small>
        </div>
        """

    st.markdown(
        f"""
        <div class="panel">
          {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def predict_image(
    model: YOLO,
    image: Image.Image,
    confidence: float,
    image_size: int,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    frame_bgr = pil_to_bgr(image)

    results = model.predict(
        source=frame_bgr,
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )

    detections = parse_yolo_result(results[0])
    annotated_bgr = draw_detections(frame_bgr, detections)
    annotated_image = bgr_to_rgb_image(annotated_bgr)

    return annotated_image, detections


def process_video(
    model: YOLO,
    uploaded_file: Any,
    confidence: float,
    image_size: int,
    frame_skip: int,
) -> tuple[Path, dict[str, int], dict[str, Any]]:
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as input_file:
        input_file.write(uploaded_file.read())
        input_path = Path(input_file.name)

    output_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name)

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise ValueError("Could not open the uploaded video.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    processed_frames = 0
    latest_detections: list[dict[str, Any]] = []

    global_summary = {
        "person": 0,
        "car": 0,
        "tree": 0,
        "free_space": 0,
    }

    total_detections = 0
    confidence_sum = 0.0

    progress = st.progress(0)
    status = st.empty()
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1)

    while True:
        ok, frame = capture.read()

        if not ok:
            break

        should_predict = frame_index % frame_skip == 0

        if should_predict:
            results = model.predict(
                source=frame,
                conf=confidence,
                imgsz=image_size,
                verbose=False,
            )

            latest_detections = parse_yolo_result(results[0])
            frame_summary = summarize(latest_detections)

            for key, value in frame_summary.items():
                global_summary[key] += value

            for detection in latest_detections:
                total_detections += 1
                confidence_sum += float(detection["confidence"])

            processed_frames += 1

        annotated = draw_detections(frame, latest_detections)
        writer.write(annotated)

        frame_index += 1
        progress.progress(min(frame_index / max(total_frames, 1), 1.0))
        status.write(f"Processing frame {frame_index}/{total_frames}")

    capture.release()
    writer.release()

    progress.empty()
    status.empty()

    video_stats = {
        "processed_frames": processed_frames,
        "total_frames": frame_index,
        "width": width,
        "height": height,
        "fps": fps,
        "total_detections": total_detections,
        "average_confidence": confidence_sum / total_detections if total_detections else 0.0,
    }

    return output_path, global_summary, video_stats


def render_video_summary(summary: dict[str, int], stats: dict[str, Any], frame_skip: int) -> None:
    total_objects = sum(summary.values())

    if total_objects >= 50:
        risk_level = "high"
    elif total_objects >= 10:
        risk_level = "medium"
    else:
        risk_level = "low"

    risk_class = {
        "low": "risk-low",
        "medium": "risk-medium",
        "high": "risk-high",
    }[risk_level]

    st.markdown(
        f"""
        <div class="panel">
          <h3>🎞️ Video AI Summary</h3>
          <p>
            <strong>Processed frames:</strong> {stats["processed_frames"]}<br>
            <strong>Total frames read:</strong> {stats["total_frames"]}<br>
            <strong>Frame skip:</strong> {frame_skip}<br>
            <strong>Video size:</strong> {stats["width"]} × {stats["height"]}<br>
            <strong>FPS:</strong> {stats["fps"]:.2f}
          </p>
          <p>
            <strong>Total detections across processed frames:</strong> {total_objects}<br>
            <strong>Average detection confidence:</strong> {stats["average_confidence"] * 100:.1f}%<br>
            <strong>Video activity level:</strong>
            <span class="{risk_class}">{risk_level.upper()}</span>
          </p>
          <ul>
            <li>Cars detected across frames: {summary.get("car", 0)}</li>
            <li>Persons detected across frames: {summary.get("person", 0)}</li>
            <li>Trees detected across frames: {summary.get("tree", 0)}</li>
            <li>Free-space regions detected across frames: {summary.get("free_space", 0)}</li>
          </ul>
          <p>
            <strong>Recommendation:</strong>
            Use a lower frame skip for more detailed temporal analysis, or a higher frame skip for faster processing.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="glass" style="text-align:center; padding:70px 20px;">
          <div style="font-size:64px;">🛰️</div>
          <h2>No prediction yet</h2>
          <p style="color:#94a3b8;">Upload an image or video and click Predict.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()

    try:
        model = load_model()
    except Exception as error:
        st.error(str(error))
        st.stop()

    render_hero(model)

    st.markdown("## Upload & Controls")

    controls_col, options_col = st.columns([1.4, 1])

    with controls_col:
        uploaded_file = st.file_uploader(
            "Choose image or video",
            type=SUPPORTED_IMAGES + SUPPORTED_VIDEOS,
        )

    with options_col:
        confidence_percent = st.slider("Confidence", 5, 90, 25, 5)
        image_size = st.slider("Image size", 320, 1280, 640, 32)
        frame_skip = st.slider("Video frame skip", 1, 10, 1, 1)

    confidence = confidence_percent / 100

    initial_summary = {
        "person": 0,
        "car": 0,
        "tree": 0,
        "free_space": 0,
    }

    if uploaded_file is None:
        render_metrics(initial_summary)
        render_empty_state()
        return

    file_ext = Path(uploaded_file.name).suffix.lower().replace(".", "")

    if file_ext in SUPPORTED_IMAGES:
        image = Image.open(uploaded_file).convert("RGB")

        if st.button("Predict Image", use_container_width=True):
            with st.spinner("Running YOLO prediction..."):
                annotated_image, detections = predict_image(
                    model=model,
                    image=image,
                    confidence=confidence,
                    image_size=image_size,
                )

            summary = summarize(detections)
            ai_summary = generate_ai_summary(
                detections=detections,
                image_width=image.width,
                image_height=image.height,
                media_type="image",
            )

            render_metrics(summary)

            left, right = st.columns([1.6, 1])

            with left:
                st.image(
                    annotated_image,
                    caption="Annotated result",
                    use_container_width=True,
                )

                st.download_button(
                    label="Download detections CSV",
                    data=detections_to_csv(detections),
                    file_name="detections.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with right:
                render_ai_summary(ai_summary)
                render_detections(detections)

        else:
            render_metrics(initial_summary)
            st.image(
                image,
                caption="Uploaded image preview",
                use_container_width=True,
            )

    elif file_ext in SUPPORTED_VIDEOS:
        st.video(uploaded_file)

        if st.button("Predict Video", use_container_width=True):
            with st.spinner("Processing video. This may take time on free CPU..."):
                output_path, summary, stats = process_video(
                    model=model,
                    uploaded_file=uploaded_file,
                    confidence=confidence,
                    image_size=image_size,
                    frame_skip=frame_skip,
                )

            render_metrics(summary)
            render_video_summary(summary, stats, frame_skip)

            st.success(f"Video processed. Frames with prediction: {stats['processed_frames']}")

            video_bytes = output_path.read_bytes()
            st.video(video_bytes)

            st.download_button(
                label="Download annotated video",
                data=video_bytes,
                file_name="annotated_video.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
        else:
            render_metrics(initial_summary)

    else:
        st.error("Unsupported file type.")


if __name__ == "__main__":
    main()
