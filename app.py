from __future__ import annotations

import tempfile
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

        .result-panel, .summary-panel {
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(15, 23, 42, 0.46);
          border-radius: 24px;
          padding: 18px;
          margin-top: 12px;
        }

        .summary-panel h3 {
          margin-top: 0;
          color: #f8fafc;
        }

        .summary-panel p, .summary-panel li {
          color: #cbd5e1;
          line-height: 1.6;
        }

        .summary-panel .risk-low {
          color: #22c55e;
          font-weight: 900;
        }

        .summary-panel .risk-medium {
          color: #facc15;
          font-weight: 900;
        }

        .summary-panel .risk-high {
          color: #fb7185;
          font-weight: 900;
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
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

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
                    "width": round(x2 - x1, 2),
                    "height": round(y2 - y1, 2),
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

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(text, font, 0.65, 2)
        text_w, text_h = text_size
        y_label_top = max(0, y1 - text_h - 14)

        cv2.rectangle(
            output,
            (x1, y_label_top),
            (x1 + text_w + 14, y1),
            color,
            -1,
        )

        cv2.putText(
            output,
            text,
            (x1 + 7, max(18, y1 - 8)),
            font,
            0.65,
            (2, 6, 23),
            2,
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


def generate_ai_summary(
    detections: list[dict[str, Any]],
    image_width: int | None = None,
    image_height: int | None = None,
    media_type: str = "image",
) -> dict[str, Any]:
    summary = summarize(detections)

    total = len(detections)
    cars = summary.get("car", 0)
    persons = summary.get("person", 0)
    trees = summary.get("tree", 0)
    free_spaces = summary.get("free_space", 0)

    if detections:
        avg_confidence = sum(item["confidence"] for item in detections) / len(detections)
    else:
        avg_confidence = 0.0

    occupied_percent = None
    if image_width and image_height and image_width > 0 and image_height > 0:
        image_area = image_width * image_height
        detected_area = sum(item["box"]["area"] for item in detections)
        occupied_percent = min(100.0, (detected_area / image_area) * 100)

    observations: list[str] = []

    if total == 0:
        observations.append(
            "No target objects were detected. The scene may be empty, unclear, or below the selected confidence threshold."
        )
    else:
        observations.append(
            f"The model detected {total} target object{'s' if total != 1 else ''} in this {media_type}."
        )

    if cars > 0:
        observations.append(
            f"{cars} car{'s were' if cars != 1 else ' was'} detected, suggesting vehicle presence or parking activity."
        )

    if persons > 0:
        observations.append(
            f"{persons} person{'s were' if persons != 1 else ' was'} detected, indicating human activity in the scene."
        )

    if trees > 0:
        observations.append(
            f"{trees} tree region{'s were' if trees != 1 else ' was'} detected, suggesting visible vegetation or green infrastructure."
        )

    if free_spaces > 0:
        observations.append(
            f"{free_spaces} free-space region{'s were' if free_spaces != 1 else ' was'} detected, which may indicate available open areas."
        )

    if occupied_percent is not None:
        observations.append(
            f"The detected bounding boxes cover approximately {occupied_percent:.1f}% of the image area."
        )

    if avg_confidence >= 0.75:
        confidence_text = "The average detection confidence is high."
    elif avg_confidence >= 0.45:
        confidence_text = "The average detection confidence is moderate."
    elif avg_confidence > 0:
        confidence_text = "The average detection confidence is low, so the result should be reviewed carefully."
    else:
        confidence_text = "No confidence score is available because no objects were detected."

    observations.append(confidence_text)

    risk_score = 0

    if persons >= 5:
        risk_score += 2
    elif persons >= 1:
        risk_score += 1

    if cars >= 10:
        risk_score += 2
    elif cars >= 3:
        risk_score += 1

    if free_spaces == 0 and (cars + persons) >= 5:
        risk_score += 1

    if avg_confidence < 0.35 and total > 0:
        risk_score += 1

    if risk_score >= 4:
        risk_level = "high"
        recommendation = (
            "The scene appears crowded or complex. Manual review is recommended, especially if this is used for safety monitoring."
        )
    elif risk_score >= 2:
        risk_level = "medium"
        recommendation = (
            "The scene shows moderate activity. Review detections and consider lowering or raising confidence depending on false positives."
        )
    else:
        risk_level = "low"
        recommendation = (
            "The scene appears relatively simple based on the detected objects."
        )

    if total == 0:
        risk_level = "low"
        recommendation = (
            "Try lowering the confidence threshold or uploading a clearer image if objects are expected."
        )

    return {
        "summary": summary,
        "total": total,
        "average_confidence": avg_confidence,
        "occupied_percent": occupied_percent,
        "observations": observations,
        "risk_level": risk_level,
        "recommendation": recommendation,
    }


def render_ai_summary(ai_summary: dict[str, Any]) -> None:
    risk_level = ai_summary["risk_level"]
    risk_class = {
        "low": "risk-low",
        "medium": "risk-medium",
        "high": "risk-high",
    }.get(risk_level, "risk-medium")

    observations_html = "".join(
        f"<li>{observation}</li>" for observation in ai_summary["observations"]
    )

    avg_conf = ai_summary["average_confidence"] * 100

    occupied = ai_summary["occupied_percent"]
    occupied_text = "N/A" if occupied is None else f"{occupied:.1f}%"

    st.markdown(
        f"""
        <div class="summary-panel">
          <h3>🤖 AI Scene Summary</h3>
          <p>
            <strong>Risk / complexity level:</strong>
            <span class="{risk_class}">{risk_level.upper()}</span>
          </p>
          <p>
            <strong>Average confidence:</strong> {avg_conf:.1f}%<br>
            <strong>Approximate detected area:</strong> {occupied_text}
          </p>
          <ul>
            {observations_html}
          </ul>
          <p><strong>Recommendation:</strong> {ai_summary["recommendation"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(model: YOLO) -> None:
    names = ", ".join(str(name) for name in model.names.values())

    left, right = st.columns([2.2, 1])

    with left:
        st.markdown(
            """
            <section class="hero glass">
              <div class="badge">🔨🤖🔧 Custom Trained YOLO Predictor</div>
              <h1>
                Predict & visualize
                <span>cars, people, trees, free space</span>
              </h1>
              <p>
                Upload an image or video. Your trained <b>best.pt</b> model runs inside Streamlit,
                then this dashboard visualizes detections, confidence scores, counts,
                and an automatic AI-style scene summary.
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
                <small>Classes: {names}</small>
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
                  <small>{label}</small>
                  <strong>{value}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_detections(detections: list[dict[str, Any]]) -> None:
    st.markdown("### Detections")

    if not detections:
        st.markdown(
            """
            <div class="result-panel">
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
          <strong>#{index} {detection["label"]}</strong>
          <span style="float:right;">{detection["confidence"] * 100:.1f}%</span><br>
          <small>
            x1: {box["x1"]}, y1: {box["y1"]}<br>
            width: {box["width"]}, height: {box["height"]}
          </small>
        </div>
        """

    st.markdown(
        f"""
        <div class="result-panel">
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
) -> tuple[Path, dict[str, int], int]:
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

    return output_path, global_summary, processed_frames


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

            left, right = st.columns([1.7, 1])

            with left:
                st.image(
                    annotated_image,
                    caption="Annotated result",
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
                output_path, summary, processed_frames = process_video(
                    model=model,
                    uploaded_file=uploaded_file,
                    confidence=confidence,
                    image_size=image_size,
                    frame_skip=frame_skip,
                )

            render_metrics(summary)

            video_ai_summary = {
                "risk_level": "medium" if sum(summary.values()) > 0 else "low",
                "average_confidence": 0.0,
                "occupied_percent": None,
                "observations": [
                    f"The video was processed using frame skip {frame_skip}.",
                    f"The model processed {processed_frames} frames for object detection.",
                    f"Detected object totals across processed frames: "
                    f"{summary.get('car', 0)} cars, "
                    f"{summary.get('person', 0)} persons, "
                    f"{summary.get('tree', 0)} trees, "
                    f"{summary.get('free_space', 0)} free-space regions.",
                ],
                "recommendation": (
                    "Review the annotated video and use a lower frame skip for more detailed temporal analysis."
                ),
            }

            render_ai_summary(video_ai_summary)

            st.success(f"Video processed. Frames with prediction: {processed_frames}")

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
