import streamlit as st
import cv2
import numpy as np
import os
import time

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

st.set_page_config(page_title="Drowsiness Detection System", page_icon="🚗", layout="wide")

MODEL_PATH = "detect.tflite"
LABEL_MAP = {1: "supir-sadar", 2: "supir-mengantuk", 3: "supir-tertidur"}
COLOR_MAP = {1: (34, 197, 94), 2: (245, 158, 11), 3: (239, 68, 68)}


@st.cache_resource
def load_model():
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return (
        interpreter,
        interpreter.get_input_details(),
        interpreter.get_output_details(),
    )


try:
    interpreter, input_details, output_details = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


def generate_demo_images():
    os.makedirs("demo_images", exist_ok=True)
    from PIL import Image, ImageDraw

    img1 = Image.new("RGB", (640, 480), color="#1e293b")
    draw1 = ImageDraw.Draw(img1)
    draw1.ellipse([220, 100, 420, 380], outline="#22c55e", width=5)
    draw1.ellipse([260, 180, 300, 220], outline="#22c55e", width=3)
    draw1.ellipse([280, 200, 290, 210], fill="#22c55e")
    draw1.ellipse([340, 180, 380, 220], outline="#22c55e", width=3)
    draw1.ellipse([350, 200, 360, 210], fill="#22c55e")
    draw1.arc([280, 260, 360, 320], start=0, end=180, fill="#22c55e", width=4)
    draw1.text((30, 30), "DEMO: SUPIR SADAR (ALERT DRIVER)", fill="#22c55e")
    img1.save("demo_images/driver_alert.jpg")

    img2 = Image.new("RGB", (640, 480), color="#1e293b")
    draw2 = ImageDraw.Draw(img2)
    draw2.ellipse([220, 100, 420, 380], outline="#f59e0b", width=5)
    draw2.line([250, 200, 310, 200], fill="#f59e0b", width=5)
    draw2.line([330, 200, 390, 200], fill="#f59e0b", width=5)
    draw2.line([280, 290, 360, 290], fill="#f59e0b", width=4)
    draw2.text((30, 30), "DEMO: SUPIR MENGANTUK (SLEEPY DRIVER)", fill="#f59e0b")
    img2.save("demo_images/driver_sleepy.jpg")

    img3 = Image.new("RGB", (640, 480), color="#1e293b")
    draw3 = ImageDraw.Draw(img3)
    draw3.ellipse([220, 100, 420, 380], outline="#ef4444", width=5)
    draw3.arc([250, 180, 310, 220], start=0, end=180, fill="#ef4444", width=5)
    draw3.arc([330, 180, 390, 220], start=0, end=180, fill="#ef4444", width=5)
    draw3.ellipse([290, 270, 350, 330], outline="#ef4444", width=4)
    draw3.text((30, 30), "DEMO: SUPIR TERTIDUR (ASLEEP DRIVER)", fill="#ef4444")
    img3.save("demo_images/driver_asleep.jpg")


if not os.path.exists("demo_images/driver_alert.jpg"):
    generate_demo_images()


def preprocess_image(image, target_size=320):
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(enhanced, (new_w, new_h), interpolation=interp)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    left = pad_w // 2
    padded = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=128
    )

    input_data = np.expand_dims(padded, axis=0).astype(np.float32)
    if input_details[0]["dtype"] == np.float32:
        input_data = (input_data - 127.5) / 127.5

    return input_data, scale, top, left, h, w


def adjust_boxes(detections, scale, top, left, orig_h, orig_w, target_size=320):
    adjusted = []
    for det in detections:
        ymin, xmin, ymax, xmax = det["box"]
        by = (ymin * target_size - top) / scale
        bx = (xmin * target_size - left) / scale
        ey = (ymax * target_size - top) / scale
        ex = (xmax * target_size - left) / scale
        by = max(0, by)
        bx = max(0, bx)
        ey = min(orig_h, ey)
        ex = min(orig_w, ex)
        det = det.copy()
        det["box_px"] = (int(by), int(bx), int(ey), int(ex))
        adjusted.append(det)
    return adjusted


def run_drowsiness_detection(image, score_threshold=0.5):
    if image is None:
        return None, "<p style='text-align:center;'>No input image provided.</p>", ""

    if interpreter is None:
        return (
            image,
            "<div style='color:red; font-weight:bold; text-align:center;'>Error: Model failed to load.</div>",
            "",
        )

    input_data, scale, top, left, orig_h, orig_w = preprocess_image(image)
    draw_img = image.copy()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    outname = output_details[0]["name"]
    if "StatefulPartitionedCall" in outname:
        scores_idx, boxes_idx, classes_idx = 0, 1, 3
    else:
        classes_idx, boxes_idx, scores_idx = 0, 1, 3

    raw_scores = interpreter.get_tensor(output_details[scores_idx]["index"])[0]
    raw_boxes = interpreter.get_tensor(output_details[boxes_idx]["index"])[0]
    raw_classes = interpreter.get_tensor(output_details[classes_idx]["index"])[0]
    raw_count = interpreter.get_tensor(output_details[2]["index"])[0]

    num_detections = int(raw_count)
    detections = []

    for i in range(num_detections):
        score = float(raw_scores[i])
        if score < score_threshold:
            continue
        class_id = int(raw_classes[i])
        box = raw_boxes[i]
        detections.append(
            {
                "class_id": class_id,
                "label": LABEL_MAP.get(class_id, f"ID {class_id}"),
                "score": score,
                "box": box,
            }
        )

    detections = adjust_boxes(detections, scale, top, left, orig_h, orig_w)

    for det in detections:
        ymin_px, xmin_px, ymax_px, xmax_px = det["box_px"]

        color = COLOR_MAP.get(det["class_id"], (255, 255, 255))
        label_text = f"{det['label']} ({det['score']:.2%})"

        cv2.rectangle(
            draw_img, (xmin_px, ymin_px), (xmax_px, ymax_px), color, thickness=3
        )

        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        text_y = max(ymin_px, text_h + 10)
        cv2.rectangle(
            draw_img,
            (xmin_px, text_y - text_h - 10),
            (xmin_px + text_w, text_y),
            color,
            -1,
        )
        cv2.putText(
            draw_img,
            label_text,
            (xmin_px, text_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if not detections:
        alert_html = """
        <div style="background-color: #1e293b; color: #94a3b8; border: 1px solid #334155; padding: 20px; border-radius: 12px; text-align: center;">
            <p style="font-size: 1.5em; margin: 0; font-weight: bold;">TIDAK ADA DETEKSI</p>
            <p style="margin: 5px 0 0 0; font-size: 0.9em;">Atur slider Threshold lebih rendah atau pilih gambar driver yang lebih jelas.</p>
        </div>
        """
    else:
        has_tertidur = any(d["class_id"] == 3 for d in detections)
        has_mengantuk = any(d["class_id"] == 2 for d in detections)

        if has_tertidur:
            alert_html = """
            <div style="background-color: #fef2f2; color: #b91c1c; border: 2px solid #ef4444; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.25);">
                <p style="font-size: 1.8em; margin: 0; font-weight: bold;">BAHAYA: SUPIR TERTIDUR!</p>
                <p style="margin: 5px 0 0 0; font-size: 1.1em; font-weight: 500;">Model mendeteksi pengemudi dalam keadaan tertidur lelap.</p>
            </div>
            """
        elif has_mengantuk:
            alert_html = """
            <div style="background-color: #fffbeb; color: #b45309; border: 2px solid #f59e0b; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(245, 158, 11, 0.25);">
                <p style="font-size: 1.8em; margin: 0; font-weight: bold;">PERINGATAN: SUPIR MENGANTUK</p>
                <p style="margin: 5px 0 0 0; font-size: 1.1em; font-weight: 500;">Model mendeteksi indikasi kantuk. Berikan alarm pengingat!</p>
            </div>
            """
        else:
            alert_html = """
            <div style="background-color: #f0fdf4; color: #15803d; border: 2px solid #22c55e; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(34, 197, 94, 0.25);">
                <p style="font-size: 1.8em; margin: 0; font-weight: bold;">AMAN: SUPIR FOKUS</p>
                <p style="margin: 5px 0 0 0; font-size: 1.1em; font-weight: 500;">Model mendeteksi pengemudi dalam kondisi sadar dan terjaga.</p>
            </div>
            """

    if not detections:
        table_md = "*Tidak ada data deteksi untuk ditampilkan.*"
    else:
        table_md = "| No | Kelas Deteksi | Skor Keyakinan | Koordinat Box |\n"
        table_md += "| :--- | :--- | :--- | :--- |\n"
        for idx, det in enumerate(detections):
            ymin, xmin, ymax, xmax = det["box"]
            box_str = f"[{ymin:.2f}, {xmin:.2f}, {ymax:.2f}, {xmax:.2f}]"
            table_md += (
                f"| {idx+1} | **{det['label']}** | {det['score']:.2%} | `{box_str}` |\n"
            )

    return draw_img, alert_html, table_md


st.markdown(
    """
<style>
    .stApp { background: #0f172a; color: #f1f5f9; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #f1f5f9; }
    .stApp .stRadio label { color: #f1f5f9; }
    .stApp .stSlider label { color: #f1f5f9; }
    .stApp .stFileUploader label { color: #f1f5f9; }
    .stApp .stMarkdown { color: #f1f5f9; }
    .stButton button {
        background: #3b82f6; color: white; font-weight: bold;
        border-radius: 8px; border: none; padding: 0.5em 1em;
    }
    .stButton button:hover { background: #2563eb; color: white; }
    .header-box {
        text-align: center; margin-bottom: 25px;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white; padding: 25px; border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }
    .header-box h1 { margin: 0; font-size: 2.5em; font-weight: 800;
        text-transform: uppercase; letter-spacing: 1px; color: white; }
    .header-box p { margin: 8px 0 0 0; font-size: 1.1em; opacity: 0.9; color: white; }
    div[data-testid="stExpander"] {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 8px; margin-top: 10px;
    }
    div[data-testid="stExpander"] summary { color: #f1f5f9; font-weight: 600; }
    div[data-testid="stExpander"] .streamlit-expanderContent { background: #1e293b; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="header-box">
    <h1>Driver Drowsiness Detection System</h1>
    <p>Visualisasi & Pengujian Real-Time Model Object Detection TFLite (SSD MobileNet V2 FPN-Lite 320x320)</p>
</div>
""",
    unsafe_allow_html=True,
)

mode = st.radio(
    "Select Input Method",
    ["Upload Image", "Webcam Snapshot", "Live Webcam"],
    horizontal=True,
)

if mode == "Upload Image":
    col1, col2 = st.columns([5, 7])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )
        threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

        with st.expander("Try Demo Images"):
            demo_dir = "demo_images"
            if os.path.exists(demo_dir):
                for fname in sorted(os.listdir(demo_dir)):
                    if fname.endswith((".jpg", ".png")):
                        if st.button(f" {fname}"):
                            with open(
                                os.path.join(demo_dir, fname), "rb"
                            ) as f:
                                st.session_state.demo_img = f.read()

    img_bytes = None
    if uploaded_file is not None:
        img_bytes = uploaded_file.getvalue()
    elif "demo_img" in st.session_state:
        img_bytes = st.session_state.demo_img

    with col2:
        if img_bytes is not None:
            file_bytes = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result_img, alert_html, table_md = run_drowsiness_detection(
                image, threshold
            )

            st.markdown(alert_html, unsafe_allow_html=True)
            st.image(result_img, width="stretch")
            st.markdown("###  Detection Details")
            st.markdown(table_md)
        else:
            st.info(" Upload an image or select a demo image to begin.")

elif mode == "Webcam Snapshot":
    col1, col2 = st.columns([5, 7])

    with col1:
        camera_image = st.camera_input("Take a picture")
        threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

    with col2:
        if camera_image is not None:
            file_bytes = np.frombuffer(camera_image.getvalue(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result_img, alert_html, table_md = run_drowsiness_detection(
                image, threshold
            )

            st.markdown(alert_html, unsafe_allow_html=True)
            st.image(result_img, width="stretch")
            st.markdown("###  Detection Details")
            st.markdown(table_md)
        else:
            st.info(" Capture a photo using your webcam to begin.")

elif mode == "Live Webcam":
    threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

    try:
        from streamlit_webrtc import webrtc_streamer, VideoFrame

        RTC_CONFIG = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }

        class LiveDetector:
            def __init__(self):
                self.threshold = 0.5
                self.last_alert_time = 0
                self.mengantuk_count = 0
                self.tertidur_count = 0

            def process(self, frame: VideoFrame) -> np.ndarray:
                img_bgr = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                result_rgb, alert_html, _ = run_drowsiness_detection(
                    img_rgb, self.threshold
                )
                now = time.time()
                if "TERTIDUR" in alert_html:
                    self.tertidur_count += 1
                    self.mengantuk_count = 0
                    if self.tertidur_count >= 3 and now - self.last_alert_time > 2:
                        try:
                            import winsound
                            winsound.Beep(1200, 500)
                        except ImportError:
                            pass
                        self.last_alert_time = now
                        self.tertidur_count = 0
                elif "MENGANTUK" in alert_html:
                    self.mengantuk_count += 1
                    self.tertidur_count = 0
                    if self.mengantuk_count >= 3 and now - self.last_alert_time > 2:
                        try:
                            import winsound
                            winsound.Beep(800, 500)
                        except ImportError:
                            pass
                        self.last_alert_time = now
                        self.mengantuk_count = 0
                else:
                    self.mengantuk_count = 0
                    self.tertidur_count = 0
                return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        detector = LiveDetector()
        detector.threshold = threshold

        webrtc_streamer(
            key="drowsiness-live",
            video_frame_callback=detector.process,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
        )
    except ImportError:
        st.warning(
            " Live webcam requires `streamlit-webrtc`. "
            "Install it with: `pip install streamlit-webrtc`"
        )
        st.info(
            "Meanwhile, use **Upload Image** or **Webcam Snapshot** modes above."
        )
