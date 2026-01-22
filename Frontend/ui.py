import streamlit as st
import requests
from PIL import Image
import base64
import io
import re
from collections import Counter
import os

# ================= CONFIG =================
API_URL = os.getenv("API_URL", "http://backend:8000/detect")

st.set_page_config(
    page_title="Vehicle Detection System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= GLOBAL STYLES =================
st.markdown(
    """
    <style>
        body {
            background-color: #0f172a;
        }
        .card {
            background-color: #1e293b;
            padding: 18px;
            border-radius: 14px;
            margin-bottom: 16px;
        }
        .title {
            color: white;
            font-size: 32px;
            font-weight: 700;
        }
        .subtitle {
            color: #cbd5e1;
            font-size: 16px;
        }
        .section-title {
            color: white;
            font-size: 20px;
            margin-bottom: 10px;
        }
        .stButton button {
            width: 100%;
            height: 3em;
            border-radius: 10px;
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ================= HEADER =================
st.markdown(
    """
    <div class="card">
        <div class="title">Vehicle Detection System</div>
        <div class="subtitle">
            YOLO-powered real-time vehicle detection
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ================= LAYOUT =================
left_col, right_col = st.columns([1, 1.2], gap="large")

# ================= LEFT: UPLOAD =================
with left_col:
    st.markdown(
        """
        <div class="card">
            <div class="section-title">📤 Upload Image</div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")

        st.markdown(
            """
            <div class="card">
                <div class="section-title">🖼️ Input Image</div>
            """,
            unsafe_allow_html=True
        )

        st.image(input_image,width=550)

        st.markdown("</div>", unsafe_allow_html=True)

        detect_btn = st.button("Detect Vehicles")

# ================= RIGHT: RESULTS =================
with right_col:
    if uploaded_file and detect_btn:
        with st.spinner("Detecting vehicles..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            try:
                response = requests.post(API_URL, files=files, timeout=60)
                response.raise_for_status()
            except Exception as e:
                st.error(f"API Error: {e}")
                st.stop()

            result = response.json()

        # -------- Detected Image --------
        st.markdown(
            """
            <div class="card">
                <div class="section-title">Detection Output</div>
            """,
            unsafe_allow_html=True
        )

        encoded_image = result.get("image")

        if encoded_image:
            image_bytes = base64.b64decode(encoded_image)
            detected_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.image(detected_image, width=550)
        else:
            st.warning("No output image returned")

        st.markdown("</div>", unsafe_allow_html=True)

        # -------- Vehicle Summary --------
        st.markdown(
            """
            <div class="card">
                <div class="section-title"> Vehicle Count Summary</div>
            """,
            unsafe_allow_html=True
        )

        detections = result.get("detections", [])

        if detections:
            vehicle_classes = [
                re.findall(r"([a-zA-Z]+)", det)[0]
                for det in detections
            ]

            vehicle_counts = Counter(vehicle_classes)

            m1, m2 = st.columns(2)
            m1.metric(" Total Vehicles", sum(vehicle_counts.values()))
            m2.metric(" Vehicle Types", len(vehicle_counts))

            count_table = [
                {"Vehicle Type": k.capitalize(), "Count": v}
                for k, v in vehicle_counts.items()
            ]

            st.table(count_table)
        else:
            st.info("No vehicles detected")

        st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown(
    """
    <div style="text-align:center; color:#94a3b8; margin-top:30px;">
        Built with Streamlit • YOLO • FastAPI
    </div>
    """,
    unsafe_allow_html=True
)
