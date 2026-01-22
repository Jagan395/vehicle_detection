import streamlit as st
import requests
from PIL import Image
import base64
import io
import re
from collections import Counter

# ================= CONFIG =================
API_URL = "http://backend:8000/detect"
# For local run (no Docker), comment above and use:
# API_URL = "http://127.0.0.1:8000/detect"

st.set_page_config(
    page_title="Vehicle Detection",
    layout="centered"
)

# ================= HEADER =================
st.markdown(
    """
    <div style="background-color:#1e293b; padding:20px; border-radius:16px;">
        <h1 style="color:white;">Vehicle Detection System</h1>
        <p style="color:#cbd5e1;">YOLO-powered vehicle detection and counting dashboard</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ================= UPLOAD =================
st.markdown(
    """
    <div style="background-color:#1e293b; padding:20px; border-radius:16px; margin-top:20px;">
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    " Upload an image",
    type=["jpg", "jpeg", "png"]
)

st.markdown("</div>", unsafe_allow_html=True)

# ================= MAIN LOGIC =================
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)

    # -------- Input Image --------
    st.markdown(
        """
        <div style="background-color:#1e293b; padding:20px; border-radius:16px; margin-top:20px;">
            <h3 style="color:white;"> Input Image</h3>
        """,
        unsafe_allow_html=True
    )

    st.image(input_image, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    # -------- Detect Button --------
    if st.button("🚦 Detect Vehicles", width="stretch"):
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
            except Exception as e:
                st.error(f"API connection failed: {e}")
                st.stop()

            if response.status_code != 200:
                st.error("Backend error")
                st.text(response.text)
                st.stop()

            result = response.json()

            # -------- Detected Image --------
            st.markdown(
                """
                <div style="background-color:#1e293b; padding:20px; border-radius:16px; margin-top:20px;">
                    <h3 style="color:white;">📸 Detected Image</h3>
                """,
                unsafe_allow_html=True
            )

            encoded_image = result.get("image")

            if encoded_image:
                image_bytes = base64.b64decode(encoded_image)
                detected_image = Image.open(io.BytesIO(image_bytes))
                st.image(detected_image, width="stretch")
            else:
                st.warning("Detected image not returned by API")

            st.markdown("</div>", unsafe_allow_html=True)

            # -------- Vehicle Count Summary --------
            st.markdown(
                """
                <div style="background-color:#1e293b; padding:20px; border-radius:16px; margin-top:20px;">
                    <h3 style="color:white;"> Vehicle Count Summary</h3>
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

                st.metric(
                    "Total Vehicles Detected",
                    sum(vehicle_counts.values())
                )

                count_table = [
                    {"Vehicle Type": k.capitalize(), "Count": v}
                    for k, v in vehicle_counts.items()
                ]

                st.table(count_table)
            else:
                st.info("No vehicles detected")

            st.markdown("</div>", unsafe_allow_html=True)
