import streamlit as st
import requests
from PIL import Image
import os
import re
from collections import Counter

# ================= CONFIG =================
API_URL = "http://127.0.0.1:8000/detect"

st.set_page_config(
    page_title="Vehicle Detection",
    layout="centered"
)




# ================= HEADER =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.title(" Vehicle Detection System")
st.write("YOLO-powered vehicle detection and counting dashboard")
st.markdown('</div>', unsafe_allow_html=True)

# ================= UPLOAD =================
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    " Upload an image",
    type=["jpg", "jpeg", "png"]
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # ================= INPUT IMAGE =================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(" Input Image")
    st.image(image, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    # ================= DETECT BUTTON =================
    if st.button("Detect Vehicles", width="stretch"):
        with st.spinner("Detecting vehicles..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()

                # ================= OUTPUT IMAGE =================
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader(" Detected Image")

                detected_path = result.get("output_image")
                if detected_path and os.path.exists(detected_path):
                    detected_img = Image.open(detected_path)
                    st.image(detected_img, width="stretch")
                else:
                    st.warning("Detected image not found")

                st.markdown('</div>', unsafe_allow_html=True)

                # ================= VEHICLE COUNT =================
                raw_detections = result.get("detections", "")

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("🚦 Vehicle Count Summary")

                if raw_detections:
                    vehicle_classes = re.findall(
                        r"([a-zA-Z]+)\s*\(",
                        raw_detections
                    )

                    vehicle_counts = Counter(vehicle_classes)

                    st.metric(
                        "Total Vehicles Detected",
                        sum(vehicle_counts.values())
                    )

                    count_table = [
                        {
                            "Vehicle Type": k.capitalize(),
                            "Count": v
                        }
                        for k, v in vehicle_counts.items()
                    ]

                    st.table(count_table)

                else:
                    st.info("No vehicles detected")

                st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.error("API Error")
                st.text(response.text)
