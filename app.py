import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile
import os

# Load YOLOv5 model from best.pt
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    model.conf = 0.25
    return model

model = load_model()

# Tooth number estimation based on horizontal image location
def get_tooth_number(x_center):
    if x_center < 200:
        return "Tooth 18-14"
    elif x_center < 400:
        return "Tooth 13-23"
    elif x_center < 600:
        return "Tooth 24-28"
    else:
        return "Tooth 38-34"

# Color mapping for conditions
color_map = {
    "ulcer": (255, 0, 0),         # Red
    "lesion": (0, 255, 0),        # Green
    "caries": (255, 255, 0),      # Yellow
    "stain": (0, 0, 255),         # Blue
    "calculus": (255, 165, 0),    # Orange
    "missing": (255, 255, 255),   # White
    "broken": (128, 0, 128),      # Purple
    "root_stamp": (0, 255, 255)   # Cyan
}

# Normalize label key
def normalize_label(label):
    return label.lower().replace(" ", "_")

# Generate PDF Report
def generate_pdf(patient_info, results, image_paths):
    filename = f"{patient_info['name'].replace(' ', '_')}_report.pdf"
    filepath = os.path.join(tempfile.gettempdir(), filename)
    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "AffoDent Oral Screening Report")

    c.setFont("Helvetica", 12)
    y = height - 90
    for key, value in patient_info.items():
        c.drawString(50, y, f"{key.capitalize()}: {value}")
        y -= 20

    y -= 10
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Findings:")
    y -= 20

    if not results:
        c.drawString(60, y, "No abnormalities detected.")
    else:
        for item in results:
            c.drawString(60, y, f"- {item['label']} at {item['tooth_number']} (Confidence: {item['conf']:.2f})")
            y -= 18

    for img_path in image_paths:
        c.showPage()
        c.drawImage(img_path, 50, 250, width=500, preserveAspectRatio=True, mask='auto')

    c.save()
    return filepath

# Streamlit Interface
st.set_page_config(page_title="AffoDent Oral Screening", layout="centered")
st.title("AffoDent Oral Screening App")

st.markdown("Please enter patient information and upload dental photographs.")

with st.form("patient_info_form"):
    name = st.text_input("Patient Name")
    age = st.text_input("Age")
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    complaint = st.text_input("Chief Complaint")
    history = st.text_area("Medical History")
    submitted = st.form_submit_button("Submit")

if submitted:
    st.success("Patient data saved. Now upload at least 2 dental photos.")

    uploaded_images = st.file_uploader(
        "Upload 2 to 6 Dental Images (Frontal, Lateral, Occlusal, Tongue, Palate)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_images:
        results_summary = []
        temp_image_paths = []

        for uploaded_file in uploaded_images:
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
            results = model(img_array)
            preds = results.xyxy[0]

            for *box, conf, cls in preds:
                x1, y1, x2, y2 = map(int, box)
                x_center = (x1 + x2) // 2
                label = model.names[int(cls)].lower()
                norm_label = normalize_label(label)
                tooth_number = get_tooth_number(x_center)

                results_summary.append({
                    "label": label,
                    "conf": float(conf),
                    "tooth_number": tooth_number
                })

                color = color_map.get(norm_label, (200, 200, 200))
                cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
                text = f"{label} ({tooth_number})"
                cv2.putText(img_array, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            temp_img_path = os.path.join(tempfile.gettempdir(), f"annotated_{uploaded_file.name}")
            cv2.imwrite(temp_img_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            temp_image_paths.append(temp_img_path)

            st.image(img_array, caption=f"Annotated: {uploaded_file.name}", use_column_width=True)

        patient_data = {
            "name": name,
            "age": age,
            "sex": sex,
            "complaint": complaint,
            "history": history
        }

        pdf_path = generate_pdf(patient_data, results_summary, temp_image_paths)
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name=os.path.basename(pdf_path))
