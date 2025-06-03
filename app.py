import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io

# Load the model once and cache it
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# Define color codes for each class (example)
COLOR_MAP = {
    "caries": "red",
    "broken_tooth": "orange",
    "missing_tooth": "blue",
    "oral_lesion": "green",
    "oral_ulcer": "purple",
    "calculus": "yellow",
    "stain": "brown",
    # add more classes as per your model
}

# Dummy tooth numbering for demo (you should map by bbox positions)
TOOTH_NUMBERS = {
    0: "11", 1: "12", 2: "13", 3: "14", 4: "15",
    5: "21", 6: "22", 7: "23", 8: "24", 9: "25",
    # Extend as needed
}

def draw_boxes(image, results):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    
    for i, box in enumerate(results.boxes):
        cls = results.names[int(box.cls[0])]
        color = COLOR_MAP.get(cls, "white")
        
        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Add label with class and tooth number if available
        tooth_num = TOOTH_NUMBERS.get(i, "?")
        label = f"{cls} (Tooth {tooth_num})"
        text_size = draw.textsize(label, font=font)
        
        # Draw label background
        draw.rectangle([x1, y1 - text_size[1], x1 + text_size[0], y1], fill=color)
        # Draw label text
        draw.text((x1, y1 - text_size[1]), label, fill="black", font=font)
    return img

st.title("AffoDent Oral Screening App")

uploaded_file = st.file_uploader("Upload an intraoral photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # Run detection
    results = model(image)

    # Draw boxes and tooth numbers
    annotated_img = draw_boxes(image, results[0])

    st.image(annotated_img, caption="Detected Dental Conditions", use_column_width=True)

    # You can add PDF generation or detailed report here
