# AffoDent Oral Screening App

**AffoDent Oral Screening App** is a Streamlit-based AI-powered dental screening application that detects various dental conditions from uploaded photos using a custom YOLOv5 model (`best.pt`).

---

## Features

- Detects dental conditions such as:
  - Ulcers
  - Lesions
  - Caries (cavities)
  - Stains
  - Calculus
  - Missing teeth
  - Broken teeth
  - Root stamps

- Color-coded bounding boxes on images with labels and simulated tooth numbers.
- Generates a detailed PDF report with patient info, findings, and annotated images.
- Simple and intuitive interface for dental professionals.
- Runs locally or deployed via Streamlit Community Cloud.

---

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/deepdentalapp.git
   cd deepdentalapp
