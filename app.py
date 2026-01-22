import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os
from PIL import Image
import re
import requests # Added for downloading the model

# --- Configuration --- #
# Direct URL to the raw best.pt file on GitHub
GITHUB_MODEL_URL = "https://raw.githubusercontent.com/FaithKinya/winesandspirits-label-detection/main/runs/detect/bottle_label_detection/weights/best.pt"
LOCAL_MODEL_PATH = "best.pt" # Path where the model will be saved locally in the Streamlit app container

# Function to download the model if it doesn't exist locally
def download_model_if_not_exists(url, local_path):
    if not os.path.exists(local_path):
        st.write(f"Downloading model from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise an exception for HTTP errors
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.write("Model downloaded successfully!")
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading model: {e}")
            st.stop() # Stop the app if model cannot be downloaded
    else:
        st.write("Model already exists locally.")

# Download the model at app startup
download_model_if_not_exists(GITHUB_MODEL_URL, LOCAL_MODEL_PATH)

# Initialize YOLO model (load only once)
@st.cache_resource
def load_yolo_model():
    return YOLO(LOCAL_MODEL_PATH)

# Initialize EasyOCR reader (load only once)
@st.cache_resource
def load_ocr_reader():
    # Using CPU for EasyOCR; change gpu=True if your deployment environment has a GPU
    return easyocr.Reader(['en'], gpu=False)

# Load the models
model = load_yolo_model()
reader = load_ocr_reader()

# Define the brand correction dictionary (copy-pasted from your notebook)
brand_corrections = {
    'Heineken': 'HEINEKEN',
    'HEINEKEN': 'HEINEKEN',
    'ROYAL': 'ROYAL CHALLENGE',
    'Royal': 'ROYAL CHALLENGE',
    'ROYAL CHALLENGE': 'ROYAL CHALLENGE',
    'Old Chef': 'OLD CHEF',
    'OldChef': 'OLD CHEF',
    'OLD CHEF': 'OLD CHEF',
    'Old': 'OLD CHEF',
    'BACARDi': 'BACARDI',
    'BACARDI': 'BACARDI',
    'cartA': 'BACARDI',
    'AMRUT': 'AMRUT',
    'BAGPIPER': 'BAGPIPER',
    'Sealdea': 'BAGPIPER',
    'PPRodUCE Of': 'PPRodUCEOF',
    'PPRodUCEOf': 'PPRodUCEOF',
    'PPRodUCEOF': 'PPRodUCEOF',
    'ud': 'PPRodUCEOF',
    'LAGER Brep': 'LAGER',
    '<ektn': 'LAGER',
    'PPEMIU@': 'PREMIUM',
    'QUAL Y': 'QUALITY',
    'QuAl<': 'QUALITY',
    'Smd': 'SMIRNOFF',
    'IsMRNDFf': 'SMIRNOFF',
    'EF': 'SMIRNOFF',
    'ghpe SDisvild': 'SMIRNOFF',
    '3M': '3M',
    'IMPERIAL': 'IMPERIAL',
    'AMPERIAL': 'IMPERIAL',
    'WPER': 'WHISKY',
    'TrOm': 'UNKNOWN',
    'gume': 'UNKNOWN',
    'B@OMi': 'UNKNOWN',
    'Fhcm': 'UNKNOWN',
    'sirUNG': 'STRONG',
    'Mon': 'UNKNOWN',
    'BIRAC': 'UNKNOWN',
    'Nol': 'MCDOWELLS',
    'McDowells': 'MCDOWELLS',
    'MD DOWELLS': 'MCDOWELLS',
    'Unknown': 'UNKNOWN',
    'JNGT': 'UNKNOWN',
    'ERL': 'UNKNOWN',
    'CK': 'UNKNOWN',
    'GIMBA': 'UNKNOWN',
    'RON': 'UNKNOWN',
    'VEMOO': 'UNKNOWN',
    'STAG': 'STAG',
    'ROYA': 'ROYAL STAG'
}

# Pre-process brand_corrections for efficient lookup
corrected_mapping = {}
for variant, standardized_name in brand_corrections.items():
    if standardized_name not in corrected_mapping:
        corrected_mapping[standardized_name] = []
    corrected_mapping[standardized_name].append(variant)

# Define the clean_and_classify_text function
def clean_and_classify_text(ocr_texts, brand_corrections_map):

    def get_standardized_brand(text_to_classify):
        if not isinstance(text_to_classify, str):
            return 'UNKNOWN'

        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text_to_classify).upper().strip()
        if not cleaned_text:
            return 'UNKNOWN'

        for standardized_name, variants in brand_corrections_map.items():
            for variant in variants:
                cleaned_variant = re.sub(r'[^a-zA-Z0-9\s]', '', variant).upper().strip()
                if cleaned_text == cleaned_variant or cleaned_text in cleaned_variant or cleaned_variant in cleaned_text:
                    return standardized_name

        tokens = cleaned_text.split()
        for token in tokens:
            if len(token) < 2:
                continue
            for standardized_name, variants in brand_corrections_map.items():
                for variant in variants:
                    cleaned_variant = re.sub(r'[^a-zA-Z0-9\s]', '', variant).upper().strip()
                    if token == cleaned_variant or token in cleaned_variant or cleaned_variant in token:
                        return standardized_name
        return 'UNKNOWN'

    best_match = 'UNKNOWN'
    for text in ocr_texts:
        classified = get_standardized_brand(text)
        if classified != 'UNKNOWN':
            best_match = classified
            break
    return best_match

# --- Streamlit App Layout --- #
st.set_page_config(page_title="Bottle Label Detector with OCR", layout="wide")
st.title("Bottle Label Detector with OCR")
st.markdown("Upload an image of a bottle to detect labels and extract text.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image using PIL (for Streamlit compatibility) then convert to OpenCV format
    image_pil = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image_pil)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.image(image_pil, caption="Uploaded Image", use_column_width=True)
    st.write("")

    st.subheader("Processing Image...")

    # Perform object detection
    # verbose=False suppresses logging messages which is good for Streamlit deployment
    results = model.predict(img_cv2, conf=0.25, verbose=False)

    if results and results[0].boxes:
        annotated_image = img_cv2.copy()
        detections_summary = []

        for i, box in enumerate(results[0].boxes.data):
            xmin, ymin, xmax, ymax = map(int, box[:4])

            # Crop the detected region
            cropped_img = annotated_image[ymin:ymax, xmin:xmax]

            # Perform OCR
            ocr_detections = reader.readtext(cropped_img)
            raw_ocr_texts = [detection[1] for detection in ocr_detections]
            classified_brand = clean_and_classify_text(raw_ocr_texts, corrected_mapping)

            # Draw bounding box
            cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Annotate with OCR text and classified brand
            display_raw_ocr = ', '.join(raw_ocr_texts)
            text_info = f"OCR: {display_raw_ocr}"
            brand_info = f"Brand: {classified_brand}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            text_color = (0, 0, 255) # Blue
            brand_color = (255, 0, 0) # Red
            line_height = 25

            # Put OCR text
            cv2.putText(annotated_image, text_info, (xmin, ymin - line_height - 10),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            # Put Classified Brand
            cv2.putText(annotated_image, brand_info, (xmin, ymin - 10),
                        font, font_scale, brand_color, font_thickness, cv2.LINE_AA)

            detections_summary.append({
                "Bounding Box": f"[{xmin}, {ymin}, {xmax}, {ymax}]",
                "Raw OCR Text": display_raw_ocr,
                "Classified Brand": classified_brand
            })

        st.subheader("Detected Labels and OCR Results")
        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Image with Detections", use_column_width=True)

        if detections_summary:
            st.subheader("Detailed Results")
            st.table(detections_summary)
        else:
            st.warning("No labels detected in the image.")
    else:
        st.warning("No bottle labels found in the uploaded image.")
else:
    st.info("Please upload an image to get started.")