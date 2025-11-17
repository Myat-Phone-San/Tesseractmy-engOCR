import cv2
import numpy as np
import pandas as pd
import streamlit as st
import fitz # PyMuPDF
from PIL import Image
import re
import pytesseract
import os

# --- Configuration and Initialization ---
# Tesseract executable path (as provided by the user)
TESSERACT_PATH = r"C:\Users\myatphonesan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# TESSERACT_LANGUAGES: Set to 'eng+mya' to enable both English and Myanmar OCR
TESSERACT_LANGUAGES = 'eng+mya'

def configure_tesseract():
    """Sets the Tesseract command path."""
    try:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    except Exception as e:
        st.error(f"Error setting Tesseract path. Please check if Tesseract is installed at: {TESSERACT_PATH}. Error details: {e}")
        st.stop()

# Set the page configuration early
st.set_page_config(
    page_title="Myanmar DL OCR Extractor (Final)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define Normalized Region Coordinates (0-1000 scale) for Driving Licence ---
# These fixed regions assume the document is in a landscape (horizontal) orientation 
# and correctly upright.
TARGET_FIELD_REGIONS = {
    # Number (No.)
    "Number (No.)": [240, 270, 500, 370],
    # Name
    "Name": [247, 420, 600, 480],
    # NRC No. (N.R.C No.)
    "NRC No. (N.R.C No.)": [247, 540, 650, 630],
    # Date of Birth
    "Date of Birth": [247, 660, 650, 750],
    # Blood Type
    "Blood Type": [250, 795, 300, 850],
    # Valid Up To
    "Valid Up To": [255, 920, 450, 980],
}

# --- Rotation Correction based on Aspect Ratio ---
def correct_aspect_rotation(image_array):
    """
    Attempts to auto-rotate the image to a horizontal (landscape) orientation 
    if the height is greater than the width.
    """
    H, W, _ = image_array.shape
    
    # A standard driving license is a landscape (W > H) document.
    if H > W:
        # Use an internal print/log instead of Streamlit info for a cleaner UI
        # print("Aspect ratio suggests Portrait (H > W). Rotating image 90 degrees clockwise to landscape.") 
        image_array = cv2.rotate(image_array, cv2.ROTATE_90_CLOCKWISE)
        
    return image_array

# --- Perspective Correction (Deskew/Warp) ---
def warp_document(image_array, display_status=False):
    """
    Finds the largest rectangular contour (the document) and performs 
    a perspective transformation to create a top-down, straightened view.
    """
    # 1. Preprocessing
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    # 2. Find Contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    screenCnt = None
    
    # 3. Find the largest four-point contour (the document)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Check for 4 points and ensure the contour is reasonably large
        if len(approx) == 4 and cv2.contourArea(c) > 5000:
            screenCnt = approx
            break
            
    if screenCnt is None:
        # Return a warning message to the main logic instead of using st.warning here
        return image_array, image_array, "Warping skipped: Document outline not found." 

    # 4. Perform Perspective Transformation 
    pts = screenCnt.reshape(4, 2).astype(np.float32)
    
    # Stable sorting based on position: Top-Left (smallest sum), Bottom-Right (largest sum)
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)] 
    rect[2] = pts[np.argmax(s)] 
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)] 
    rect[3] = pts[np.argmax(diff)] 
    
    (tl, tr, br, bl) = rect

    # Calculate the new width and height
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_array, M, (maxWidth, maxHeight))
    
    img_contour = image_array.copy()
    cv2.drawContours(img_contour, [screenCnt], -1, (0, 255, 0), 20) 

    return warped, img_contour, "Warping successful."


# --- Core Extraction Logic (Targeted by Region) ---
def extract_fields_by_region(image_array, active_regions):
    """ Extracts text for the specified fields using targeted regions defined by active_regions. """
    kv_data = {key: 'N/A' for key in active_regions.keys()}
    H, W, _ = image_array.shape
    img_boxes = cv2.cvtColor(image_array.copy(), cv2.COLOR_BGR2RGB)

    for key, (x_min_norm, y_min_norm, x_max_norm, y_max_norm) in active_regions.items():
        x_min = int(x_min_norm * W / 1000)
        y_min = int(y_min_norm * H / 1000)
        x_max = int(x_max_norm * W / 1000)
        y_max = int(y_max_norm * H / 1000)

        cropped_img = image_array[y_min:y_max, x_min:x_max]
        if cropped_img.size == 0: continue
        
        # Use PSM 7 for single line extraction of fields
        psm_config = '--psm 7' 

        text_raw = pytesseract.image_to_string(
            cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), 
            lang=TESSERACT_LANGUAGES, 
            config=psm_config
        ).strip()
        
        extracted_value = 'N/A'
        if text_raw:
            extracted_value = re.sub(r'[\n\s]+', ' ', text_raw).strip()

            # Specific cleaning for NRC No.
            if key == "NRC No. (N.R.C No.)":
                nrc_pattern = r'(\d+[\/\\-][a-zA-Z\s\u1000-\u109F]+[\(\[][a-zA-Z\s\u1000-\u109F]+[\)\]]\d+)'
                nrc_match = re.search(nrc_pattern, extracted_value.replace(' ', ''))
                if nrc_match:
                    extracted_value = nrc_match.group(1).replace('/', '/') 
                else:
                    extracted_value = extracted_value
            
            # Simple cleaning for date fields
            elif "Date" in key or "Valid" in key:
                 extracted_value = re.sub(r'[^\d\s\-\/]', '', extracted_value).strip()


        if extracted_value and extracted_value != 'N/A':
            kv_data[key] = extracted_value.strip()

        cv2.rectangle(img_boxes, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    img_with_boxes = Image.fromarray(img_boxes)
    kv_df_list = [{'Key Label (Form Text)': k, 'Extracted Value': v} for k, v in kv_data.items()]
    df_kv_pairs = pd.DataFrame(kv_df_list)
    
    return df_kv_pairs, img_with_boxes

# --- Full Text Extraction (Non-Structured) ---
def extract_full_text(image_array):
    """ Runs Tesseract on the entire image to get non-structured text using all configured languages. """
    with st.spinner("Extracting full page text (Non-Structured)..."):
        full_text = pytesseract.image_to_string(
            cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB),
            lang=TESSERACT_LANGUAGES 
        ).strip()
    return full_text

# --- Utility Functions (File Handling) ---
def handle_file_upload(uploaded_file):
    """ Handles file uploads, converting them to an OpenCV array. """
    file_type = uploaded_file.type
    file_type_str = 'image'
    
    try:
        file_bytes = uploaded_file.read()
        if 'pdf' in file_type:
            file_type_str = 'pdf'
            with st.spinner("Converting PDF page 1 to image (150 DPI)..."):
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                if doc.page_count == 0:
                    st.error("Could not process PDF. The document is empty or unreadable.")
                    return None, None
                page = doc.load_page(0)
                DPI = 150
                zoom_factor = DPI / 72
                matrix = fitz.Matrix(zoom_factor, zoom_factor)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                doc.close()
                return img_array, file_type_str
        else:
            img_array = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            return img_array, file_type_str
    except Exception as e:
        st.error(f"Error loading file. Check if it's a valid Image or non-encrypted PDF. Error details: {e}")
        return None, None

def get_download_button(data, is_dataframe, file_format, label, file_name_base, help_text=""):
    """Generates a common download button for different formats."""
    df = data
    if file_format == 'csv' and is_dataframe:
        data_out = df.to_csv(index=False).encode('utf-8')
        mime = 'text/csv'
    elif file_format in ['txt', 'doc']:
        if is_dataframe:
            data_out = df.to_string(index=False).encode('utf-8')
        else: # For non-structured text
            data_out = data.encode('utf-8')
        mime = 'text/plain' 
    else:
        return 

    final_name = f'{file_name_base}.{file_format}'
    st.download_button(
        label=label,
        data=data_out,
        file_name=final_name,
        mime=mime,
        help=help_text
    )

# --- Streamlit Application Layout ---
def main():
    # Configure Tesseract path first
    configure_tesseract()
    
    st.title("Myanmar Driving License OCR Extractor")

    # 1. File Upload
    uploaded_file = st.file_uploader(
        "Choose a Document File (Image or PDF)",
        type=['jpg', 'jpeg', 'png', 'pdf'],
        help="For multi-page PDFs, only the first page will be processed."
    )
    st.markdown("---")

    image_array = None
    file_type = None
    
    if uploaded_file is not None:
        # Removed st.info() about file upload
        image_array, file_type = handle_file_upload(uploaded_file)

    # --- OCR Processing and Results Display ---
    if image_array is not None and file_type is not None:
        
        st.subheader("2. Preprocessing: Rotation and Perspective Correction")
        
        # 2. Correct 90/270 rotation based on Aspect Ratio (Ensures Landscape)
        rotated_image = correct_aspect_rotation(image_array.copy())
        
        # 3. Perspective Warp (Deskew)
        # The third return value captures the status/warning message
        warped_image, image_with_contour, warp_status = warp_document(rotated_image)
        
        # Display warp status/warning only if it's not a success message
        if "Warping skipped" in warp_status:
             st.warning(warp_status)

        # Run extractions on the fully corrected image
        df_kv_pairs, img_with_boxes = extract_fields_by_region(warped_image, TARGET_FIELD_REGIONS)
        full_text = extract_full_text(warped_image)

        col_contour, col_warped = st.columns(2)
        
        with col_contour:
             st.markdown("### 🖼️ Document after Aspect Rotation (Outline Detected)")
             # Used use_container_width=True to remove the warning
             st.image(cv2.cvtColor(image_with_contour, cv2.COLOR_BGR2RGB), caption="Document after rotation correction. Detected Outline (Green)", use_container_width=True)
             
        with col_warped:
            st.markdown("### 📐 Final Warped Image with Extracted Regions")
             # Used use_container_width=True to remove the warning
            st.image(img_with_boxes, caption="Document after full correction (Red Boxes = Fixed Regions Applied)", use_container_width=True)


        st.markdown("---")
        
        st.subheader("3. OCR Extraction Results")

        # Results Display
        tab_structured, tab_non_structured = st.tabs(["📄 Structured Table", "📋 All Non-Structured Text"])

        with tab_structured:
            st.markdown("### 🔑 Extracted Key-Value Pairs")
            st.dataframe(
                df_kv_pairs[['Key Label (Form Text)', 'Extracted Value']], 
                use_container_width=True, 
                hide_index=True
            )
            
            st.markdown("#### Download Options (Table)")
            col_csv, col_txt, col_word = st.columns(3)
            with col_csv:
                get_download_button(df_kv_pairs, True, 'csv', "📥 Download CSV", 'dl_extracted_key_value_pairs')
            with col_txt:
                get_download_button(df_kv_pairs, True, 'txt', "📥 Download TXT", 'dl_extracted_key_value_pairs')
            with col_word:
                get_download_button(df_kv_pairs, True, 'doc', "📥 Download DOC", 'dl_extracted_key_value_pairs', help_text="Saves the table data as a text file with a .doc extension.")
        
        with tab_non_structured:
            st.markdown("### Full Extracted Text (Eng + Mya) from Warped Document")
            st.text_area(
                label="Non-Structured Text",
                value=full_text,
                height=400,
                help="This is the raw text output from Tesseract using both English and Myanmar languages, extracted from the corrected image."
            )
            
            st.markdown("#### Download Options (Full Text)")
            col_txt_full, col_word_full, _ = st.columns(3)
            with col_txt_full:
                get_download_button(full_text, False, 'txt', "📥 Download TXT", 'dl_full_extracted_text')
            with col_word_full:
                get_download_button(full_text, False, 'doc', "📥 Download DOC", 'dl_full_extracted_text', help_text="Saves the full text as a text file with a .doc extension.")
            
        st.markdown("---")

if __name__ == '__main__':
    main()
