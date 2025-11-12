import cv2
import numpy as np
import pandas as pd
import streamlit as st
import fitz # PyMuPDF
from PIL import Image
from io import BytesIO
import re
import pytesseract

# --- Configuration and Initialization ---
# NOTE for Linux Deployment: We rely on the system-wide installation of tesseract
# being in the PATH, thus TESSERACT_PATH and related configuration functions are removed.

# TESSERACT_LANGUAGES: Set to 'eng+mya' to enable both English and Myanmar OCR
# Ensure the user has installed 'tesseract-ocr-mya' or has configured Tesseract to
# recognize 'mya' language data in the Linux environment.
TESSERACT_LANGUAGES = 'eng+mya'

# Set the page configuration early
st.set_page_config(
    page_title="Document OCR Extractor (Eng + Myanmar)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define Normalized Region Coordinates (0-1000 scale) ---
# [x_min, y_min, x_max, y_max] for targeted extraction

# 1. Tightly defined regions, optimized for clean PDF-converted images
TARGET_REGIONS_PDF = {
    "Applicant Name": [50, 120, 480, 200],
    "Documentary Credit No.": [480, 120, 680, 200],
    "Original Credit Amount": [680, 120, 930, 200],
    "Contact Person / Tel": [50, 200, 480, 300],
    "Beneficiary Name": [50, 300, 480, 429], 
}

# 2. Expanded regions, with a buffer for minor misalignment in image scans/photos
TARGET_REGIONS_IMAGE = {
    # Expanded regions to allow for greater variance in alignment
    "Applicant Name": [50, 100, 500, 220], # Expanded vertically
    "Documentary Credit No.": [450, 100, 700, 220], # Expanded horizontally & vertically
    "Original Credit Amount": [650, 100, 950, 220], # Expanded horizontally & vertically
    "Contact Person / Tel": [50, 180, 500, 320], # Expanded vertically
    "Beneficiary Name": [50, 280, 500, 420], # Expanded vertically
}

# Global dictionary to hold the active regions (set in main())
TARGET_FIELD_REGIONS = {}


# --- Core Extraction Logic (Targeted by Region) ---
def extract_fields_by_region(image_array, active_regions):
    """ Extracts text for the specified fields using targeted regions defined by active_regions. """
    kv_data = {key: '-' for key in active_regions.keys()}
    H, W, _ = image_array.shape
    # Create a copy for drawing bounding boxes (must be RGB for PIL/Streamlit image display later)
    img_boxes = cv2.cvtColor(image_array.copy(), cv2.COLOR_BGR2RGB)

    for key, (x_min_norm, y_min_norm, x_max_norm, y_max_norm) in active_regions.items():
        # Denormalize coordinates
        x_min = int(x_min_norm * W / 1000)
        y_min = int(y_min_norm * H / 1000)
        x_max = int(x_max_norm * W / 1000)
        y_max = int(y_max_norm * H / 1000)

        # Crop and run Tesseract
        cropped_img = image_array[y_min:y_max, x_min:x_max]
        if cropped_img.size == 0: 
            # Draw empty box if crop failed
            cv2.rectangle(img_boxes, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3) # Use Blue for failed crop
            continue
        
        try:
            # Convert BGR (OpenCV) to RGB (Tesseract/Pillow standard) for OCR
            cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            
            text_raw = pytesseract.image_to_string(
                cropped_img_rgb, 
                lang=TESSERACT_LANGUAGES,
                config='--psm 6' # Assume a single uniform block of text
            ).strip()
        except pytesseract.TesseractNotFoundError:
            st.error("Tesseract is not installed or not in PATH. Check deployment configuration.")
            text_raw = ""
        except Exception as e:
            st.warning(f"OCR failed for field '{key}': {e}")
            text_raw = ""

        extracted_value = '-'
        if text_raw:
            # Simple value extraction logic
            lines = [line.strip() for line in text_raw.split('\n') if line.strip()]
            
            # Heuristic 1: Look for key and take the following text
            key_index = -1
            for i, line in enumerate(lines):
                # Check for the key or the first word of the key (case-insensitive)
                key_match_str = key.split(' ')[0].lower()
                if key_match_str in line.lower(): 
                    key_index = i
                    break

            if key_index != -1:
                value_lines = lines[key_index + 1:]
                if value_lines:
                    extracted_value = " ".join(value_lines)
                elif len(lines) == 1 and key.lower() in lines[0].lower():
                    # Heuristic 2: Same line key-value handling
                    key_end_index = lines[0].lower().find(key.lower()) + len(key)
                    # Remove common delimiters and spaces from the start of the value
                    value_on_same_line = re.sub(r'^\s*[:\.\-\—\s]', '', lines[0][key_end_index:].strip()).strip()
                    if value_on_same_line:
                        extracted_value = value_on_same_line
            
            # Fallback: use all text if key wasn't clearly found but text exists
            if extracted_value == '-' and len(lines) > 0 and key.split(' ')[0].lower() not in text_raw.lower():
                extracted_value = " ".join(lines) 

        # Post-process specific fields (e.g., currency)
        if key == "Original Credit Amount" and extracted_value != '-':
            # Find common patterns for currency amounts (digits, dots, commas, optional currency code)
            amount_match = re.search(r'([A-Z]{3,4}\s?)?([\d\.\,]+)', extracted_value.replace(' ', ''))
            if amount_match:
                # Capture currency code (if present) and amount part
                currency_code = amount_match.group(1).strip() if amount_match.group(1) else ''
                amount_str = amount_match.group(2).replace(',', '')
                try:
                    # Attempt standard float conversion and formatting
                    extracted_value = f"{currency_code} {float(amount_str):,.2f}"
                except ValueError:
                    # If conversion fails, keep the raw matched string
                    extracted_value = f"{currency_code} {amount_str}"
            else:
                # If no amount match, use the raw extracted text as a fallback
                extracted_value = extracted_value.replace('\n', ' ').strip()
                
        elif extracted_value != '-':
            extracted_value = extracted_value.replace('\n', ' ').strip()
            
        if extracted_value and extracted_value != '-':
            kv_data[key] = extracted_value.strip()

        # Draw the box for visualization (Red border)
        cv2.rectangle(img_boxes, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    img_with_boxes = Image.fromarray(img_boxes)
    kv_df_list = [{'Key Label (Form Text)': k, 'Extracted Value': v} for k, v in kv_data.items()]
    df_kv_pairs = pd.DataFrame(kv_df_list)
    
    return df_kv_pairs, img_with_boxes

# --- Full Text Extraction (Non-Structured) ---
def extract_full_text(image_array):
    """ Runs Tesseract on the entire image to get non-structured text. """
    with st.spinner("Extracting full page text (Non-Structured)..."):
        try:
            # Convert BGR (OpenCV) to RGB (Tesseract/Pillow standard) for OCR
            image_array_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            full_text = pytesseract.image_to_string(
                image_array_rgb,
                lang=TESSERACT_LANGUAGES
            ).strip()
            return full_text
        except pytesseract.TesseractNotFoundError:
            st.error("Tesseract is not installed or not in PATH. Cannot perform full text extraction.")
            return "Error: Tesseract not found."
        except Exception as e:
            st.error(f"Error during full text extraction: {e}")
            return f"Error: {e}"


# --- Utility Functions ---
def handle_file_upload(uploaded_file):
    """
    Handles file uploads, converting them to an OpenCV array and determining the file type.
    Returns: (img_array, file_type_str)
    """
    file_type = uploaded_file.type
    file_type_str = 'image' # Default to image
    
    try:
        file_bytes = uploaded_file.read()
        if 'pdf' in file_type:
            file_type_str = 'pdf'
            with st.spinner("Converting PDF page 1 to image (150 DPI)..."):
                # fitz (PyMuPDF) conversion logic
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                if doc.page_count == 0:
                    st.error("Could not process PDF. The document is empty or unreadable.")
                    return None, None
                page = doc.load_page(0)
                DPI = 150
                zoom_factor = DPI / 72
                matrix = fitz.Matrix(zoom_factor, zoom_factor)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                # Get image data buffer and reshape (pix.n is usually 3 for RGB)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                # Convert from RGB to BGR for OpenCV compatibility
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                doc.close()
                return img_array, file_type_str
        else:
            # Image file decoding logic
            img_array = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            return img_array, file_type_str
    except Exception as e:
        st.error(f"Error loading file. Check if it's a valid Image or non-encrypted PDF. Error details: {e}")
        return None, None

def get_download_button(data, is_dataframe, file_format, label, file_name_base, help_text=""):
    """Generates a common download button for different formats."""
    df = data if is_dataframe else None
    data_out = None
    mime = 'text/plain' 

    if file_format == 'csv' and is_dataframe:
        data_out = df.to_csv(index=False).encode('utf-8')
        mime = 'text/csv'
    elif file_format in ['txt', 'doc']:
        if is_dataframe:
            data_out = df.to_string(index=False).encode('utf-8')
        else: # For non-structured text (string data)
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
    
    st.title("🎯Document OCR Extractor (English & Myanmar)")
    st.markdown("This tool uses **Tesseract OCR (eng+mya)** to extract specific fields and the full page text from a document. Ensure your deployment environment has Tesseract and the Myanmar language pack installed.")

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
        st.info(f"File **'{uploaded_file.name}'** uploaded. Starting file conversion...")
        image_array, file_type = handle_file_upload(uploaded_file)

    # --- OCR Processing and Results Display ---
    if image_array is not None and file_type is not None:
        
        # --- DYNAMIC REGION SELECTION ---
        global TARGET_FIELD_REGIONS # Use the global variable
        if file_type == 'pdf':
            TARGET_FIELD_REGIONS = TARGET_REGIONS_PDF
            st.success(f"**PDF Mode:** Using **Tightly Defined Regions** for extraction.")
        else:
            TARGET_FIELD_REGIONS = TARGET_REGIONS_IMAGE
            st.warning(f"**Image/Scan Mode:** Using **Expanded Regions** for extraction to account for misalignment.")
        
        st.subheader("2. OCR Processing and Result Formats")
        
        # Run both extractions with the dynamically selected regions
        df_kv_pairs, img_with_boxes = extract_fields_by_region(image_array, TARGET_FIELD_REGIONS)
        full_text = extract_full_text(image_array)

        col_img, col_data_tabs = st.columns([1, 2])
        
        with col_img:
            st.markdown("### 🖼️ OCR Visualization")
            st.image(img_with_boxes, caption=f"Targeted extraction regions ({file_type.upper()} Mode - Red Boxes)", use_column_width=True)
            
        with col_data_tabs:
            # Create two tabs: one for Structured, one for Non-Structured
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
                    get_download_button(df_kv_pairs, True, 'csv', "📥 Download CSV", 'structured_key_value_pairs')
                with col_txt:
                    get_download_button(df_kv_pairs, True, 'txt', "📥 Download TXT", 'structured_key_value_pairs')
                with col_word:
                    get_download_button(df_kv_pairs, True, 'doc', "📥 Download DOC", 'structured_key_value_pairs', help_text="Saves the table data as a text file with a .doc extension.")
            
            with tab_non_structured:
                st.markdown("### Full Extracted Text (Sorted by Read Order)")
                st.text_area(
                    label="Non-Structured Text",
                    value=full_text,
                    height=400,
                    help="This is the raw text output from Tesseract across the entire image."
                )
                
                st.markdown("#### Download Options (Full Text)")
                col_txt_full, col_word_full, _ = st.columns(3)
                with col_txt_full:
                    get_download_button(full_text, False, 'txt', "📥 Download TXT", 'full_extracted_text')
                with col_word_full:
                    get_download_button(full_text, False, 'doc', "📥 Download DOC", 'full_extracted_text', help_text="Saves the full text as a text file with a .doc extension.")
                
        st.markdown("---")

if __name__ == '__main__':
    main()
