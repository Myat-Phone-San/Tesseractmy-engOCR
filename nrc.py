import cv2
import numpy as np
import pandas as pd
import streamlit as st
import fitz # PyMuPDF
from PIL import Image
from io import BytesIO
import re
import pytesseract
import os

# --- Configuration and Initialization ---

# NOTE: This path is often specific to the user's local environment. 
# It must be correct for the code to run locally if pytesseract is not in the system PATH.
# The user provided: r"C:\Users\myatphonesan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
TESSERACT_PATH = os.getenv('TESSERACT_OCR_PATH', r"C:\Users\myatphonesan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")

# TESSERACT_LANGUAGES: Set to 'eng+mya' to enable both English and Myanmar OCR
TESSERACT_LANGUAGES = 'eng+mya'

def configure_tesseract():
    """Sets the Tesseract command path."""
    try:
        # Check if the path is set and the file exists before attempting to set it
        if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        elif TESSERACT_PATH:
             st.warning(f"Tesseract executable not found at specified path: {TESSERACT_PATH}. Tesseract might still work if it's in your system PATH.")
    except Exception as e:
        st.error(f"Error setting Tesseract path. Error details: {e}")

# Set the page configuration early
st.set_page_config(
    page_title="Myanmar NRC OCR Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define Normalized Region Coordinates (0-1000 scale) for NRC/ID Card ---
# [x_min, y_min, x_max, y_max] for targeted extraction. 
# Optimized for the right-hand side, where most form fields are written.

NRC_TARGET_REGIONS = {
    # Key fields as translated from a typical Myanmar ID/Verification Card
    "ID Number (အမှတ်)": [400, 180, 950, 270],
    "Name (အမည်)": [400, 380, 950, 460],
    "Father's Name (ဖခင်အမည်)": [400, 460, 950, 540],
    "Date of Birth (မွေးသက္ကရာဇ်)": [400, 540, 950, 620],
    "Nationality/Religion (လူမျိုး/ဘာသာ)": [400, 620, 950, 700],
}

# Global dictionary to hold the active regions (set in main())
TARGET_FIELD_REGIONS = NRC_TARGET_REGIONS


# --- Core Extraction Logic (Targeted by Region) ---
def extract_fields_by_region(image_array, active_regions):
    """ Extracts text for the specified fields using targeted regions defined by active_regions. """
    kv_data = {key: '-' for key in active_regions.keys()}
    H, W, _ = image_array.shape
    img_boxes = cv2.cvtColor(image_array.copy(), cv2.COLOR_BGR2RGB)

    for key, (x_min_norm, y_min_norm, x_max_norm, y_max_norm) in active_regions.items():
        # Denormalize coordinates
        x_min = int(x_min_norm * W / 1000)
        y_min = int(y_min_norm * H / 1000)
        x_max = int(x_max_norm * W / 1000)
        y_max = int(y_max_norm * H / 1000)

        # Crop and run Tesseract
        cropped_img = image_array[y_min:y_max, x_min:x_max]
        if cropped_img.size == 0: continue
        
        # Use a localized Myanmar/English language pack
        text_raw = pytesseract.image_to_string(
            cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), 
            lang=TESSERACT_LANGUAGES
        ).strip()
        
        extracted_value = '-'
        if text_raw:
            # Simple cleanup: remove key text and line breaks from the raw text
            lines = [line.strip() for line in text_raw.split('\n') if line.strip()]
            
            # Extract the Myanmar key part for better filtering
            myanmar_key_match = re.search(r'\((.*?)\)', key)
            key_text_to_remove = myanmar_key_match.group(1).strip() if myanmar_key_match else key.split(' ')[0]
            
            # Simple heuristic: try to remove the key label from the extracted text
            cleaned_text = " ".join(lines)
            
            # Use regex to find and remove the key label (Myanmar text)
            # This is a rough attempt since Myanmar OCR output can be messy
            key_pattern = re.escape(key_text_to_remove) + r'\s*[:\.\-\—]*'
            value_only = re.sub(key_pattern, '', cleaned_text, flags=re.IGNORECASE).strip()
            
            # Final check to ensure we didn't remove everything
            if value_only and value_only.lower() != key_text_to_remove.lower():
                 extracted_value = value_only
            elif lines:
                 # Fallback: just use all the text found in the box
                 extracted_value = " ".join(lines)
        
        if extracted_value and extracted_value != '-':
            kv_data[key] = extracted_value.replace('\n', ' ').strip()

        # Draw the box for visualization (Red border)
        cv2.rectangle(img_boxes, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    img_with_boxes = Image.fromarray(img_boxes)
    # Reformat the output DataFrame list, showing only the English translation key
    kv_df_list = [{'Key Label (English)': k.split('(')[0].strip(), 'Extracted Value': v} for k, v in kv_data.items()]
    df_kv_pairs = pd.DataFrame(kv_df_list)
    
    return df_kv_pairs, img_with_boxes

# --- Full Text Extraction (Non-Structured) ---
def extract_full_text(image_array):
    """ Runs Tesseract on the entire image to get non-structured text. """
    with st.spinner("Extracting full page text (Non-Structured)..."):
        full_text = pytesseract.image_to_string(
            cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB),
            lang=TESSERACT_LANGUAGES
        ).strip()
    return full_text

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
                # Use PyMuPDF (fitz) for PDF conversion
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
            # Use OpenCV for image decoding
            img_array = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            return img_array, file_type_str
    except Exception as e:
        st.error(f"Error loading file. Check if it's a valid Image or non-encrypted PDF. Error details: {e}")
        return None, None

def get_download_button(data, is_dataframe, file_format, label, file_name_base, help_text=""):
    """Generates a common download button for different formats."""
    df = data
    data_out = None
    mime = ''

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
        # Fallback for unexpected formats
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
    
    st.title("🇲🇲 Myanmar NRC/ID Card OCR Extractor")
    st.markdown("This tool uses **Tesseract OCR (eng+mya)** to extract specific fields and the full page text from a Myanmar Identity Document, using regions optimized for the common ID card layout.")

    # 1. File Upload
    uploaded_file = st.file_uploader(
        "Choose an NRC or ID Card Image/Scan (JPG, PNG, PDF)",
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
        
        st.success(f"**Processing Mode:** Using **NRC Target Regions** for extraction.")
        
        st.subheader("2. OCR Processing and Result Formats")
        
        # Run both extractions with the fixed NRC regions
        df_kv_pairs, img_with_boxes = extract_fields_by_region(image_array, NRC_TARGET_REGIONS)
        full_text = extract_full_text(image_array)

        col_img, col_data_tabs = st.columns([1, 2])
        
        with col_img:
            st.markdown("### 🖼️ OCR Visualization")
            st.image(img_with_boxes, caption=f"Targeted extraction regions (Red Boxes)", use_column_width=True)
            
        with col_data_tabs:
            # Create two tabs: one for Structured, one for Non-Structured
            tab_structured, tab_non_structured = st.tabs(["📄 Structured Table", "📋 All Non-Structured Text"])

            with tab_structured:
                st.markdown("### 🔑 Extracted Key-Value Pairs")
                st.dataframe(
                    df_kv_pairs[['Key Label (English)', 'Extracted Value']], 
                    use_container_width=True, 
                    hide_index=True
                )
                
                st.markdown("#### Download Options (Table)")
                col_csv, col_txt, col_word = st.columns(3)
                with col_csv:
                    get_download_button(df_kv_pairs, True, 'csv', "📥 Download CSV", 'nrc_structured_key_value_pairs')
                with col_txt:
                    get_download_button(df_kv_pairs, True, 'txt', "📥 Download TXT", 'nrc_structured_key_value_pairs')
                with col_word:
                    get_download_button(df_kv_pairs, True, 'doc', "📥 Download DOC", 'nrc_structured_key_value_pairs', help_text="Saves the table data as a text file with a .doc extension.")
            
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
                    get_download_button(full_text, False, 'txt', "📥 Download TXT", 'nrc_full_extracted_text')
                with col_word_full:
                    get_download_button(full_text, False, 'doc', "📥 Download DOC", 'nrc_full_extracted_text', help_text="Saves the full text as a text file with a .doc extension.")
                
        st.markdown("---")

if __name__ == '__main__':
    main()