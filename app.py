import cv2
import numpy as np
import pandas as pd
import streamlit as st
import fitz # PyMuPDF
from PIL import Image
from io import BytesIO
import re 
import pytesseract
import os # Necessary for path checking

# --- Configuration and Initialization (Tesseract Path) ---

# Set the Tesseract executable path as requested by the user.
# NOTE: This path is specific to the user's local machine and might need adjustment.
TESSERACT_PATH = r"C:\Users\myatphonesan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def configure_tesseract():
    """Sets the Tesseract command path and checks if the custom path exists."""
    try:
        # Check if running in an environment where Tesseract is expected to be installed
        if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
            return True
        elif os.name == 'posix' or TESSERACT_PATH.lower() == 'tesseract':
            # Assume tesseract is in PATH on Linux/Mac, or if the default name is used
            pytesseract.pytesseract.tesseract_cmd = 'tesseract'
            return True
        else:
            st.warning(f"Tesseract executable not found at **{TESSERACT_PATH}**. Falling back to system PATH...")
            # Allow Tesseract to try running from PATH
            pytesseract.pytesseract.tesseract_cmd = 'tesseract'
            return False 
    except Exception as e:
        st.error(f"Error configuring Tesseract. Please check installation. Error details: {e}")
        return False

# Set the page configuration early
st.set_page_config(
    page_title="Document OCR Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define Normalized Region Coordinates (0-1000 scale) ---
# [x_min, y_min, x_max, y_max]

# 1. Tightly defined regions, optimized for clean PDF-converted images
TARGET_REGIONS_PDF = {
    "Applicant Name": [50, 120, 480, 200],
    "Documentary Credit No.": [480, 120, 680, 200],
    "Original Credit Amount": [680, 120, 930, 200],
    "Contact Person / Tel": [50, 200, 480, 300],
    "Beneficiary Name": [50, 300, 480, 400], 
}

# 2. Expanded regions, with a buffer for minor misalignment in image scans/photos
TARGET_REGIONS_IMAGE = {
    # Expanded horizontally (x_min - 20, x_max + 20) and vertically (y_min - 10, y_max + 10)
    "Applicant Name": [50, 150, 470, 240],
    "Documentary Credit No.": [460, 150, 650, 240],
    "Original Credit Amount": [640, 150, 950, 240],
    "Contact Person / Tel": [50, 230, 470, 310],
    "Beneficiary Name": [50, 330, 470, 410],
}


# --- Core Extraction Logic (Targeted by Region) ---

def extract_fields_by_region(image_array, regions_dict):
    """
    Extracts text for the specified fields by cropping the image 
    to predefined normalized regions (from the provided regions_dict) 
    and running Tesseract on each.
    """
    kv_data = {key: '-' for key in regions_dict.keys()}
    
    if image_array is None or image_array.size == 0 or image_array.ndim != 3:
        return pd.DataFrame(), None
        
    H, W, _ = image_array.shape
    
    # Placeholder for drawing boxes
    img_boxes = cv2.cvtColor(image_array.copy(), cv2.COLOR_BGR2RGB)
    
    # Use the appropriate dictionary for region extraction
    for key, (x_min_norm, y_min_norm, x_max_norm, y_max_norm) in regions_dict.items():
        
        # 1. Denormalize coordinates to actual pixel values
        x_min = int(x_min_norm * W / 1000)
        y_min = int(y_min_norm * H / 1000)
        x_max = int(x_max_norm * W / 1000)
        y_max = int(y_max_norm * H / 1000)
        
        # Ensure coordinates are within bounds
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(W, x_max), min(H, y_max)

        # 2. Crop the image to the target region
        cropped_img = image_array[y_min:y_max, x_min:x_max]
        
        if cropped_img.size == 0:
            continue

        # 3. Run Tesseract on the cropped image
        # Using configuration for more robust text recognition on structured documents
        config_tesseract = '--psm 6' # Assume a single uniform block of text
        # Only using 'eng' language for better accuracy on these specific fields
        text_raw = pytesseract.image_to_string(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), lang='eng', config=config_tesseract).strip()
        
        # 4. Process the extracted text
        extracted_value = '-'
        
        if text_raw:
            lines = [line.strip() for line in text_raw.split('\n') if line.strip()]
            key_lower = key.lower().strip()
            key_index = -1
            
            # Find the line/part that contains the key label 
            for i, line in enumerate(lines):
                if key_lower in line.lower(): 
                    key_index = i
                    break
            
            if key_index != -1:
                # The value is all subsequent text joined together
                value_lines = lines[key_index + 1:]
                
                # Special handling for single-line fields
                if not value_lines and key_lower in lines[key_index].lower():
                    line_text = lines[key_index]
                    key_start = line_text.lower().find(key_lower)
                    key_end_index = key_start + len(key)
                    
                    value_on_same_line = line_text[key_end_index:].strip()
                    value_on_same_line = re.sub(r'[\:\-\=\—\.]', '', value_on_same_line).strip()
                    
                    if value_on_same_line:
                        extracted_value = value_on_same_line
                
                elif value_lines:
                    extracted_value = " ".join(value_lines)
            
            # Fallback: if no key was found, the whole region might just be the value
            if extracted_value == '-' and lines:
                all_text = " ".join(lines)
                if key_lower not in all_text.lower():
                    extracted_value = all_text


        # 5. Post-process specific fields for better presentation/accuracy
        if key == "Original Credit Amount" and extracted_value != '-':
            # Robust cleaning for currency/amount
            extracted_value = extracted_value.upper().replace('O', '0').replace('S', '5').replace('L', '1')
            amount_match = re.search(r'(\d[\d\.\,]*)', extracted_value.replace(' ', ''))
            currency = 'EUR' if 'EUR' in extracted_value else ''
            
            if amount_match:
                amount_str = amount_match.group(0)
                try:
                    cleaned_amount = re.sub(r'[^\d\.\,]', '', amount_str)
                    
                    # Heuristic to handle European/American formatting
                    if ',' in cleaned_amount and '.' in cleaned_amount:
                        if cleaned_amount.rfind(',') > cleaned_amount.rfind('.'):
                            amount_float = float(cleaned_amount.replace('.', '').replace(',', '.'))
                        else:
                            amount_float = float(cleaned_amount.replace(',', ''))
                    elif ',' in cleaned_amount:
                        amount_float = float(cleaned_amount.replace(',', '.'))
                    else:
                        amount_float = float(cleaned_amount)
                    
                    extracted_value = f"{currency} {amount_float:,.2f}".strip()
                except ValueError:
                    extracted_value = amount_str.strip()

        elif key in ["Applicant Name", "Contact Person / Tel", "Beneficiary Name"] and extracted_value != '-':
            extracted_value = re.sub(r'\s+', ' ', extracted_value).strip()
        
        # Final cleanup for values
        if extracted_value and extracted_value != '-':
            kv_data[key] = extracted_value.strip()

        # Draw the box for visualization (Red border)
        cv2.rectangle(img_boxes, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    # Convert the box-drawn image to PIL Image for Streamlit
    img_with_boxes = Image.fromarray(img_boxes)

    # Convert KV data to DataFrame
    kv_df_list = [{'Key Label (Form Text)': k, 'Extracted Value': v} for k, v in kv_data.items()]
    df_kv_pairs = pd.DataFrame(kv_df_list)

    return df_kv_pairs, img_with_boxes

# --- Utility Functions ---

def handle_file_upload(uploaded_file):
    """Handles file uploads, converting them to an OpenCV array and returning the file type."""
    file_type = uploaded_file.type
    
    try:
        file_bytes = uploaded_file.read()

        if 'pdf' in file_type:
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
                return img_array, 'PDF'
            
        else: # Handle image files (jpg, png, etc.)
            img_array = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            return img_array, 'Image'

    except Exception as e:
        st.error(f"Error loading file. Check if it's a valid Image or non-encrypted PDF. Error details: {e}")
        return None, None

def get_download_button(data, file_format, label, file_name_base, help_text=""):
    """Generates a common download button for different formats."""
    
    df = data
    if file_format == 'csv':
        data_out = df.to_csv(index=False).encode('utf-8')
        mime = 'text/csv'
        final_name = f'{file_name_base}.csv'
    elif file_format == 'txt':
        data_out = df.to_string(index=False).encode('utf-8')
        mime = 'text/plain'
        final_name = f'{file_name_base}.txt'
    else: # doc
        data_out = df.to_string(index=False).encode('utf-8')
        mime = 'application/msword'
        final_name = f'{file_name_base}.doc'
        
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
    
    st.title("🎯Document OCR Extractor (Tesseract)")
    st.markdown("""
        This tool uses **two distinct sets of normalized region coordinates** based on the file type for optimal extraction:
        - **PDF Mode:** Uses tighter regions (optimized for clean digital form layout).
        - **Image Mode (JPG/PNG):** Uses slightly buffered regions (expanded boxes) to account for potential misalignment or margins in scans/photos.
    """)
    
    # 1. File Upload
    uploaded_file = st.file_uploader(
        "Choose a Document File",
        type=['jpg', 'jpeg', 'png', 'pdf'],
        help="For multi-page PDFs, only the first page will be processed."
    )

    st.markdown("---")

    image_array = None
    file_mode = None
    
    if uploaded_file is not None:
        # Determine file type and get image array
        image_array, file_mode = handle_file_upload(uploaded_file)
        
        if image_array is not None:
            # Select the appropriate region dictionary
            regions_to_use = TARGET_REGIONS_PDF if file_mode == 'PDF' else TARGET_REGIONS_IMAGE
            regions_description = "Tightly defined (PDF Mode)" if file_mode == 'PDF' else "Expanded/Buffered (Image Mode)"
            
            st.success(f"File **'{uploaded_file.name}'** uploaded. **Mode: {file_mode}**. Using **{regions_description}** regions.")

            st.subheader("2. Extracted Results")
            
            # Run targeted extraction
            with st.spinner(f"Running targeted OCR using {regions_description}..."):
                df_kv_pairs, img_with_boxes = extract_fields_by_region(image_array, regions_to_use)

            # Display results in a two-column layout
            col_img, col_data = st.columns([1, 2])
            
            with col_img:
                st.markdown("### 🖼️ OCR Visualization (Targeted Regions)")
                st.image(img_with_boxes, caption="Targeted extraction regions (Red Boxes) scaled to image size.", use_column_width=True)

            with col_data:
                st.markdown("### 🔑 Extracted Key-Value Pairs")
                
                # Displaying the custom, refined output
                st.dataframe(df_kv_pairs[['Key Label (Form Text)', 'Extracted Value']], use_container_width=True, hide_index=True)
                
                st.markdown("#### Download Options")
                col_csv, col_txt, col_word = st.columns(3)
                
                file_name_base = uploaded_file.name.split('.')[0] + '_extracted'

                with col_csv:
                    get_download_button(df_kv_pairs, 'csv', "📥 Download CSV", file_name_base)

                with col_txt:
                    get_download_button(df_kv_pairs, 'txt', "📥 Download TXT", file_name_base)
                    
                with col_word:
                    get_download_button(df_kv_pairs, 'doc', "📥 Download DOC (Word)", file_name_base, help_text="Saves the table data as a text file with a .doc extension.")
                
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Built with Tesseract, OpenCV, Pandas, Streamlit, and PyMuPDF.
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()