import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes # Used as per common deployment practice for PDF handling
import re
import pytesseract
import os
import io

# --- Configuration and Initialization ---
# TESSERACT_LANGUAGES: Set to 'eng+mya' to enable both English and Myanmar OCR
# In a cloud environment (like Streamlit Cloud), Tesseract must be installed 
# and in the PATH. The fixed path and configure_tesseract() are removed 
# as they are not valid for cross-system deployment.
TESSERACT_LANGUAGES = 'eng+mya'

# Set the page configuration early
st.set_page_config(
    page_title="Myanmar DL OCR Extractor (Deployment Ready)",
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
        # print("Aspect ratio suggests Portrait (H > W). Rotating image 90 degrees clockwise to landscape.") 
        image_array = cv2.rotate(image_array, cv2.ROTATE_90_CLOCKWISE)
        
    return image_array

# --- Perspective Correction (Deskew/Warp) ---
def warp_document(image_array):
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
        # Return the original image and a warning
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
    # Draw contour with thicker line for visibility
    cv2.drawContours(img_contour, [screenCnt], -1, (0, 255, 0), 20) 

    return warped, img_contour, "Warping successful."


# --- Core Extraction Logic (Targeted by Region) ---
def extract_fields_by_region(image_array, active_regions):
    """ Extracts text for the specified fields using targeted regions defined by active_regions. """
    kv_data = {key: 'N/A' for key in active_regions.keys()}
    H, W, _ = image_array.shape
    img_boxes = image_array.copy() # Use BGR copy for drawing

    for key, (x_min_norm, y_min_norm, x_max_norm, y_max_norm) in active_regions.items():
        x_min = int(x_min_norm * W / 1000)
        y_min = int(y_min_norm * H / 1000)
        x_max = int(x_max_norm * W / 1000)
        y_max = int(y_max_norm * H / 1000)

        cropped_img = image_array[y_min:y_max, x_min:x_max]
        if cropped_img.size == 0: continue
        
        # Use PSM 7 for single line extraction of fields (more focused)
        psm_config = '--psm 7' 

        # Convert to RGB for pytesseract
        text_raw = pytesseract.image_to_string(
            cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), 
            lang=TESSERACT_LANGUAGES, 
            config=psm_config
        ).strip()
        
        extracted_value = 'N/A'
        if text_raw:
            # Normalize whitespace
            extracted_value = re.sub(r'[\n\s]+', ' ', text_raw).strip()

            # Specific cleaning for NRC No.
            if key == "NRC No. (N.R.C No.)":
                # Matches patterns like 12/KaTaKa(N)123456 or 12/KaMaNa(N)123456
                # Handles both Burmese and English characters in the location part
                # The regex is adjusted for common OCR errors (replacing space, backslash, etc. for initial match)
                nrc_pattern = r'(\d+[\/\\-][a-zA-Z\s\u1000-\u109F]+[\(\[][a-zA-Z\s\u1000-\u109F]+[\)\]]\d+)'
                # Clean up extracted_value before searching for the pattern
                nrc_text_cleaned = extracted_value.replace(' ', '').replace('\\', '/').replace('-', '/')

                nrc_match = re.search(nrc_pattern, nrc_text_cleaned)
                if nrc_match:
                    # Re-insert the proper slash after matching
                    extracted_value = nrc_match.group(1).replace('/', '/') 
                else:
                    extracted_value = extracted_value # keep raw if pattern not found
            
            # Simple cleaning for date fields
            elif "Date" in key or "Valid" in key:
                 # Remove non-date/non-separator characters
                 extracted_value = re.sub(r'[^\d\s\-\/]', '', extracted_value).strip()


        if extracted_value and extracted_value != 'N/A':
            kv_data[key] = extracted_value.strip()

        # Draw box on the image (using BGR for cv2)
        cv2.rectangle(img_boxes, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3) # Blue box for emphasis

    # Convert final image with boxes from BGR to RGB for PIL/Streamlit display
    img_with_boxes = Image.fromarray(cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))
    kv_df_list = [{'Key Label (Form Text)': k, 'Extracted Value': v} for k, v in kv_data.items()]
    df_kv_pairs = pd.DataFrame(kv_df_list)
    
    return df_kv_pairs, img_with_boxes

# --- Full Text Extraction (Non-Structured) ---
def extract_full_text(image_array):
    """ Runs Tesseract on the entire image to get non-structured text using all configured languages. """
    with st.spinner("Extracting full page text (Non-Structured)..."):
        # Convert to RGB for pytesseract
        full_text = pytesseract.image_to_string(
            cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB),
            lang=TESSERACT_LANGUAGES 
        ).strip()
    return full_text

# --- Utility Functions (File Handling) ---
def handle_file_upload(uploaded_file):
    """ 
    Handles file uploads, converting them to an OpenCV array.
    Uses pdf2image for PDF conversion, suitable for cloud deployment.
    """
    file_type = uploaded_file.type
    file_type_str = 'image'
    
    try:
        file_bytes = uploaded_file.read()
        if 'pdf' in file_type:
            file_type_str = 'pdf'
            with st.spinner("Converting PDF page 1 to image (200 DPI)..."):
                # Use pdf2image for conversion
                images = convert_from_bytes(
                    file_bytes, 
                    dpi=200, 
                    first_page=0, 
                    last_page=1,
                    fmt='jpeg' # Use a common format
                )
                if not images:
                    st.error("Could not process PDF. The document is empty or unreadable.")
                    return None, None
                
                # Convert the first PIL Image to a numpy array, then to BGR for OpenCV
                img_pil = images[0]
                img_array = np.array(img_pil.convert('RGB'))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                return img_array, file_type_str
        else:
            # Handle standard image files
            img_array = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            return img_array, file_type_str
            
    except Exception as e:
        st.error(f"Error loading file. Check if it's a valid Image or non-encrypted PDF. Error details: {e}")
        return None, None

def get_download_button(data, is_dataframe, file_format, label, file_name_base, help_text=""):
    """Generates a common download button for different formats."""
    data_out = None
    mime = 'text/plain' 
    final_name = f'{file_name_base}.{file_format}'
    df = data

    if file_format == 'csv' and is_dataframe:
        data_out = df.to_csv(index=False).encode('utf-8')
        mime = 'text/csv'
    elif file_format in ['txt', 'doc']:
        if is_dataframe:
            # Format DataFrame nicely for text file
            text_output = ""
            for index, row in df.iterrows():
                text_output += f"{row['Key Label (Form Text)']}: {row['Extracted Value']}\n"
            data_out = text_output.encode('utf-8')
        else: # For non-structured text
            data_out = data.encode('utf-8')
        # mime is kept as 'text/plain' even for .doc as it's a simple text-based download

    if data_out is not None:
        st.download_button(
            label=label,
            data=data_out,
            file_name=final_name,
            mime=mime,
            help=help_text
        )

# --- Streamlit Application Layout ---
def main():
    
    st.title("🇲🇲 Myanmar Driving License OCR Extractor (Deployment Ready)")
    st.info("This application is configured for cloud deployment (e.g., Streamlit Cloud) using system packages for Tesseract and Poppler.")
    
    # 1. File Upload
    uploaded_file = st.file_uploader(
        "**1. Upload Document File (Image or PDF)**",
        type=['jpg', 'jpeg', 'png', 'pdf'],
        help="For multi-page PDFs, only the first page will be processed. Ensure the image is clear and the document is not heavily obscured."
    )
    st.markdown("---")

    image_array = None
    file_type = None
    
    if uploaded_file is not None:
        with st.spinner(f"Loading {uploaded_file.type.split('/')[-1]}..."):
            image_array, file_type = handle_file_upload(uploaded_file)

    # --- OCR Processing and Results Display ---
    if image_array is not None and file_type is not None:
        
        st.subheader("2. Preprocessing & Perspective Correction")
        
        # 2. Correct 90/270 rotation based on Aspect Ratio (Ensures Landscape)
        rotated_image = correct_aspect_rotation(image_array.copy())
        
        # 3. Perspective Warp (Deskew)
        with st.spinner("Applying perspective correction..."):
            warped_image, image_with_contour, warp_status = warp_document(rotated_image)
        
        # Display warp status/warning only if it's not a success message
        if "Warping skipped" in warp_status:
              st.warning(warp_status + " Proceeding with the rotated image.")

        # Run extractions on the fully corrected image
        df_kv_pairs, img_with_boxes = extract_fields_by_region(warped_image, TARGET_FIELD_REGIONS)
        full_text = extract_full_text(warped_image)

        # Display Images
        col_contour, col_warped = st.columns(2)
        
        with col_contour:
            st.markdown("#### 🖼️ Document Outline Detection")
            # Convert BGR to RGB for Streamlit display
            img_rgb_contour = cv2.cvtColor(image_with_contour, cv2.COLOR_BGR2RGB)
            st.image(img_rgb_contour, caption="Detected Outline (Green).", use_container_width=True)
            
        with col_warped:
            st.markdown("#### 📐 Final Warped Image for OCR")
            # img_with_boxes is already RGB (PIL Image)
            st.image(img_with_boxes, caption="Fixed Regions Applied (Red Boxes).", use_container_width=True)

        st.markdown("---")
        
        st.subheader("3. OCR Extraction Results")

        # Results Display
        tab_structured, tab_non_structured = st.tabs(["📄 Structured Key-Value Pairs", "📋 Full Non-Structured Text"])

        with tab_structured:
            st.markdown("### Extracted Fields using Fixed Regions")
            st.dataframe(
                df_kv_pairs[['Key Label (Form Text)', 'Extracted Value']], 
                use_container_width=True, 
                hide_index=True
            )
            
            st.markdown("#### Download Table Data")
            col_csv, col_txt, _ = st.columns(3)
            with col_csv:
                get_download_button(df_kv_pairs, True, 'csv', "📥 Download CSV", 'dl_extracted_key_value_pairs')
            with col_txt:
                get_download_button(df_kv_pairs, True, 'txt', "📥 Download TXT", 'dl_extracted_key_value_pairs')
        
        with tab_non_structured:
            st.markdown("### Raw OCR Output (Eng + Mya)")
            st.text_area(
                label="Non-Structured Text",
                value=full_text,
                height=400,
                help="Raw text output from the entire corrected image, useful for debugging."
            )
            
            st.markdown("#### Download Full Text")
            col_txt_full, col_word_full, _ = st.columns(3)
            with col_txt_full:
                get_download_button(full_text, False, 'txt', "📥 Download TXT", 'dl_full_extracted_text')
            with col_word_full:
                get_download_button(full_text, False, 'doc', "📥 Download DOC", 'dl_full_extracted_text', help_text="Saves the full text as a text file with a .doc extension.")
            
        st.markdown("---")

if __name__ == '__main__':
    main()
