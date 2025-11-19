import easyocr
import cv2
import numpy as np
import streamlit as st
import re
from datetime import datetime
import pandas as pd
from io import BytesIO

# --- Configuration ---
st.set_page_config(
    page_title="Myanmar Driving License Extractor",
    layout="wide"
)

# --- 1. Core OCR Engine ---
@st.cache_resource
def load_ocr_reader():
    # We strictly use English model to avoid Burmese noise
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr_reader()

# --- 2. Helper Functions ---

def get_box_props(bbox):
    """Returns x_min, y_min, x_max, y_max, center_x, center_y"""
    x_min = min(p[0] for p in bbox)
    x_max = max(p[0] for p in bbox)
    y_min = min(p[1] for p in bbox)
    y_max = max(p[1] for p in bbox)
    return x_min, y_min, x_max, y_max, (x_min + x_max)/2, (y_min + y_max)/2

def parse_date(text):
    """Extracts date and returns a datetime object for comparison."""
    # Pattern for d-m-y or d.m.y or d/m/y
    match = re.search(r'\b(\d{1,2}[-\./]\d{1,2}[-\./]\d{4})\b', text)
    if match:
        d_str = match.group(1).replace(".", "-").replace("/", "-")
        try:
            return datetime.strptime(d_str, "%d-%m-%Y"), match.group(1)
        except:
            pass
    return None, None

def is_valid_blood_type(text):
    """Strictly checks for A, B, AB, O (and fixes 0 -> O)."""
    clean = text.upper().replace("0", "O").replace(".", "").strip()
    if clean in ["A", "B", "AB", "O", "A+", "B+", "AB+", "O+"]:
        return clean
    return None

def merge_boxes(box_list):
    """Merges a list of bounding boxes into a single bounding box."""
    if not box_list: return None
    all_x = [p[0] for box in box_list for p in box]
    all_y = [p[1] for box in box_list for p in box]
    return [[min(all_x), min(all_y)], [max(all_x), min(all_y)], 
            [max(all_x), max(all_y)], [min(all_x), max(all_y)]]

def handle_file_upload(uploaded_file):
    """Converts uploaded file to a CV2 image object."""
    file_bytes = uploaded_file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def create_downloadable_files(extracted_data):
    """Creates file contents for TXT, CSV, and DOCX/Word."""
    
    # Extract only the final text results
    results_dict = {k: v['text'] for k, v in extracted_data.items()}
    
    # 1. Plain Text (TXT)
    txt_content = "\n".join([f"{key}: {value}" for key, value in results_dict.items()])
    
    # 2. CSV
    df = pd.DataFrame(results_dict.items(), columns=['Field', 'Value'])
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    # 3. Word/DOCX (Simple approach: just a fancier text file, but labeled as DOCX)
    # Note: Generating a true DOCX requires a library like python-docx, which adds complexity.
    # For a simple, dependency-free solution, we'll use the plain text format but suggest a .doc/.docx name.
    # User can copy/paste or use a simple RTF structure if a true word document is required.
    # A simple, clean text file is sufficient for the data transfer purpose.
    doc_content = "\n".join([f"{key}\t{value}" for key, value in results_dict.items()])
    
    return txt_content, csv_content, doc_content

# --- 3. Main Logic (extract_data_robust remains the same) ---

def extract_data_robust(raw_results):
    data = {
        "License No": {"text": "", "box": None, "anchor_box": None},
        "Name": {"text": "", "box": None, "anchor_box": None},
        "NRC No": {"text": "", "box": None, "anchor_box": None},
        "Date of Birth": {"text": "", "box": None, "anchor_box": None}, # Calculated via Date Logic
        "Blood Type": {"text": "", "box": None, "anchor_box": None},
        "Valid Up": {"text": "", "box": None, "anchor_box": None} # Calculated via Date Logic
    }

    # --- A. Pre-processing & Global Date Search ---
    clean_blocks = []
    all_dates = [] # Stores tuples: (datetime_obj, text_string, box)

    for item in raw_results:
        bbox, text, prob = item
        if len(text.strip()) < 1: continue
        
        # 1. Check for Dates immediately
        dt_obj, dt_str = parse_date(text)
        if dt_obj:
            all_dates.append({'obj': dt_obj, 'text': dt_str, 'box': bbox})

        clean_blocks.append({
            "box": bbox,
            "text": text,
            "clean": text.lower().replace(".", "").strip(),
            "props": get_box_props(bbox) # x_min, y_min, x_max, y_max...
        })

    # --- B. Auto-Assign Dates (The Fix for Valid Up) ---
    # Sort dates chronologically. 
    # Logic: Earliest is Birthday. Latest is Expiry.
    if all_dates:
        all_dates.sort(key=lambda x: x['obj'])
        
        # Earliest -> DOB
        data["Date of Birth"]["text"] = all_dates[0]['text']
        data["Date of Birth"]["box"] = all_dates[0]['box']
        
        # Latest -> Valid Up (If different from DOB)
        if len(all_dates) > 1:
            data["Valid Up"]["text"] = all_dates[-1]['text']
            data["Valid Up"]["box"] = all_dates[-1]['box']
        elif len(all_dates) == 1:
            # If only one date, it's ambiguous, but usually DOB
            data["Date of Birth"]["text"] = all_dates[0]['text']

    # --- C. Anchor Logic for Text Fields ---
    anchors = {
        "License No": ["license", "no", "b/"],
        "Name": ["name"],
        "NRC No": ["nrc"],
        "Blood Type": ["blood", "type"]
    }

    for field, keywords in anchors.items():
        anchor_block = None
        
        # Find Anchor
        for block in clean_blocks:
            if any(k in block["clean"] for k in keywords):
                # Avoid false positives
                if field == "License No" and "nrc" in block["clean"]: continue
                anchor_block = block
                break
        
        if anchor_block:
            data[field]["anchor_box"] = anchor_block["box"]
            ax_min, ay_min, ax_max, ay_max, ax_center, ay_center = anchor_block["props"]
            
            candidates = []
            
            for block in clean_blocks:
                if block == anchor_block: continue
                # Skip block if it's already used as a Date (prevents putting DOB in Name)
                if any(d['text'] in block["text"] for d in all_dates): continue
                
                bx_min, by_min, bx_max, by_max, bx_center, by_center = block["props"]

                # SPATIAL FILTER:
                # 1. Must be to the RIGHT of anchor start
                is_right = bx_min > ax_min
                
                # 2. Vertical Alignment
                # For Blood Type: Allow looking slightly BELOW the label (The Fix for "O")
                if field == "Blood Type":
                    # Allow candidate center to be below anchor center, but not too far
                    vert_aligned = (by_center > ay_min) and (by_center < ay_max + 30)
                else:
                    # Strict alignment for Name/NRC
                    vert_aligned = abs(ay_center - by_center) < 25

                if is_right and vert_aligned:
                    candidates.append(block)

            # Sort candidates Left -> Right
            if candidates:
                candidates.sort(key=lambda b: b["props"][0])
                
                # --- Specific Field Processing ---
                if field == "Blood Type":
                    # Find first valid blood type in candidates
                    for c in candidates:
                        bt = is_valid_blood_type(c["text"])
                        if bt:
                            data[field]["text"] = bt
                            data[field]["box"] = c["box"]
                            break
                    
                elif field == "License No":
                    # Regex fallback for license
                    full_str = " ".join([c["text"] for c in candidates])
                    match = re.search(r'([A-Z]{1,2}/[\d]{4,6}/[\d]{2,4})', full_str)
                    if match:
                        data[field]["text"] = match.group(1)
                        data[field]["box"] = merge_boxes([c["box"] for c in candidates])
                    else:
                         # Just take the text
                         data[field]["text"] = full_str
                         data[field]["box"] = merge_boxes([c["box"] for c in candidates])

                else: # Name, NRC
                    # Filter Burmese noise
                    valid_c = [c for c in candidates if not any('\u1000' <= char <= '\u109f' for char in c["text"])]
                    if valid_c:
                        data[field]["text"] = " ".join([c["text"] for c in valid_c])
                        data[field]["box"] = merge_boxes([c["box"] for c in valid_c])

    # Final Fallback for NRC if empty (Regex)
    if not data["NRC No"]["text"]:
        # Look for typical NRC pattern in all blocks
        for block in clean_blocks:
            if re.search(r'\d{1,2}/[A-Z]{6}\(N\)\d{6}', block["text"].replace(" ", "")):
                data["NRC No"]["text"] = block["text"]
                data["NRC No"]["box"] = block["box"]
                break
                
    # Clean NRC
    if data["NRC No"]["text"]:
        data["NRC No"]["text"] = data["NRC No"]["text"].upper().replace(" ", "")

    return data

# --- 4. Visualization ---

def draw_annotated_image(image_cv, extracted_data):
    """Draws bounding boxes for anchors (Blue) and values (Green) on the image."""
    img_out = image_cv.copy()
    for field, info in extracted_data.items():
        # Anchor (Blue)
        if info['anchor_box'] is not None:
            tl, tr, br, bl = info['anchor_box']
            # Draw blue rectangle for the label/anchor
            cv2.rectangle(img_out, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (255, 0, 0), 2)
        
        # Value (Green)
        if info['box'] is not None:
            tl, tr, br, bl = info['box']
            # Draw green rectangle for the extracted value
            cv2.rectangle(img_out, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (0, 255, 0), 2)
            # Label the box with the field name
            cv2.putText(img_out, field, (int(tl[0]), int(tl[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            
    return img_out

# --- 5. UI ---

st.title("🪪 Myanmar License Extractor")


uploaded_file = st.file_uploader("Upload License", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image_cv = handle_file_upload(uploaded_file)
    
    if image_cv is not None:
        with st.spinner("Processing..."):
            # 1. OCR reading
            raw_results = reader.readtext(image_cv)
            
            # 2. Data extraction logic
            extracted_data = extract_data_robust(raw_results)
            
            # 3. Visualization
            annotated_img = draw_annotated_image(image_cv, extracted_data)
            
            # 4. Create downloadable content
            txt_file, csv_file, doc_file = create_downloadable_files(extracted_data)

        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Annotated Image")
            st.image(annotated_img, channels="BGR", use_container_width=True)
            
        with col2:
            st.header("Extraction Results")
            with st.form("res"):
                st.text_input("License No", value=extracted_data["License No"]["text"])
                st.text_input("Name", value=extracted_data["Name"]["text"])
                st.text_input("NRC No", value=extracted_data["NRC No"]["text"])
                st.text_input("Date of Birth", value=extracted_data["Date of Birth"]["text"])
                st.text_input("Blood Type", value=extracted_data["Blood Type"]["text"])
                st.text_input("Valid Up To", value=extracted_data["Valid Up"]["text"])
                st.form_submit_button("Done")
                
            st.subheader("Download Data")
            
            # --- Download Buttons ---
            
            # CSV Button
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_file,
                file_name="license_data.csv",
                mime="text/csv",
                help="Download data as a Comma Separated Values file."
            )
            
            # Plain Text Button
            st.download_button(
                label="⬇️ Download Plain Text",
                data=txt_file,
                file_name="license_data.txt",
                mime="text/plain",
                help="Download data as a simple text file."
            )

            # Word/DOC Button (Using simple text/tab separation)
            st.download_button(
                label="⬇️ Download Word (.doc)",
                data=doc_file,
                file_name="license_data.doc",
                mime="application/msword", # Use appropriate MIME type for Word
                help="Download data as a simple text file, suggesting a Word format."
            )
