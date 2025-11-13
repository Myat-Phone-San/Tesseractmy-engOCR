
# 🧾 Document OCR Extractor

A **Streamlit web app** for extracting key-value text pairs from structured trade or form documents (images or PDFs) using **Tesseract OCR**, **OpenCV**, and **PyMuPDF**.  
This app isolates predefined regions of interest (ROIs) on the page to extract targeted fields like “Applicant Name”, “Documentary Credit No.”, “Original Credit Amount”, etc.

---

## 🚀 Features

- 📤 Upload image or PDF documents (first page processed for PDFs)
- 🧠 OCR extraction using **Pytesseract**
- 🔍 Predefined region-based field detection
- 📊 Output results as key-value pairs
- 📦 Download results as **CSV**, **TXT**, or **Word (.doc)**
- 🖼️ Visualization of extraction regions (red bounding boxes)

---

## 🧩 Technologies Used

| Library | Purpose |
|----------|----------|
| **Streamlit** | Web UI and deployment |
| **OpenCV (cv2)** | Image processing |
| **PyMuPDF (fitz)** | PDF to image conversion |
| **Pytesseract** | OCR text extraction |
| **Pandas** | Data handling |
| **Pillow (PIL)** | Image handling |

---

## 🛠️ Installation

### 1. Clone this repository
```bash
git clone https://github.com/Myat-Phone-San/TesseractOCRtesting.git
cd TesseractOCRtesting
```

### 2. Create a virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate    # For macOS/Linux
venv\Scripts\activate       # For Windows
```

### 3. Install dependencies
Create a file named **requirements.txt** with the following contents:
```txt
streamlit
opencv-python-headless
numpy
pandas
pymupdf
pillow
pytesseract
```

Then install them:
```bash
pip install -r requirements.txt
```

### 4. (Optional) For Streamlit Cloud deployment
Add a file named **packages.txt** containing:
```txt
tesseract-ocr
poppler-utils
```

---

## ▶️ How to Run Locally

```bash
streamlit run app.py
```

Then open your browser at:
```
http://localhost:8501
```

---

## 📂 Project Structure

```
📁 TesseractOCRtesting/
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── packages.txt            # For Streamlit Cloud (system packages)
└── README.md               # Project documentation
```

---

## 🧠 How It Works

1. **Upload** a document (image or PDF).  
2. The app converts PDF pages to images using **PyMuPDF**.  
3. The system extracts text from specific pre-defined regions using **Pytesseract**.  
4. Extracted fields are displayed as a key-value table and visualized with bounding boxes.  
5. Results can be **downloaded** in multiple formats (CSV, TXT, DOC).

---

## 🧾 Example Output

| Key Label (Form Text)        | Extracted Value        |
|------------------------------|------------------------|
| Applicant Name               | John Trading Co., Ltd. |
| Documentary Credit No.       | DC-2025-8743           |
| Original Credit Amount       | EUR 15,000.00          |
| Contact Person / Tel         | Mr. Aung - 0945001001  |
| Beneficiary Name             | Shwe Import Co. Ltd.   |

---

## 🧑‍💻 Author

**Myat Phone San**  
📧 myatphonesan131619.email@example.com  
🔗 [LinkedIn](http://linkedin.com/in/myat-phone-san-3759842a8/) • [GitHub](https://github.com/Myat-Phone-San)

---


