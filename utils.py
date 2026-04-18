import os
import pandas as pd
from pypdf import PdfReader
from docx import Document
from PIL import Image

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    chunks, metadatas = [], []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            chunks.append(text)
            metadatas.append({
                "page": i+1,
                "source": os.path.basename(file_path),
                "type": "pdf"
            })
    return chunks, metadatas

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    chunks, metadatas = [], []
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip():
            chunks.append(para.text)
            metadatas.append({
                "page": i+1,
                "source": os.path.basename(file_path),
                "type": "word"
            })
    return chunks, metadatas

def extract_text_from_excel(file_path):
    chunks, metadatas = [], []
    xl = pd.ExcelFile(file_path)
    for sheet in xl.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        text = f"Sheet: {sheet}\n{df.to_string()}"
        chunks.append(text)
        metadatas.append({
            "page": 1,
            "source": os.path.basename(file_path),
            "type": "excel",
            "sheet": sheet
        })
    return chunks, metadatas

def extract_text_from_image(file_path):
    try:
        import pytesseract
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        if text.strip():
            return [text], [{
                "page": 1,
                "source": os.path.basename(file_path),
                "type": "image"
            }]
    except Exception as e:
        return [f"Could not extract text: {str(e)}"], [{
            "page": 1,
            "source": os.path.basename(file_path),
            "type": "image"
        }]
    return [], []

def extract_text(file_path, file_type):
    ext = file_type.lower()
    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext in ["docx", "doc"]:
        return extract_text_from_docx(file_path)
    elif ext in ["xlsx", "xls"]:
        return extract_text_from_excel(file_path)
    elif ext in ["png", "jpg", "jpeg"]:
        return extract_text_from_image(file_path)
    return [], []