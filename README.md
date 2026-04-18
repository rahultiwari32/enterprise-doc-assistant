# 🇮🇳 BharatDocs AI

> Chat with any document in English or Hindi — powered by Groq LLM + RAG

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama3-orange)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 What is BharatDocs AI?

BharatDocs AI is an enterprise-grade document intelligence assistant
that lets anyone — regardless of technical background — upload a document
and have a natural conversation with it in English or Hindi.

Built for Indian businesses, students, and professionals who deal with
complex documents daily — invoices, contracts, reports, and more.

---

## ✨ Features

### 📄 Multi-format Document Support
- PDF files (text-based and scanned)
- Word documents (.docx)
- Excel spreadsheets (.xlsx)
- Images with text (JPG, PNG) via OCR

### 🗣️ Bilingual Voice Interface
- Voice input in English and Hindi
- Voice output — hear answers read aloud
- Language toggle in the UI

### 🧠 AI-Powered RAG Pipeline
- Semantic search over document chunks
- Page-level source citations
- Conversation memory within session
- Powered by Groq (Llama 3.3 70B)

### 🇮🇳 India-Specific Features
- GST Invoice Reader — auto extract invoice details
- Simple language mode for complex documents
- Hindi UI labels and voice responses

### 💼 Enterprise Features
- Auto-suggested questions after upload
- Export full chat as PDF report
- Multi-document support
- Clean professional UI

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (Llama 3.3 70B) — Free tier |
| Embeddings | HuggingFace all-MiniLM-L6-v2 — Local |
| Vector Store | FAISS — Local |
| Framework | LangChain |
| OCR | Tesseract |
| Voice Input | SpeechRecognition + Google STT |
| Voice Output | gTTS (Google Text-to-Speech) |
| UI | Streamlit |
| PDF Export | FPDF |

---

## 🏗️ Architecture

User (Voice/Text)
↓
BharatDocs AI (Streamlit UI)
↓
Document Processor
├── PDF → PyPDF
├── Word → python-docx
├── Excel → openpyxl + pandas
└── Image → Tesseract OCR
↓
Text Chunker (LangChain RecursiveTextSplitter)
↓
Embeddings (HuggingFace all-MiniLM-L6-v2)
↓
FAISS Vector Store
↓
Similarity Search (Top-K chunks)
↓
Groq LLM (Llama 3.3 70B)
↓
Answer + Page Citations + Voice Output

---

## 🚀 Run Locally

### Prerequisites
- Python 3.11+
- Tesseract OCR installed
- Groq API key (free at console.groq.com)

### Installation

```bash
# Clone the repo
git clone https://github.com/rahultiwari32/enterprise-doc-assistant.git
cd enterprise-doc-assistant

# Create conda environment
conda create -n docassist python=3.11 -y
conda activate docassist

# Install dependencies
pip install -r requirements.txt

# Install tesseract
brew install tesseract  # Mac
sudo apt install tesseract-ocr  # Ubuntu

# Add your API key
echo "GROQ_API_KEY=your_key_here" > .env

# Run the app
streamlit run app.py
```

---

## 📸 Demo

> Upload any PDF, Word, Excel or Image and start chatting in English or Hindi!

---

## 🎯 Use Cases

- **Students** — Chat with textbooks, research papers
- **Business** — Analyze reports, contracts, invoices
- **Legal** — Understand agreements in simple language
- **Finance** — Extract data from bank statements, GST invoices
- **HR** — Query policy documents, job descriptions

---

## 📁 Project Structure

bharatdocs-ai/
├── app.py              # Main Streamlit application
├── ingestion.py        # Document processing pipeline
├── requirements.txt    # Python dependencies
├── .env               # API keys (not committed)
├── data/              # Sample documents
├── faiss_index/       # Vector store (auto-generated)
└── README.md          # This file

---

## 🗺️ Roadmap

- [x] PDF, Word, Excel, Image support
- [x] RAG pipeline with citations
- [x] English + Hindi voice interface
- [x] GST Invoice Reader
- [x] Auto-suggested questions
- [x] Export chat as PDF
- [ ] WhatsApp integration
- [ ] Mobile app (React Native)
- [ ] Support for 10+ Indian languages
- [ ] On-premise deployment option

---

## 👨‍💻 Author

**Rahul Tiwari** — GenAI Developer

7 years enterprise software experience + PG in AI & ML

[![GitHub](https://img.shields.io/badge/GitHub-rahultiwari32-black)](https://github.com/rahultiwari32)
[![LinkedIn]](https://www.linkedin.com/in/rahul-tiwari-4083451a1/)

---

## 📄 License

MIT License — feel free to use and modify!

---

> Built with ❤️ for India 🇮🇳