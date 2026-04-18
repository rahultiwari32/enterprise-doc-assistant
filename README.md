# 🤖 Enterprise Document Assistant

A RAG-based Document Intelligence Assistant that lets you chat with any PDF document using AI.

## 🎯 What it does
- Upload any PDF document
- Ask questions in natural language
- Get accurate answers with page-level source citations
- Maintains conversation history during session

## 🛠️ Tech Stack
| Component | Technology |
|---|---|
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| Framework | LangChain |
| UI | Streamlit |

## 🏗️ Architecture
PDF Upload → Text Extraction → Chunking → Embedding → FAISS Index
↓
User Question → Embedding → Similarity Search → Context + LLM → Answer

## 🚀 Run Locally

1. Clone the repo
```bash
   git clone https://github.com/rahultiwari32/enterprise-doc-assistant.git
   cd enterprise-doc-assistant
```

2. Install dependencies
```bash
   pip install -r requirements.txt
```

3. Add your API key in `.env`

4. Run the app
```bash
   streamlit run app.py
```

## 📸 Demo
Upload any PDF and start chatting instantly!

## 👨‍💻 Author
Rahul Tiwari — GenAI Developer
[GitHub](https://github.com/rahultiwari32) | [
    LinkedIn](https://www.linkedin.com/in/rahul-tiwari-4083451a1/)