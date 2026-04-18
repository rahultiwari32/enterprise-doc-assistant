import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print(f"✅ Loaded PDF: {len(reader.pages)} pages")
    return text

def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    print("⏳ Loading embedding model (first time ~2 min)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")
    print("✅ Vector store saved!")
    return vector_store

if __name__ == "__main__":
    pdf_path = "data/sample.pdf"
    if not os.path.exists(pdf_path):
        print("⚠️  No PDF found in data/ folder")
    else:
        text = load_pdf(pdf_path)
        chunks = split_text(text)
        create_vector_store(chunks)
        print("\n🎉 Ingestion complete! Ready to chat.")