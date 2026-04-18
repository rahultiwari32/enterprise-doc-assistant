import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import streamlit as st

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load vector store
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

def get_relevant_chunks(query: str, vector_store, k=3):
    docs = vector_store.similarity_search(query, k=k)
    return docs

def ask_groq(question: str, context: str) -> str:
    prompt = f"""You are an intelligent document assistant. 
Use ONLY the context below to answer the question.
Always mention which part of the document your answer comes from.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# --- Streamlit UI ---
st.set_page_config(page_title="Doc Assistant", page_icon="🤖")
st.title("🤖 Enterprise Document Assistant")
st.caption("Upload a document and chat with it using AI")

# Load vector store
with st.spinner("Loading knowledge base..."):
    vector_store = load_vector_store()
st.success("✅ Document loaded! Ask me anything.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask a question about your document..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = get_relevant_chunks(question, vector_store)
            context = "\n\n".join([doc.page_content for doc in docs])
            answer = ask_groq(question, context)
            st.markdown(answer)

            # Show sources
            with st.expander("📄 Source chunks used"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}")

    st.session_state.messages.append({"role": "assistant", "content": answer})