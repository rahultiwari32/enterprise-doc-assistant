import os
import tempfile
import base64
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from gtts import gTTS
import streamlit as st
from utils import extract_text
try:
    import speech_recognition as sr
    VOICE_ENABLED = True
except ImportError:
    VOICE_ENABLED = False
import uuid

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MAX_FILE_SIZE_MB = 10
# --- Language Config ---
LANG = {
    "en": {
        "title": "BharatDocs AI",
        "subtitle": "Chat with any document in English or Hindi",
        "upload": "Upload Document",
        "upload_help": "PDF, Word, Excel, or Image",
        "chat_placeholder": "Ask anything about your document...",
        "thinking": "Thinking...",
        "indexing": "Indexing document...",
        "ready": "Ready",
        "clear": "Clear Chat",
        "sources": "Sources",
        "voice_input": "Voice Input",
        "speak": "Speak Answer",
        "suggest": "Suggested Questions",
        "no_doc": "Please upload a document to get started!",
        "pages": "Pages",
        "chunks": "Chunks",
        "type": "Type",
    },
    "hi": {
        "title": "भारतडॉक्स AI",
        "subtitle": "अपने दस्तावेज़ से हिंदी या अंग्रेजी में बात करें",
        "upload": "दस्तावेज़ अपलोड करें",
        "upload_help": "PDF, Word, Excel, या Image",
        "chat_placeholder": "अपने दस्तावेज़ के बारे में कुछ भी पूछें...",
        "thinking": "सोच रहा हूँ...",
        "indexing": "दस्तावेज़ इंडेक्स हो रहा है...",
        "ready": "तैयार",
        "clear": "चैट साफ करें",
        "sources": "स्रोत",
        "voice_input": "आवाज़ से पूछें",
        "speak": "उत्तर सुनें",
        "suggest": "सुझाए गए प्रश्न",
        "no_doc": "शुरू करने के लिए कोई दस्तावेज़ अपलोड करें!",
        "pages": "पृष्ठ",
        "chunks": "खंड",
        "type": "प्रकार",
    }
}

st.set_page_config(
    page_title="BharatDocs AI",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp { background: #f7f8fa; }
    .main .block-container { padding: 1.5rem 2rem; }
    .chat-user {
        background: #E6F1FB;
        border: 1px solid #B5D4F4;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px;
        font-size: 13px;
        color: #042C53;
        display: inline-block;
        max-width: 80%;
    }
    .chat-ai {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 18px 18px 18px 4px;
        padding: 10px 16px;
        font-size: 13px;
        color: #1a1a1a;
        display: inline-block;
        max-width: 80%;
    }
    .src-pill {
        display: inline-block;
        background: #f0f4ff;
        border: 1px solid #c0cff0;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 11px;
        color: #185FA5;
        margin: 3px 3px 0 0;
    }
    .suggest-btn {
        background: #EAF3DE;
        border: 1px solid #C0DD97;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 12px;
        color: #3B6D11;
        cursor: pointer;
        margin: 4px;
    }
    .stat-card {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
    }
    .stat-num { font-size: 18px; font-weight: 600; color: #185FA5; }
    .stat-lbl { font-size: 10px; color: #888; }
    .ready-pill {
        background: #EAF3DE; color: #3B6D11;
        padding: 3px 12px; border-radius: 20px;
        font-size: 11px; font-weight: 500;
    }
    .brand {
        background: linear-gradient(135deg, #FF9933, #FFFFFF, #138808);
        -webkit-background-clip: text;
        font-size: 22px; font-weight: 700;
    }

    /* ---- MOBILE RESPONSIVE ---- */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.8rem 0.8rem !important;
        }
        .chat-user {
            font-size: 12px !important;
            padding: 8px 12px !important;
            max-width: 95% !important;
        }
        .chat-ai {
            font-size: 12px !important;
            padding: 8px 12px !important;
            max-width: 95% !important;
        }
        .src-pill {
            font-size: 10px !important;
            padding: 2px 8px !important;
        }
        .stat-card {
            padding: 6px !important;
        }
        .stat-num {
            font-size: 14px !important;
        }
        .stat-lbl {
            font-size: 9px !important;
        }
        h2 {
            font-size: 18px !important;
        }
        [data-testid="stSidebar"] {
            min-width: 80vw !important;
            max-width: 80vw !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"] {
            min-width: 0 !important;
            max-width: 0 !important;
        }
        .stChatInput {
            font-size: 13px !important;
        }
        div[data-testid="column"] {
            min-width: 0 !important;
        }
    }

    @media (max-width: 480px) {
        .chat-user, .chat-ai {
            max-width: 100% !important;
            font-size: 11px !important;
        }
        .main .block-container {
            padding: 0.5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

dMAX_FILE_SIZE_MB = 10

def index_document(uploaded_file):
    # File size check
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"❌ File too large! Maximum size is {MAX_FILE_SIZE_MB}MB. Your file is {file_size_mb:.1f}MB.")
        return None, 0, 0

    ext = uploaded_file.name.split(".")[-1].lower()

    # Supported format check
    supported = ["pdf", "docx", "doc", "xlsx", "xls", "png", "jpg", "jpeg"]
    if ext not in supported:
        st.error(f"❌ Unsupported file type: .{ext}. Please upload PDF, Word, Excel or Image.")
        return None, 0, 0

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name

        chunks, metadatas = extract_text(tmp_path, ext)

        if not chunks:
            st.error("❌ Could not extract text from this file. Please try another document.")
            return None, 0, 0

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        final_chunks, final_meta = [], []
        for chunk, meta in zip(chunks, metadatas):
            splits = splitter.split_text(chunk)
            for s in splits:
                final_chunks.append(s)
                final_meta.append(meta)

        embeddings = get_embeddings()

        # Session-based vector store — unique per user session
        session_id = st.session_state.get("session_id", "default")
        index_path = f"faiss_index_{session_id}"
        vs = FAISS.from_texts(final_chunks, embeddings, metadatas=final_meta)
        vs.save_local(index_path)
        st.session_state.index_path = index_path

        return vs, len(final_chunks), len(chunks)

    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        return None, 0, 0

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def ask_groq(question, context, lang="en"):
    lang_instruction = "Respond in Hindi." if lang == "hi" else "Respond in English."
    prompt = f"""You are BharatDocs AI, an intelligent document assistant for Indian users.
{lang_instruction}
Use ONLY the context below to answer. Always mention page numbers and source file.
If you cannot find the answer, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        if "rate_limit" in str(e).lower():
            return "⚠️ Too many requests right now. Please wait a moment and try again."
        elif "invalid_api_key" in str(e).lower():
            return "⚠️ API key issue. Please contact the administrator."
        else:
            return f"⚠️ Something went wrong. Please try again."
def extract_gst_details(context, lang="en"):
    lang_instruction = "Respond in Hindi." if lang == "hi" else "Respond in English."
    prompt = f"""You are a GST invoice analyzer for Indian businesses.
Extract the following details from the invoice content below.
{lang_instruction}
Return ONLY these fields, if not found write "Not found":

- Invoice Number
- Invoice Date
- Seller Name
- Seller GSTIN
- Buyer Name  
- Buyer GSTIN
- Place of Supply
- HSN/SAC Code
- Taxable Amount
- CGST Amount
- SGST Amount
- IGST Amount
- Total Tax
- Total Amount (with tax)
- Payment Status

Invoice Content:
{context}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception:
        return "⚠️ Could not extract GST details. Please try again."
    
def get_suggestions(context, lang="en"):
    lang_instruction = "Generate questions in Hindi." if lang == "hi" else "Generate questions in English."
    prompt = f"""Based on this document content, suggest 3 short, useful questions a user might ask.
{lang_instruction}
Return ONLY the 3 questions, one per line, no numbering.

Content: {context[:1000]}"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    questions = response.choices[0].message.content.strip().split("\n")
    return [q.strip() for q in questions if q.strip()][:3]

def text_to_speech(text, lang="en"):
    tts_lang = "hi" if lang == "hi" else "en"
    tts = gTTS(text=text[:500], lang=tts_lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        return f.name

def voice_to_text(lang="en"):
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... speak now!")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5)
            sr_lang = "hi-IN" if lang == "hi" else "en-IN"
            text = recognizer.recognize_google(audio, language=sr_lang)
            return text
    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        return None
    except Exception:
        return None

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

lang = st.session_state.lang
L = LANG[lang]

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0'>
        <div style='font-size:32px'>🇮🇳</div>
        <div style='font-size:18px;font-weight:700;color:#FF9933'>Bharat</div>
        <div style='font-size:18px;font-weight:700;color:#138808'>Docs AI</div>
        <div style='font-size:11px;color:#888;margin-top:4px'>Powered by Groq + RAG</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Language Toggle
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🇬🇧 English",
                     type="primary" if lang == "en" else "secondary",
                     use_container_width=True):
            st.session_state.lang = "en"
            st.rerun()
    with col2:
        if st.button("🇮🇳 हिंदी",
                     type="primary" if lang == "hi" else "secondary",
                     use_container_width=True):
            st.session_state.lang = "hi"
            st.rerun()

    st.markdown("---")

    uploaded_file = st.file_uploader(
        L["upload"],
        type=["pdf", "docx", "xlsx", "xls", "png", "jpg", "jpeg"],
        help=L["upload_help"]
    )

    if uploaded_file:
        with st.spinner(L["indexing"]):
            vs, num_chunks, num_pages = index_document(uploaded_file)
            if vs:
                st.session_state.vector_store = vs
                st.session_state.messages = []
                st.session_state.suggestions = []
                st.session_state.doc_name = uploaded_file.name
                st.session_state.num_chunks = num_chunks
                st.session_state.num_pages = num_pages
                st.session_state.doc_type = uploaded_file.name.split(".")[-1].upper()

    elif os.path.exists("faiss_index/index.faiss"):
        if "vector_store" not in st.session_state:
            with st.spinner("Loading..."):
                st.session_state.vector_store = FAISS.load_local(
                    "faiss_index", get_embeddings(),
                    allow_dangerous_deserialization=True
                )

    if "vector_store" in st.session_state:
        st.markdown(f"""
        <div style='background:#E6F1FB;border-radius:8px;padding:8px 12px;
             font-size:12px;color:#042C53;font-weight:500;margin-bottom:10px'>
            📄 {st.session_state.get("doc_name", "Document")}
        </div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-num'>{st.session_state.get("num_pages","−")}</div>
                <div class='stat-lbl'>{L["pages"]}</div></div>""",
                unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-num'>{st.session_state.get("num_chunks","−")}</div>
                <div class='stat-lbl'>{L["chunks"]}</div></div>""",
                unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-num'>{st.session_state.get("doc_type","−")}</div>
                <div class='stat-lbl'>{L["type"]}</div></div>""",
                unsafe_allow_html=True)

        st.markdown("---")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button(f"🗑️ {L['clear']}", use_container_width=True):
                st.session_state.messages = []
                st.session_state.suggestions = []
                st.rerun()
        with col_c2:
            if st.button("❌ Remove Doc", use_container_width=True):
                st.session_state.messages = []
                st.session_state.suggestions = []
                if "vector_store" in st.session_state:
                    del st.session_state.vector_store
                if "doc_name" in st.session_state:
                    del st.session_state.doc_name
                if "num_chunks" in st.session_state:
                    del st.session_state.num_chunks
                if "num_pages" in st.session_state:
                    del st.session_state.num_pages
                if "doc_type" in st.session_state:
                    del st.session_state.doc_type
                import shutil
                if os.path.exists("faiss_index"):
                    shutil.rmtree("faiss_index")
                    os.makedirs("faiss_index")
                st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px;color:#aaa;text-align:center;line-height:2'>
        <div style='margin-bottom:4px;font-size:11px;color:#888;font-weight:500'>
            Powered by
        </div>
        <span style='background:#f0f0f0;border-radius:4px;padding:2px 6px;margin:2px;font-size:10px;color:#555'>Groq</span>
        <span style='background:#f0f0f0;border-radius:4px;padding:2px 6px;margin:2px;font-size:10px;color:#555'>HuggingFace</span>
        <span style='background:#f0f0f0;border-radius:4px;padding:2px 6px;margin:2px;font-size:10px;color:#555'>FAISS</span>
        <span style='background:#f0f0f0;border-radius:4px;padding:2px 6px;margin:2px;font-size:10px;color:#555'>LangChain</span>
        <br><br>
        <span style='color:#FF9933'>●</span>
        Built with ❤️ for India 🇮🇳
        <span style='color:#138808'>●</span>
    </div>
    <hr style='border:none;border-top:0.5px solid #eee;margin:10px 0'>
    <div style='text-align:center;padding:8px 0'>
        <div style='font-size:11px;color:#888;margin-bottom:4px'>Built by</div>
        <div style='font-size:13px;font-weight:600;color:#185FA5'>Rahul Tiwari</div>
        <div style='font-size:10px;color:#aaa;margin:2px 0'>GenAI Developer</div>
        <div style='margin-top:8px;display:flex;justify-content:center;gap:8px'>
            <a href='https://github.com/rahultiwari32' target='_blank'
               style='background:#f0f0f0;border-radius:4px;padding:3px 8px;
               font-size:10px;color:#333;text-decoration:none'>
               GitHub
            </a>
            <a href='https://www.linkedin.com/in/rahul-tiwari-4083451a1/' target='_blank'
               style='background:#0077B5;border-radius:4px;padding:3px 8px;
               font-size:10px;color:white;text-decoration:none'>
               LinkedIn
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN AREA ---
col_t, col_b = st.columns([5, 1])
with col_t:
    st.markdown(f"## 🇮🇳 {L['title']}")
    st.caption(L["subtitle"])
with col_b:
    if "vector_store" in st.session_state:
        st.markdown(f"<div class='ready-pill' style='margin-top:20px'>● {L['ready']}</div>",
                    unsafe_allow_html=True)

st.markdown("---")

if "vector_store" not in st.session_state:
    st.info(f"👈 {L['no_doc']}")
    st.stop()

# Auto suggestions
if not st.session_state.suggestions and "vector_store" in st.session_state:
    with st.spinner("Generating suggestions..."):
        docs = st.session_state.vector_store.similarity_search("main topic summary", k=2)
        context = " ".join([d.page_content for d in docs])
        st.session_state.suggestions = get_suggestions(context, lang)

# GST Invoice Reader Button
if st.button("🧾 GST Reader", use_container_width=True):
    with st.spinner("Extracting GST details..."):
        try:
            docs = st.session_state.vector_store.similarity_search(
                "invoice GSTIN amount tax total seller buyer", k=5
            )
            context = "\n\n".join([d.page_content for d in docs])
            gst_details = extract_gst_details(context, lang)
            st.session_state.gst_result = gst_details
        except Exception as e:
            st.session_state.gst_result = f"Error: {str(e)}"

if "gst_result" in st.session_state:
    st.markdown("### 🧾 GST Invoice Details")
    st.markdown(st.session_state.gst_result)
    if st.button("Clear GST Result"):
        del st.session_state.gst_result
        st.rerun()

if st.session_state.suggestions:
    st.markdown(f"**💡 {L['suggest']}:**")
    cols = st.columns(len(st.session_state.suggestions))
    for i, (col, q) in enumerate(zip(cols, st.session_state.suggestions)):
        with col:
            if st.button(q, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_question = q

st.markdown("---")

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div style='display:flex;justify-content:flex-end;margin:8px 0'>
            <div class='chat-user'>{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='display:flex;justify-content:flex-start;margin:8px 0'>
            <div class='chat-ai'>{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)
        if "sources" in msg:
            pills = "".join([f"<span class='src-pill'>📄 {s}</span>"
                            for s in msg["sources"]])
            st.markdown(pills, unsafe_allow_html=True)

# Voice input button
col_voice, col_input = st.columns([1, 5])
with col_voice:
    if VOICE_ENABLED:
        if st.button(f"🎤", help=L["voice_input"], use_container_width=True):
            with st.spinner(f"🎤 {L['voice_input']}..."):
                voice_text = voice_to_text(lang)
                if voice_text:
                    st.session_state.pending_question = voice_text
                else:
                    st.warning("Could not hear anything. Please try again.")
    else:
        st.caption("🎤 Voice N/A")

# Text input
question = st.chat_input(L["chat_placeholder"])

# Handle pending question from voice or suggestion
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question

# Process question
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(f"""
    <div style='display:flex;justify-content:flex-end;margin:8px 0'>
        <div class='chat-user'>{question}</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner(L["thinking"]):
        docs = st.session_state.vector_store.similarity_search(question, k=3)
        context = "\n\n".join([
            f"[Page {d.metadata.get('page','?')} | {d.metadata.get('source','doc')}]: {d.page_content}"
            for d in docs
        ])
        answer = ask_groq(question, context, lang)
        pages = list(set([
            f"Page {d.metadata.get('page','?')} · {d.metadata.get('source','')}"
            for d in docs
        ]))

    st.markdown(f"""
    <div style='display:flex;justify-content:flex-start;margin:8px 0'>
        <div class='chat-ai'>{answer}</div>
    </div>""", unsafe_allow_html=True)
    pills = "".join([f"<span class='src-pill'>📄 {p}</span>" for p in pages])
    st.markdown(pills, unsafe_allow_html=True)

    # Voice output
    if st.button(f"🔊 {L['speak']}", key="speak_btn"):
        audio_file = text_to_speech(answer, lang)
        autoplay_audio(audio_file)
        os.unlink(audio_file)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": pages
    })