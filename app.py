"""
app.py
CBSE Science Tutor — Streamlit Application
Powered by Sarvam AI + LangChain + ChromaDB RAG
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="CBSE Science Tutor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()  # Load .env if present

# ── Imports (after page config) ───────────────────────────────────────────────
from ingest import (
    run_ingestion,
    load_vectorstore,
    get_embeddings,
    vectorstore_exists,
    CHROMA_DIR,
)
from rag_chain import build_rag_chain, get_relevant_sources


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme */
    :root {
        --primary: #1a6b3c;
        --accent: #f5a623;
        --bg-card: #f0f8f4;
        --text-muted: #666;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a6b3c 0%, #2e8b57 60%, #f5a623 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(26, 107, 60, 0.3);
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p { margin: 0.3rem 0 0; opacity: 0.9; font-size: 1rem; }

    /* Chat messages */
    .chat-user {
        background: #e8f5e9;
        border-left: 4px solid #1a6b3c;
        padding: 0.8rem 1rem;
        border-radius: 0 10px 10px 0;
        margin: 0.5rem 0;
    }
    .chat-assistant {
        background: #fff8e1;
        border-left: 4px solid #f5a623;
        padding: 0.8rem 1rem;
        border-radius: 0 10px 10px 0;
        margin: 0.5rem 0;
    }

    /* Source cards */
    .source-card {
        background: #f0f8f4;
        border: 1px solid #c8e6c9;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        background: #1a6b3c;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        margin-right: 0.3rem;
    }
    .badge-orange {
        background: #f5a623;
    }

    /* Sidebar */
    .sidebar-section {
        background: #f0f8f4;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 1rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* Info box */
    .info-box {
        background: #e3f2fd;
        border: 1px solid #90caf9;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 0.9rem;
    }

    /* Tip box */
    .tip-box {
        background: #fff3e0;
        border: 1px solid #ffcc02;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [],            # Chat history: [{role, content, sources}]
        "chain": None,             # Loaded RAG chain
        "vectorstore": None,       # Loaded ChromaDB
        "embeddings": None,        # Embedding model
        "api_key": os.getenv("SARVAM_API_KEY", ""),
        "mode": "tutor",           # "tutor" or "quiz"
        "class_level": "Class 9",
        "subject_filter": "All Topics",
        "db_ready": False,
        "chain_loaded": False,
        "loading": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Helper Functions ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings_cached():
    return get_embeddings()


def setup_db():
    """Run ingestion if DB doesn't exist, else load existing."""
    with st.spinner("⚙️ Setting up knowledge base... (first run only, ~30 sec)"):
        if not vectorstore_exists():
            run_ingestion()
        embeddings = load_embeddings_cached()
        vs = load_vectorstore(embeddings)
        st.session_state.vectorstore = vs
        st.session_state.embeddings = embeddings
        st.session_state.db_ready = True
    st.success("✅ Knowledge base ready!")


def load_chain():
    """Build RAG chain with current settings."""
    if not st.session_state.api_key:
        st.error("❌ Please enter your Sarvam API key in the sidebar.")
        return False
    if not st.session_state.db_ready:
        st.error("❌ Knowledge base not ready. Please set up first.")
        return False
    with st.spinner("🔗 Loading AI chain..."):
        try:
            chain = build_rag_chain(
                api_key=st.session_state.api_key,
                mode=st.session_state.mode,
            )
            st.session_state.chain = chain
            st.session_state.chain_loaded = True
            return True
        except Exception as e:
            st.error(f"❌ Failed to load chain: {e}")
            return False


def format_query(user_query: str) -> str:
    """Prepend class level context to query."""
    level = st.session_state.class_level
    topic = st.session_state.subject_filter
    context = f"[{level}"
    if topic != "All Topics":
        context += f" - {topic}"
    context += f"] {user_query}"
    return context


def ask_tutor(user_query: str):
    """Run the RAG chain and return answer + sources."""
    if not st.session_state.chain_loaded:
        if not load_chain():
            return None, []

    formatted_query = format_query(user_query)

    try:
        with st.spinner("🤔 Thinking..."):
            result = st.session_state.chain.invoke({"query": formatted_query})

        answer = result.get("result", "Sorry, I could not generate an answer.")

        # Extract source info
        source_docs = result.get("source_documents", [])
        sources = []
        seen = set()
        for doc in source_docs:
            src = doc.metadata.get("source", "NCERT Notes")
            preview = doc.page_content.strip()[:100]
            key = src + preview[:30]
            if key not in seen:
                seen.add(key)
                sources.append({"source": src, "preview": preview})

        return answer, sources

    except Exception as e:
        return f"⚠️ Error: {str(e)}\n\nPlease check your API key and try again.", []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 CBSE Science Tutor")
    st.markdown("---")

    # API Key
    st.markdown("### 🔑 Sarvam API Key")
    api_key_input = st.text_input(
        "Enter your key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-...",
        help="Get your free key at sarvam.ai",
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        st.session_state.chain_loaded = False  # Reset chain on key change

    if st.session_state.api_key:
        st.success("✅ API key set")
    else:
        st.markdown(
            '<div class="tip-box">💡 Get a free API key at <a href="https://sarvam.ai" target="_blank">sarvam.ai</a></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Settings")

    class_level = st.selectbox(
        "Class Level",
        ["Class 9", "Class 10", "Both (9 & 10)"],
        index=["Class 9", "Class 10", "Both (9 & 10)"].index(
            st.session_state.class_level
        ),
    )
    if class_level != st.session_state.class_level:
        st.session_state.class_level = class_level

    subject_filter = st.selectbox(
        "Topic Filter",
        [
            "All Topics",
            "Matter & States",
            "Atoms & Molecules",
            "Cell Biology",
            "Chemical Reactions",
            "Acids, Bases & Salts",
            "Metals & Non-Metals",
            "Reproduction",
            "Electricity & Magnetism",
        ],
    )
    st.session_state.subject_filter = subject_filter

    mode = st.radio(
        "Mode",
        ["📚 Tutor (Q&A)", "📝 Quiz Generator"],
        index=0,
    )
    new_mode = "tutor" if "Tutor" in mode else "quiz"
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.session_state.chain_loaded = False  # Reset on mode change

    st.markdown("---")

    # Setup & Status
    st.markdown("### 🗄️ Knowledge Base")
    if not st.session_state.db_ready:
        if st.button("⚡ Setup Knowledge Base", use_container_width=True):
            setup_db()
    else:
        st.success("✅ DB Ready")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reload Chain", use_container_width=True):
                st.session_state.chain_loaded = False
                if load_chain():
                    st.success("✅ Reloaded!")
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    st.markdown("---")

    # Quick Topics
    st.markdown("### 💡 Quick Questions")
    quick_questions = {
        "⚗️ Osmosis": "What is osmosis? Give an example.",
        "⚡ Faraday": "Explain Faraday's law of electromagnetic induction.",
        "🧪 pH Scale": "What is the pH scale and how is it used?",
        "🔬 Mitosis": "Explain the stages of mitosis with diagrams.",
        "⚖️ Mole": "What is the mole concept? Give a solved example.",
    }
    for label, question in quick_questions.items():
        if st.button(label, use_container_width=True):
            st.session_state["quick_q"] = question
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.75rem; color:#888; text-align:center;">'
        "Powered by Sarvam AI + LangChain<br>NCERT Class 9 & 10 Science"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main Content ──────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🔬 CBSE Science Tutor</h1>
    <p>Your AI-powered NCERT Science guide for Class 9 & 10 | Powered by Sarvam AI</p>
</div>
""", unsafe_allow_html=True)

# Status bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📚 Class", st.session_state.class_level)
with col2:
    st.metric("🎯 Mode", "Tutor" if st.session_state.mode == "tutor" else "Quiz")
with col3:
    db_status = "✅ Ready" if st.session_state.db_ready else "⚠️ Not Set Up"
    st.metric("🗄️ Knowledge Base", db_status)
with col4:
    ai_status = "✅ Connected" if st.session_state.chain_loaded else "⏳ Not Loaded"
    st.metric("🤖 AI Chain", ai_status)

st.markdown("---")

# Auto setup on first load
if not st.session_state.db_ready:
    st.markdown("""
    <div class="info-box">
        <b>👋 Welcome to CBSE Science Tutor!</b><br>
        This app uses RAG (Retrieval-Augmented Generation) to answer your science questions
        based on NCERT content.<br><br>
        <b>To get started:</b>
        <ol>
            <li>Enter your <b>Sarvam API key</b> in the sidebar</li>
            <li>Click <b>Setup Knowledge Base</b> to index NCERT content</li>
            <li>Start asking questions!</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Auto load chain if API key is set and DB is ready
if st.session_state.db_ready and st.session_state.api_key and not st.session_state.chain_loaded:
    load_chain()

# ── Chat Interface ────────────────────────────────────────────────────────────
st.markdown("### 💬 Chat with Your Science Tutor")

# Display chat history
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div class="tip-box" style="text-align:center; padding:1.5rem;">
            <h4>🎓 Ready to learn!</h4>
            <p>Ask me anything from CBSE Class 9 & 10 Science — definitions, explanations,
            numericals, or quiz questions!</p>
            <p><b>Try:</b> "Explain the structure of an atom" or "What is the difference between
            acids and bases?"</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user"><b>👤 You:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-assistant"><b>🤖 Tutor:</b><br>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
                # Show sources if available
                if msg.get("sources"):
                    with st.expander(f"📖 Sources used ({len(msg['sources'])} chunks)", expanded=False):
                        for src in msg["sources"]:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<span class="badge">📄 {src["source"]}</span> '
                                f'{src["preview"]}...'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

# ── Input Area ────────────────────────────────────────────────────────────────
st.markdown("---")

# Handle quick questions from sidebar
if "quick_q" in st.session_state:
    pending_q = st.session_state.pop("quick_q")
else:
    pending_q = None

# Mode-specific placeholder
if st.session_state.mode == "tutor":
    placeholder = "Ask a science question... e.g., 'What is Bohr's atomic model?' or 'Explain osmosis with example'"
else:
    placeholder = "Request a quiz... e.g., 'Give me 5 MCQs on Chemical Reactions for Class 10'"

user_input = st.chat_input(placeholder)

# Handle input (from chat input or quick buttons)
query = user_input or pending_q

if query:
    if not st.session_state.chain_loaded:
        st.error("❌ AI Chain not loaded. Please check your API key and click 'Reload Chain'.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})

        # Get answer
        answer, sources = ask_tutor(query)

        if answer:
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        st.rerun()

# ── Exam Tips Section ─────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("---")
    st.markdown("### 📌 CBSE Exam Tips")

    col1, col2, col3 = st.columns(3)
    tips = [
        ("🧪 Chemistry", [
            "Learn all chemical equations with balancing",
            "Remember reactivity series order",
            "pH scale values: 0-6 acid, 7 neutral, 8-14 base",
            "Properties of acids vs bases table",
        ]),
        ("🔬 Biology", [
            "Draw and label cell diagrams",
            "Remember: Mitochondria = powerhouse",
            "Osmosis vs Diffusion differences",
            "Male & female reproductive system diagrams",
        ]),
        ("⚡ Physics", [
            "Fleming's left/right hand rules",
            "Ohm's law: V = IR",
            "Differences: AC vs DC",
            "Right hand thumb rule for magnetic field",
        ]),
    ]

    for col, (subject, tip_list) in zip([col1, col2, col3], tips):
        with col:
            st.markdown(
                f"<div class='sidebar-section'><b>{subject}</b><ul>"
                + "".join(f"<li>{t}</li>" for t in tip_list)
                + "</ul></div>",
                unsafe_allow_html=True,
            )
