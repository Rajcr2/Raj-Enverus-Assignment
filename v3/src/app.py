"""Streamlit chatbot UI — improved V1.1 visualization."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag import RAG  # noqa: E402


st.set_page_config(
    page_title="Raj Jangam - Assignment",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.1rem 1.35rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            rgba(120, 100, 220, .10),
            rgba(60, 160, 180, .07)
        );
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: .25rem;
    }

    .hero-subtitle {
        color: #6b7280;
        font-size: .98rem;
    }

    .question-card {
        padding: .9rem 1rem;
        border-left: 4px solid #7c3aed;
        border-radius: 10px;
        background: rgba(124, 58, 237, .06);
        margin: .8rem 0 1.2rem 0;
    }

    .section-label {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: .5rem;
    }

    .evidence-meta {
        color: #6b7280;
        font-size: .86rem;
        margin-bottom: .55rem;
    }

    .evidence-text {
        line-height: 1.65;
        font-size: .96rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22);
        padding: .65rem .8rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">📄 Agent-as-a-Judge RAG Assistant</div>
        <div class="hero-subtitle">
            Ask questions about the research paper. The answer is generated from
            retrieved PDF evidence rather than from unrestricted model knowledge.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🔎 RAG System")

    st.markdown(
        """
        **Pipeline**

        `PDF → chunks → embeddings → ChromaDB → semantic retrieval → LLM`
        """
    )

    st.divider()

    st.subheader("Retrieval")
    st.write("• SentenceTransformer embeddings")
    st.write("• ChromaDB semantic search")
    st.write("• V1.1 chunk retrieval")
    st.write("• Exact identifier support")

    st.divider()

    st.subheader("Evidence shown")
    st.write("• Page number")
    st.write("• Chunk number")
    st.write("• Retrieval method")
    st.write("• Section/content metadata when available")
    st.write("• Full retrieved text")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- RAG ----------
@st.cache_resource
def load_rag():
    return RAG()


try:
    rag = load_rag()
except Exception as exc:
    st.error(
        "RAG system is not ready. Run `python src/ingest.py` first "
        "and configure `GROQ_API_KEY`."
    )
    with st.expander("Show technical error"):
        st.exception(exc)
    st.stop()


# ---------- Conversation state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_evidence(evidence):
    """Render retrieved chunks with readable metadata."""
    if not evidence:
        st.info("No retrieved evidence was returned.")
        return

    pages = []
    methods = []

    for item in evidence:
        if item.get("page") is not None:
            pages.append(str(item.get("page")))
        if item.get("found_by"):
            methods.append(str(item.get("found_by")))

    unique_pages = list(dict.fromkeys(pages))
    unique_methods = list(dict.fromkeys(methods))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Evidence chunks", len(evidence))
    with c2:
        st.metric("Pages represented", len(unique_pages))
    with c3:
        st.metric("Retrieval methods", len(unique_methods))

    if unique_methods:
        st.caption("Retrieval: " + " · ".join(unique_methods))

    for i, item in enumerate(evidence, start=1):
        page = item.get("page", "—")
        chunk = item.get("chunk", "—")
        found_by = item.get("found_by", "semantic")
        section = item.get("section")
        content_type = item.get("content_type")

        title = f"Evidence {i}  ·  Page {page}  ·  Chunk {chunk}"

        with st.expander(title, expanded=(i == 1)):
            meta = [
                f"**Retrieval:** {found_by}",
            ]

            if section:
                meta.append(f"**Section:** {section}")
            if content_type:
                meta.append(f"**Type:** {content_type}")

            st.markdown("  |  ".join(meta))

            st.markdown(
                f'<div class="evidence-text">{item.get("text", "")}</div>',
                unsafe_allow_html=True,
            )


# ---------- Previous messages ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            st.markdown("### Answer")
            st.write(message["answer"])

            st.markdown("### Retrieved Evidence")
            render_evidence(message.get("evidence", []))


# ---------- New question ----------
question = st.chat_input("Ask a question about the paper...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(
            f'<div class="question-card"><strong>Question</strong><br>{question}</div>',
            unsafe_allow_html=True,
        )

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating answer..."):
            try:
                result = rag.answer(question)
            except Exception as exc:
                st.error("The RAG pipeline failed while processing this question.")
                with st.expander("Show technical error"):
                    st.exception(exc)
                st.stop()

        answer = result.get("answer", "")
        evidence = result.get("evidence", [])

        st.markdown("### 💡 Answer")
        st.markdown(answer)

        st.divider()
        st.markdown("### 📚 Retrieved Evidence")
        render_evidence(evidence)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "answer": answer,
                "evidence": evidence,
            }
        )
