"""
app.py

Alz-SumAgent — Alzheimer's research RAG chatbot.
This is the Hugging Face Space entry point (see Dockerfile: COPY app.py . / CMD streamlit run app.py).
app/streamlit_app.py is NOT deployed — do not edit that file expecting it to affect the live Space.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from loguru import logger

from configs.settings import get_settings
from generation.rag_pipeline import RAGPipeline, ConversationMemory
from vector_store.pinecone_store import QueryInput


@st.cache_data(ttl=3600)
def get_pinecone_stats() -> dict:
    """Pull live stats from Pinecone — cached 1 hour. Auto-updates when papers added."""
    try:
        from pinecone import Pinecone
        s     = get_settings()
        pc    = Pinecone(api_key=s.pinecone_api_key)
        index = pc.Index(s.pinecone_index_name)
        stats = index.describe_index_stats()
        total = stats.total_vector_count
        return {
            "chunks": f"{total:,}",
            "papers": f"{total:,}",
        }
    except Exception as e:
        logger.warning(f"[STATS] Pinecone stats unavailable: {e}")
        return {"chunks": "N/A", "papers": "N/A"}


@st.cache_data(ttl=3600)
def get_eval_summary() -> dict:
    """
    Read precomputed eval summary written by evaluation/ragas_eval.py.
    Degrades gracefully if missing — requires evaluation/ to be COPY'd
    in the Dockerfile and evaluation/ui_summary.json to have been
    generated and committed.
    """
    import json
    try:
        path = Path(__file__).resolve().parent / "evaluation" / "ui_summary.json"
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Alz SumAgent",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Load pipeline once — cached ───────────────────────────────────
@st.cache_resource
def load_pipeline() -> RAGPipeline:
    """
    Load RAG pipeline once and cache it.
    Streamlit reruns on every interaction —
    caching prevents reloading the model every time.

    Analogy: The doctor arrives at the hospital once
    in the morning — not for every patient visit.
    """
    logger.info("[APP] Loading RAG pipeline...")
    pipeline = RAGPipeline()
    logger.info("[APP] Pipeline ready")
    return pipeline


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/brain.png", width=80)
    st.title("Alz SumAgent")
    st.markdown("*Alzheimer's Research Assistant*")
    st.divider()

    st.markdown("### Analysis mode")
    analysis_mode = st.radio(
        "Choose analysis depth:",
        options=["Quick", "Deep"],
        captions=[
            "Single Claude call — ~5 seconds",
            "Two-call agent — ~20 seconds, richer"
        ],
        index=0,
    )

    st.divider()
    st.markdown("### Search settings")
    top_k = st.slider(
        "Papers to retrieve",
        min_value=3,
        max_value=20,
        value=10,
        help="More papers = richer context but slower response"
    )

    year_from = st.selectbox(
        "Filter from year",
        options=[None, 2020, 2021, 2022, 2023, 2024, 2025],
        format_func=lambda x: "All years" if x is None else str(x),
    )

    source_filter = st.selectbox(
        "Data source",
        options=[None, "pubmed"],
        format_func=lambda x: "All sources" if x is None else x.upper(),
    )

    st.divider()
    st.markdown("### Pipeline stats")
    stats = get_pinecone_stats()
    st.metric("Papers indexed",     stats["papers"])
    st.metric("Chunks in Pinecone", stats["chunks"])
    st.metric("Embedding model",    "PubMedBERT")

    st.divider()
    st.markdown("### Evaluation")
    eval_summary = get_eval_summary()
    if eval_summary.get("faithfulness_proxy") is not None:
        col_a, col_b = st.columns(2)
        col_a.metric("Citation faithfulness", f"{eval_summary['faithfulness_proxy']:.2f}")
        col_b.metric("Retrieval hit rate", f"{eval_summary['hit_rate']:.2f}")
        st.caption(f"{eval_summary['questions_evaluated']} eval questions · updated {eval_summary['last_run']}")
    else:
        st.caption("Evaluation summary not available")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        if "conversation_memory" in st.session_state:
            st.session_state.conversation_memory.clear()
        st.rerun()

    st.divider()
    st.markdown(
        "⚠️ *For research purposes only. "
        "Not for clinical decisions.*"
    )


# ── Main UI ───────────────────────────────────────────────────────
st.title("🧠 Alz SumAgent")
st.markdown(
    "Ask any question about Alzheimer's disease research. "
    "Answers are grounded in peer-reviewed PubMed papers with citations."
)

# ── Chat history ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationMemory()

# Display chat history (previously completed turns only)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Example questions ─────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("### Example questions")
    col1, col2 = st.columns(2)

    examples = [
        "What blood biomarkers detect Alzheimer's early?",
        "How does pTau217 predict Alzheimer's progression?",
        "What is the role of APOE4 in Alzheimer's risk?",
        "What treatments are in clinical trials for Alzheimer's?",
    ]

    for i, example in enumerate(examples):
        col = col1 if i % 2 == 0 else col2
        if col.button(example, key=f"example_{i}"):
            st.session_state.example_question = example
            st.rerun()

# ── Chat input ────────────────────────────────────────────────────
chat_input_question = st.chat_input(
    "Ask about Alzheimer's research...",
)

# ── Determine the question to process this run ───────────────────
# Either the user just typed+submitted something, OR they clicked an
# example question on the previous rerun (queued in session_state).
# Both paths converge here — previously only chat_input reached
# generation, so example clicks silently did nothing.
question = None
if chat_input_question:
    question = chat_input_question.strip()
elif "example_question" in st.session_state:
    question = st.session_state.pop("example_question")

if question:
    # Validate — worst-case: empty question
    if len(question) < 3:
        st.warning("Please ask a more specific question.")
        st.stop()

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Render the user's message immediately (chat history loop above
    # already ran this rerun and won't show it otherwise)
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        _stats     = get_pinecone_stats()
        mode_label = " Deep analysis..." if analysis_mode == "Deep" else f"Searching {_stats['chunks']} research chunks..."
        with st.spinner(mode_label):
            try:
                # Call pipeline or agent based on mode
                if analysis_mode == "Deep":
                    from generation.summarizer_agent import SummarizerAgent

                    @st.cache_resource
                    def load_agent() -> SummarizerAgent:
                        return SummarizerAgent()

                    agent = load_agent()
                    answer, diagnostic = agent.ask(
                        question  = question,
                        top_k     = top_k,
                        source    = source_filter,
                        year_from = year_from,
                        memory    = st.session_state.conversation_memory,
                    )
                else:
                    pipeline = load_pipeline()
                    answer, diagnostic = pipeline.ask(
                        question  = question,
                        top_k     = top_k,
                        source    = source_filter,
                        year_from = year_from,
                        memory    = st.session_state.conversation_memory,
                    )

                # Display answer
                response_text = answer.to_display()
                st.markdown(response_text)

                # Show diagnostic in expander
                with st.expander("🔍 Search details"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Chunks retrieved", diagnostic.chunks_retrieved)
                    col2.metric("Top similarity", f"{diagnostic.top_score:.3f}")
                    col3.metric("Response time", f"{diagnostic.total_time_ms:.0f}ms")
                    col4.metric("Mode", "🔬 Deep" if analysis_mode == "Deep" else "⚡ Quick")

                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text
                })

            except Exception as e:
                # Fail loud, not silent: log full traceback + surface to user
                logger.exception("[APP] Pipeline call failed")
                error_msg = f"Pipeline error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })