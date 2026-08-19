import os
import streamlit as st

from rag.pdf_loader import load_pdf
from rag.chunker import chunk_document
from rag.vector_store import build_index
from rag.pipeline import answer_question
from rag import config




st.set_page_config(
    page_title="Legal Contract RAG",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Legal Contract RAG")

st.caption(
    "Ask questions about your amendment documents"
)



with st.sidebar:

    st.header("Document Settings")

    uploaded_files = st.file_uploader(
        "Upload amendment PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    st.header("Chunking")

    chunk_size = st.slider(
        "Chunk size (tokens)",
        min_value=100,
        max_value=1000,
        value=config.CHUNK_SIZE_TOKENS,
        step=100
    )

    overlap = st.slider(
        "Chunk overlap (tokens)",
        min_value=0,
        max_value=300,
        value=config.CHUNK_OVERLAP_TOKENS,
        step=50
    )

    st.divider()

    st.header("Retrieval & Reranking")

    search_mode = st.radio(
        "Search Mode",
        options=["hybrid", "vector", "bm25"],
        format_func=lambda x: {
            "hybrid": "🔀 Hybrid (RRF Fusion)",
            "vector": "🧠 Vector Only",
            "bm25": "🔤 BM25 Only",
        }[x],
        index=0,
        help="Hybrid combines semantic + keyword search via Reciprocal Rank Fusion"
    )

    retrieval_k = st.slider(
        "Retrieval K (candidates)",
        min_value=5,
        max_value=50,
        value=config.RETRIEVAL_K,
        step=5,
        help="How many candidates to fetch from each search method (Stage 1: fast)"
    )

    final_k = st.slider(
        "Final K (after reranking)",
        min_value=1,
        max_value=10,
        value=config.FINAL_K,
        help="How many chunks to keep after cross-encoder reranking (Stage 2: accurate)"
    )

    build_button = st.button("🔨 Build Index")



if build_button:

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    elif overlap >= chunk_size:
        st.error("Overlap must be smaller than chunk size.")

    else:
        all_chunks = []

        for uploaded_file in uploaded_files:

            os.makedirs("documents", exist_ok=True)

            path = os.path.join("documents", uploaded_file.name)

            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            pages = load_pdf(path)

            # Chunk each page individually, preserving source & page metadata
            for page in pages:
                chunks = chunk_document(page["text"], chunk_size, overlap)

                # Index each page's chunks with its source and page number
                build_index(
                    chunks,
                    config.COLLECTION_NAME,
                    source=page["source"],
                    page=page["page"]
                )

                all_chunks.extend(chunks)

        st.session_state.collection_name = config.COLLECTION_NAME
        st.session_state.indexed = True
        st.session_state.chunk_count = len(all_chunks)

        st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")



st.header("Ask the Contract")

question = st.text_input(
    "Enter your question",
    placeholder="What is the amended termination notice period?"
)

if st.button("🔍 Ask"):

    if not question:
        st.warning("Please enter a question.")

    elif "indexed" not in st.session_state:
        st.warning("Please build the index first.")

    else:
        # Pipeline: guardrails → retrieval (mode-dependent) → cross-encoder (final_k) → Gemini
        response = answer_question(question, search_mode=search_mode)

        # Handle guardrail rejections
        if response["out_of_scope"]:
            st.error("🛡️ " + response["answer"])
            st.caption(f"Reason: {response['reasoning']}")

        else:
            st.subheader("Answer")
            st.write(response["answer"])

            st.divider()
            st.subheader("📚 Retrieved Evidence (reranked)")

            for i, source in enumerate(response["sources"]):
                heading_label = source.get("heading", "")
                rerank_score = source.get("rerank_score")
                rrf_score = source.get("rrf_score")

                with st.expander(
                    f"Chunk {i + 1} — "
                    f"{source['source']} — "
                    f"Page {source['page']}"
                    + (f" — §{heading_label}" if heading_label else "")
                ):
                    score_line = f"Vector distance: {source['distance']}"
                    if rrf_score is not None:
                        score_line += f" | RRF score: {rrf_score}"
                    if rerank_score is not None:
                        score_line += f" | Cross-encoder relevance: {rerank_score}"
                    st.caption(score_line)

            st.divider()
            st.caption(f"Confidence: {response['confidence']} · {response['reasoning']}")
