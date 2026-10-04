import os

import streamlit as st

from core import config
from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_document
from retrieval.vector_store import build_index, clear_index
from pipeline.rag import answer_question
from router.agent import route as router_route
from ui.history import load_history, save_history
from ui.renderers import render_route_badge, render_mcp_tools, render_full_doc_info, render_sources

st.set_page_config(
    page_title="Legal Contract Intelligence",
    page_icon="⚖️",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_data" not in st.session_state:
    st.session_state.debug_data = {}
if "full_doc_texts" not in st.session_state:
    st.session_state.full_doc_texts = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚖️ Legal Contract Intelligence")

    st.subheader("📄 Documents")
    uploaded_files = st.file_uploader(
        "Upload contract PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    st.divider()
    st.subheader("⚙️ Chunking")
    chunk_size = st.slider("Chunk size (tokens)", 100, 1000, config.CHUNK_SIZE_TOKENS, 100)
    overlap    = st.slider("Chunk overlap (tokens)", 0, 300, config.CHUNK_OVERLAP_TOKENS, 50)

    st.divider()
    st.subheader("🔍 Retrieval")
    search_mode = st.radio(
        "Search Mode",
        options=["hybrid", "vector", "bm25"],
        format_func=lambda x: {
            "hybrid": "🔀 Hybrid (RRF Fusion)",
            "vector": "🧠 Vector Only",
            "bm25":   "🔤 BM25 Only",
        }[x],
        index=0,
    )
    retrieval_k = st.slider("Retrieval K (candidates)", 10, 100, config.RETRIEVAL_K, 10)
    mmr_k       = st.slider("MMR K (diverse chunks)",   5,  50,  config.MMR_K,       5)
    mmr_lambda  = st.slider("MMR Lambda (λ)",           0.0, 1.0, config.MMR_LAMBDA, 0.1)
    final_k     = st.slider("Final K (after reranking)", 1, 10,  config.FINAL_K)

    st.divider()
    st.subheader("🤖 Router Agent")
    use_router = st.toggle(
        "Enable Router Agent (RAG + MCP)",
        value=True,
        help=(
            "When ON, the router classifies each query and answers from "
            "contract documents (RAG), company standards (MCP), or both. "
            "When OFF, only the RAG pipeline is used."
        ),
    )

    st.divider()
    col_build, col_clear = st.columns(2)
    with col_build:
        build_button = st.button("🔨 Build Index", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages   = []
            st.session_state.debug_data = {}
            st.rerun()

    if st.session_state.get("indexed"):
        st.success(f"✅ {st.session_state.chunk_count} chunks indexed")

# ── Index building ────────────────────────────────────────────────────────────

if build_button:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF.")
    elif overlap >= chunk_size:
        st.sidebar.error("Overlap must be smaller than chunk size.")
    else:
        with st.sidebar, st.spinner("Building index…"):
            all_chunks = []
            clear_index(config.COLLECTION_NAME)
            st.session_state.full_doc_texts = {}

            for uploaded_file in uploaded_files:
                os.makedirs("documents", exist_ok=True)
                path = os.path.join("documents", uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                pages = load_pdf(path)
                st.session_state.full_doc_texts[uploaded_file.name] = "\n\n".join(
                    p["text"] for p in pages
                )

                for page in pages:
                    chunks = chunk_document(page["text"], chunk_size, overlap)
                    build_index(chunks, config.COLLECTION_NAME, source=page["source"], page=page["page"])
                    all_chunks.extend(chunks)

            st.session_state.collection_name = config.COLLECTION_NAME
            st.session_state.indexed         = True
            st.session_state.chunk_count     = len(all_chunks)

        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"✅ Indexed **{len(all_chunks)} chunks** from "
                f"**{len(uploaded_files)} file(s)**. "
                "You can now ask me questions about your contracts."
            ),
        })
        st.rerun()

# ── Chat area ─────────────────────────────────────────────────────────────────

st.title("⚖️ Legal Contract Intelligence")

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Hello! I'm your **Legal Contract Intelligence** assistant.\n\n"
            "Upload your contract PDFs in the sidebar and click **Build Index**, "
            "then ask me anything — clauses, amendments, deadlines, defined terms, and more."
        )

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i in st.session_state.debug_data:
            debug = st.session_state.debug_data[i]
            render_route_badge(debug)
            render_mcp_tools(debug)
            render_full_doc_info(debug)
            render_sources(debug)

# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about your contracts…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.get("indexed"):
            reply = (
                "Please upload your contract PDFs and click **Build Index** "
                "in the sidebar before asking questions."
            )
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            llm_history = [
                m for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant") and not m["content"].startswith("✅")
            ]

            rag_kwargs = dict(
                search_mode=search_mode,
                conversation_history=llm_history,
                retrieval_k=retrieval_k,
                mmr_k=mmr_k,
                mmr_lambda=mmr_lambda,
                final_k=final_k,
            )

            with st.spinner("Thinking…"):
                if use_router:
                    result = router_route(
                        prompt,
                        full_doc_texts=st.session_state.get("full_doc_texts") or None,
                        **rag_kwargs,
                    )
                    response = result.to_dict() if hasattr(result, "to_dict") else result
                else:
                    response = answer_question(prompt, **rag_kwargs)

            if response.get("out_of_scope"):
                reply = f"🛡️ {response['answer']}"
                st.markdown(reply)
                st.caption(f"Reason: {response.get('reasoning', '')}")
            else:
                reply = response["answer"]
                render_route_badge(response)
                st.markdown(reply)
                next_idx = len(st.session_state.messages)
                st.session_state.debug_data[next_idx] = response
                render_mcp_tools(response)
                render_full_doc_info(response)
                render_sources(response)

            st.session_state.messages.append({"role": "assistant", "content": reply})
