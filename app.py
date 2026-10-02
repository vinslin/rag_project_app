import os
import json

import streamlit as st

from rag.ingestion.pdf_loader import load_pdf
from rag.ingestion.chunker import chunk_document
from rag.retrieval.vector_store import build_index, clear_index
from rag.pipeline import answer_question
from rag import config
from router.agent import route as router_route

CHAT_HISTORY_PATH = "data/chat_history.json"

st.set_page_config(
    page_title="Legal Contract Intelligence",
    page_icon="⚖️",
    layout="wide",
)

# ── Persistence helpers ───────────────────────────────────────────────────

def _load_history() -> list:
    try:
        if os.path.exists(CHAT_HISTORY_PATH):
            with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_history(messages: list) -> None:
    os.makedirs("data", exist_ok=True)
    try:
        with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ── Session state init ────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_data" not in st.session_state:
    # maps message-list index → response dict (for sources/debug expanders)
    st.session_state.debug_data = {}


# ── Debug renderer (sources tabs) ─────────────────────────────────────────

_ORIGIN_BADGE = {
    "vector": "🧠 Semantic",
    "bm25": "🔤 BM25",
    "both": "🔀 Both",
    "hybrid": "🔀 Hybrid",
}


_ROUTE_BADGE = {
    "rag":  ("🗂️ RAG", "#1f77b4"),
    "mcp":  ("🔌 MCP Standards", "#2ca02c"),
    "both": ("⚖️ RAG + MCP", "#9467bd"),
}


def _render_route_badge(response: dict) -> None:
    route = response.get("route")
    if not route:
        return
    label, color = _ROUTE_BADGE.get(route, (route, "#888"))
    reason = response.get("route_reason", "")
    st.markdown(
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:0.78em;font-weight:600">{label}</span>'
        + (f'&nbsp;&nbsp;<span style="color:#888;font-size:0.78em">{reason}</span>' if reason else ""),
        unsafe_allow_html=True,
    )


def _render_mcp_tools(response: dict) -> None:
    tools = response.get("mcp_tools_called", [])
    if not tools:
        return
    with st.expander(f"🔌 MCP tools called ({len(tools)})", expanded=False):
        for i, tool in enumerate(tools, 1):
            name = tool.get("name", "?") if isinstance(tool, dict) else tool.name
            args = tool.get("args", {}) if isinstance(tool, dict) else tool.args
            preview = tool.get("result_preview", "") if isinstance(tool, dict) else tool.result_preview
            st.markdown(f"**{i}. `{name}`**")
            if args:
                st.json(args)
            if preview:
                st.caption(f"Result preview: {preview[:200]}")
            if i < len(tools):
                st.divider()


def _render_sources(response: dict) -> None:
    """Render retrieval debug tabs inside a collapsed expander."""
    vector_chunks = response.get("vector_chunks", [])
    bm25_chunks   = response.get("bm25_chunks", [])
    mmr_chunks    = response.get("mmr_chunks", [])
    final_sources = response.get("sources", [])

    if not (vector_chunks or bm25_chunks or mmr_chunks or final_sources):
        return

    with st.expander("📚 Sources & retrieval debug", expanded=False):
        st.caption(response.get("reasoning", ""))

        tab_labels, tab_data = [], []
        if vector_chunks:
            tab_labels.append(f"🧠 Semantic ({len(vector_chunks)})")
            tab_data.append(("vector", vector_chunks))
        if bm25_chunks:
            tab_labels.append(f"🔤 BM25 ({len(bm25_chunks)})")
            tab_data.append(("bm25", bm25_chunks))
        if mmr_chunks:
            tab_labels.append(f"🎯 MMR ({len(mmr_chunks)})")
            tab_data.append(("mmr", mmr_chunks))
        tab_labels.append(f"✅ Reranked ({len(final_sources)})")
        tab_data.append(("final", final_sources))

        tabs = st.tabs(tab_labels)
        for tab, (tab_type, chunks) in zip(tabs, tab_data):
            with tab:
                if tab_type == "final":
                    for i, src in enumerate(chunks):
                        badge = _ORIGIN_BADGE.get(src.get("retrieved_by", ""), "")
                        label = (
                            f"#{i+1} {badge} — {src['source']} — Page {src['page']}"
                            + (f" — §{src['heading']}" if src.get("heading") else "")
                        )
                        with st.expander(label, expanded=(i == 0)):
                            parts = [f"**Distance:** `{src['distance']}`"]
                            if src.get("rrf_score") is not None:
                                parts.append(f"**RRF:** `{src['rrf_score']}`")
                            if src.get("mmr_score") is not None:
                                parts.append(f"**MMR:** `{src['mmr_score']}`")
                            if src.get("rerank_score") is not None:
                                parts.append(f"**Rerank:** `{src['rerank_score']}`")
                            st.caption(" · ".join(parts))
                            st.markdown("---")
                            st.markdown(src.get("text", ""))

                elif tab_type == "mmr":
                    for chunk in chunks:
                        badge = _ORIGIN_BADGE.get(chunk.get("origin", ""), "")
                        label = (
                            f"Rank {chunk['rank']} {badge} — {chunk['source']} — Page {chunk['page']}"
                            + (f" — §{chunk['heading']}" if chunk.get("heading") else "")
                        )
                        with st.expander(label, expanded=(chunk["rank"] == 1)):
                            if chunk.get("mmr_score") is not None:
                                st.caption(f"**MMR Score:** `{chunk['mmr_score']}`")
                            st.markdown("---")
                            st.markdown(chunk["text"])

                else:
                    score_label = "Distance" if tab_type == "vector" else "BM25 Score"
                    for chunk in chunks:
                        label = (
                            f"Rank {chunk['rank']} — {chunk['source']} — Page {chunk['page']}"
                            + (f" — §{chunk['heading']}" if chunk.get("heading") else "")
                        )
                        with st.expander(label, expanded=(chunk["rank"] == 1)):
                            st.caption(f"**{score_label}:** `{chunk['score']}`")
                            st.markdown("---")
                            st.markdown(chunk["text"])


# ── Sidebar ───────────────────────────────────────────────────────────────

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


# ── Index building ────────────────────────────────────────────────────────

if build_button:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF.")
    elif overlap >= chunk_size:
        st.sidebar.error("Overlap must be smaller than chunk size.")
    else:
        with st.sidebar, st.spinner("Building index…"):
            all_chunks = []
            clear_index(config.COLLECTION_NAME)

            for uploaded_file in uploaded_files:
                os.makedirs("documents", exist_ok=True)
                path = os.path.join("documents", uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                pages = load_pdf(path)
                for page in pages:
                    chunks = chunk_document(page["text"], chunk_size, overlap)
                    build_index(
                        chunks,
                        config.COLLECTION_NAME,
                        source=page["source"],
                        page=page["page"],
                    )
                    all_chunks.extend(chunks)

            st.session_state.collection_name = config.COLLECTION_NAME
            st.session_state.indexed         = True
            st.session_state.chunk_count     = len(all_chunks)

        notify = {
            "role": "assistant",
            "content": (
                f"✅ Indexed **{len(all_chunks)} chunks** from "
                f"**{len(uploaded_files)} file(s)**. "
                "You can now ask me questions about your contracts."
            ),
        }
        st.session_state.messages.append(notify)
        st.rerun()


# ── Chat area ─────────────────────────────────────────────────────────────

st.title("⚖️ Legal Contract Intelligence")

# Greeting shown only when there is no history at all
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Hello! I'm your **Legal Contract Intelligence** assistant.\n\n"
            "Upload your contract PDFs in the sidebar and click **Build Index**, "
            "then ask me anything — clauses, amendments, deadlines, defined terms, and more."
        )

# Replay conversation
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Attach route badge, MCP tools, and sources to assistant messages with debug data
        if msg["role"] == "assistant" and i in st.session_state.debug_data:
            debug = st.session_state.debug_data[i]
            _render_route_badge(debug)
            _render_mcp_tools(debug)
            _render_sources(debug)


# ── Chat input ────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about your contracts…"):

    # Show user bubble immediately
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
            # Build LLM-safe history: exclude system notifications and the current user turn
            llm_history = [
                m for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant")
                and not m["content"].startswith("✅")
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
                    result = router_route(prompt, **rag_kwargs)
                    response = result.to_dict() if hasattr(result, "to_dict") else result
                else:
                    response = answer_question(prompt, **rag_kwargs)

            if response.get("out_of_scope"):
                reply = f"🛡️ {response['answer']}"
                st.markdown(reply)
                st.caption(f"Reason: {response.get('reasoning', '')}")
            else:
                reply = response["answer"]
                _render_route_badge(response)
                st.markdown(reply)

                # Store debug keyed by the index this message will occupy
                next_idx = len(st.session_state.messages)
                st.session_state.debug_data[next_idx] = response
                _render_mcp_tools(response)
                _render_sources(response)

            st.session_state.messages.append({"role": "assistant", "content": reply})

