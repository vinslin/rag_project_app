"""Streamlit UI renderer functions for route badges, MCP tools, and source debug."""

import streamlit as st

_ORIGIN_BADGE = {
    "vector": "🧠 Semantic",
    "bm25":   "🔤 BM25",
    "both":   "🔀 Both",
    "hybrid": "🔀 Hybrid",
}

_ROUTE_BADGE = {
    "rag":          ("🗂️ RAG",                  "#1f77b4"),
    "mcp":          ("🔌 MCP Standards",         "#2ca02c"),
    "both":         ("⚖️ RAG + MCP",             "#9467bd"),
    "full_doc":     ("📄 Full Document",          "#e87722"),
    "full_doc_mcp": ("📋 Full Compliance Review", "#d62728"),
}


def render_route_badge(response: dict) -> None:
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


def render_mcp_tools(response: dict) -> None:
    tools = response.get("mcp_tools_called", [])
    if not tools:
        return
    with st.expander(f"🔌 MCP tools called ({len(tools)})", expanded=False):
        for i, tool in enumerate(tools, 1):
            name    = tool.get("name", "?")          if isinstance(tool, dict) else tool.name
            args    = tool.get("args", {})            if isinstance(tool, dict) else tool.args
            preview = tool.get("result_preview", "")  if isinstance(tool, dict) else tool.result_preview
            st.markdown(f"**{i}. `{name}`**")
            if args:
                st.json(args)
            if preview:
                st.caption(f"Result preview: {preview[:200]}")
            if i < len(tools):
                st.divider()


def render_full_doc_info(response: dict) -> None:
    filenames = response.get("doc_filenames", [])
    tokens    = response.get("doc_token_estimate", 0)
    if not filenames:
        return
    with st.expander("📄 Document analysed", expanded=False):
        st.caption(f"~{tokens:,} estimated tokens")
        for f in filenames:
            st.markdown(f"- `{f}`")


def render_sources(response: dict) -> None:
    """Render retrieval debug tabs inside a collapsed expander."""
    vector_chunks = response.get("vector_chunks", [])
    bm25_chunks   = response.get("bm25_chunks",   [])
    mmr_chunks    = response.get("mmr_chunks",     [])
    final_sources = response.get("sources",        [])

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
