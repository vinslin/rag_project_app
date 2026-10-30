import os

from dotenv import load_dotenv
load_dotenv()

from core.logging_setup import configure_logging
configure_logging()

import streamlit as st
from groq import Groq

from core import config
from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_document
from retrieval.vector_store import build_index, clear_index
from router.agent import route as router_route

_NO_DOC_SYSTEM = (
    "You are a friendly Legal Contract Intelligence assistant. "
    "The user has not yet uploaded any contract documents. "
    "Respond warmly and helpfully — acknowledge what they said or asked, "
    "and naturally guide them to upload a contract PDF so you can assist them properly. "
    "Keep it conversational, short (2-3 sentences), and never repeat the same phrasing twice. "
    "Do NOT answer legal questions without documents — redirect creatively."
)


def _no_doc_reply(user_message: str) -> str:
    """Call Groq to generate a unique, context-aware no-document reply."""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model=config.GENERATION_MODEL,
            messages=[
                {"role": "system", "content": _NO_DOC_SYSTEM},
                {"role": "user", "content": user_message},
            ],
        )
        return resp.choices[0].message.content or "Please upload a contract PDF in the sidebar to get started!"
    except Exception:
        return "Please upload your contract PDFs in the sidebar — I'll be ready to help the moment they're indexed!"

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
if "last_uploaded_names" not in st.session_state:
    st.session_state.last_uploaded_names = []

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
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages   = []
        st.session_state.debug_data = {}
        st.rerun()

    if st.session_state.get("indexed"):
        st.success(f"✅ {st.session_state.chunk_count} chunks indexed")

# ── Auto index building when files change ─────────────────────────────────────

current_names = sorted(f.name for f in uploaded_files) if uploaded_files else []
if current_names and current_names != st.session_state.last_uploaded_names:
    st.session_state.last_uploaded_names = current_names
    try:
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
                    chunks = chunk_document(page["text"], config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS)
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
    except Exception:
        st.sidebar.error("Something went wrong. Please try again in a moment.")

# ── Chat area ─────────────────────────────────────────────────────────────────

st.title("⚖️ Legal Contract Intelligence")

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "### Welcome to Legal Contract Intelligence! ⚖️\n\n"
            "I'm your AI-powered contract analyst. \n\n"
            "**Get started:** upload one or more contract PDFs using the sidebar, then ask me anything."
        )

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about your contracts…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.get("indexed"):
            with st.spinner("Thinking…"):
                reply = _no_doc_reply(prompt)
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            llm_history = [
                m for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant") and not m["content"].startswith("✅")
            ]

            rag_kwargs = dict(
                search_mode="hybrid",
                conversation_history=llm_history,
                retrieval_k=config.RETRIEVAL_K,
                mmr_k=config.MMR_K,
                mmr_lambda=config.MMR_LAMBDA,
                final_k=config.FINAL_K,
            )

            try:
                with st.spinner("Thinking…"):
                    result = router_route(
                        prompt,
                        full_doc_texts=st.session_state.get("full_doc_texts") or None,
                        **rag_kwargs,
                    )
                    response = result.to_dict() if hasattr(result, "to_dict") else result

                if response.get("out_of_scope"):
                    reply = f"🛡️ {response['answer']}"
                    st.markdown(reply)
                    st.caption(f"Reason: {response.get('reasoning', '')}")
                else:
                    reply = response["answer"]
                    st.markdown(reply)

                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception:
                reply = "Something went wrong. Please try again in a moment."
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
