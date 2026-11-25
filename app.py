"""Streamlit UI: speak a question -> hear a grounded answer.

Run:  streamlit run app.py
"""
import hashlib

import streamlit as st

import config
import llm
import retriever
import stt
import tts

st.set_page_config(page_title="Voice RAG", page_icon="🎙️")
st.title("🎙️ Speech-to-Speech RAG")

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Language", list(config.LANGUAGES.keys()))
    whisper_code, voice = config.LANGUAGES[language]
    show_sources = st.checkbox("Show sources", value=True)
    st.caption(f"LLM: {config.GROQ_MODEL}\n\nWhisper: {config.WHISPER_MODEL}")
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.last_audio_hash = None
        st.rerun()

# ---------- state ----------
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_audio_hash", None)

if retriever.index_size() == 0:
    st.warning("No index found. Put documents in `data/` and run `python ingest.py` first.")
    st.stop()


def handle_question(question: str):
    hits = retriever.retrieve(question)
    text = llm.answer(question, hits, language=language, history=st.session_state.history)
    try:
        audio = tts.synthesize(text, voice)
    except Exception as e:  # edge-tts is unofficial; degrade to text-only
        st.warning(f"Text-to-speech failed ({e}). Showing text only.")
        audio = b""
    st.session_state.history.append(
        {"question": question, "answer": text, "audio": audio, "sources": hits, "fresh": True}
    )


# ---------- input ----------
audio_in = st.audio_input("Ask your question")
typed = st.chat_input("...or type your question")

try:
    if audio_in is not None:
        data = audio_in.getvalue()
        digest = hashlib.md5(data).hexdigest()
        # The mic widget keeps its last recording across reruns: process each only once.
        if digest != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = digest
            with st.spinner("Transcribing..."):
                question = stt.transcribe(data, language=whisper_code)
            if not question:
                st.warning("I couldn't hear anything. Please try again.")
            else:
                with st.spinner("Thinking..."):
                    handle_question(question)
    elif typed:
        with st.spinner("Thinking..."):
            handle_question(typed)
except Exception as e:
    st.error(f"Something went wrong: {e}")

# ---------- conversation ----------
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["audio"]:
            st.audio(turn["audio"], format="audio/mp3", autoplay=turn.get("fresh", False))
        turn["fresh"] = False  # autoplay only on the first render
        if show_sources:
            with st.expander("Sources"):
                for h in turn["sources"]:
                    st.markdown(f"**{h['source']}** (page {h['page']}, similarity {h['score']})")
                    st.caption(h["text"][:300] + "...")
