"""Grounded answer generation using the Groq API."""
from groq import Groq

import config

SYSTEM_PROMPT = """You are a voice assistant that answers questions using ONLY the provided context.

Rules:
- If the answer is not in the context, say you don't have that information. Do not guess.
- Reply in {language}.
- Your answer will be spoken aloud: use 2-3 short, conversational sentences.
- Never use bullet points, markdown, headings, emojis, or URLs.
- Write numbers and dates the way a person would say them."""


def _client() -> Groq:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
    return Groq(api_key=config.GROQ_API_KEY)


def build_context(hits) -> str:
    return "\n\n".join(
        f"[{i}] (source: {h['source']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )


def answer(question: str, hits, language: str = "English", history=None) -> str:
    """history: list of {"question":..., "answer":...}; only the last 3 are used."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(language=language)}]
    for turn in (history or [])[-3:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({
        "role": "user",
        "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}",
    })
    resp = _client().chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()
