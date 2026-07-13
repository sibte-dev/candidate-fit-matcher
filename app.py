"""
Candidate-Fit Matcher — a small RAG (Retrieval-Augmented Generation) demo.

Given a job description and a batch of resumes, this app:
1. Embeds the job description and each resume using a sentence-transformer model.
2. Retrieves and ranks candidates by semantic similarity (the "R" in RAG).
3. Generates a short, grounded explanation of *why* each top candidate fits,
   using an LLM if a Hugging Face token is available, or a transparent
   keyword-overlap fallback if not (so the demo always works with zero setup).

Built by Syed Sibte Hassan Shah as a portfolio project connecting a
recruiting background with applied RAG / NLP engineering.
"""

import os
import re
from dataclasses import dataclass

import gradio as gr
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Embedding backend
# ---------------------------------------------------------------------------
# We lazy-load the embedding model so the app boots instantly and only pays
# the download/load cost the first time a match is actually requested.

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        # Small, fast, CPU-friendly model — good default for a free Space.
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    return np.array(model.encode(texts, normalize_embeddings=True))


# ---------------------------------------------------------------------------
# Optional LLM-generated explanations via Hugging Face Inference
# ---------------------------------------------------------------------------
# If HF_TOKEN is set as a Space secret, we use it to generate a natural
# explanation of the match. Otherwise we fall back to a deterministic,
# fully transparent keyword-overlap explanation. Either path is grounded
# only in the actual resume/job text — no hallucinated claims.

HF_TOKEN = os.environ.get("HF_TOKEN")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on",
    "is", "are", "as", "at", "by", "be", "this", "that", "will", "we",
    "you", "your", "our", "it", "from", "have", "has", "who", "years",
    "experience", "role", "job", "candidate", "candidates", "team",
}


def keyword_overlap_explanation(job_text: str, resume_text: str, top_n: int = 6) -> str:
    """Deterministic, no-API-key fallback: surfaces shared meaningful terms."""

    def tokenize(text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z][a-zA-Z\-\+]{2,}", text.lower())
        return {w for w in words if w not in STOPWORDS}

    job_terms = tokenize(job_text)
    resume_terms = tokenize(resume_text)
    shared = sorted(job_terms & resume_terms)[:top_n]

    if not shared:
        return "No strong keyword overlap detected — similarity score is based on semantic meaning rather than shared terms."
    return "Shared key terms with the job description: " + ", ".join(shared) + "."


def llm_explanation(job_text: str, resume_text: str) -> str:
    """Use an HF-hosted instruction model to explain the match, grounded in the text."""
    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=HF_TOKEN)
        prompt = (
            "You are a recruiting assistant. Based ONLY on the text below, "
            "write a 2-sentence, factual explanation of why this candidate's "
            "resume fits the job description. Do not invent details that are "
            "not present in the text.\n\n"
            f"JOB DESCRIPTION:\n{job_text}\n\n"
            f"RESUME:\n{resume_text}\n\n"
            "Explanation:"
        )
        result = client.text_generation(
            prompt,
            model="HuggingFaceH4/zephyr-7b-beta",
            max_new_tokens=120,
            temperature=0.3,
        )
        return result.strip()
    except Exception as e:  # noqa: BLE001 — any failure, fall back gracefully
        return keyword_overlap_explanation(job_text, resume_text) + f"\n\n(LLM explanation unavailable: {e})"


def explain_match(job_text: str, resume_text: str) -> str:
    if HF_TOKEN:
        return llm_explanation(job_text, resume_text)
    return keyword_overlap_explanation(job_text, resume_text)


# ---------------------------------------------------------------------------
# Core matching logic
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    name: str
    text: str


def parse_resumes(raw: str) -> list[Candidate]:
    """
    Resumes are separated by a line of three or more dashes: ---
    Optionally, the first line of each block can be 'Name: <candidate name>'.
    """
    blocks = re.split(r"\n-{3,}\n", raw.strip())
    candidates = []
    for i, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue
        name = f"Candidate {i + 1}"
        first_line = block.split("\n", 1)[0]
        if first_line.lower().startswith("name:"):
            name = first_line.split(":", 1)[1].strip()
            block = block.split("\n", 1)[1] if "\n" in block else ""
        candidates.append(Candidate(name=name, text=block))
    return candidates


def match_candidates(job_description: str, resumes_raw: str, top_k: int):
    if not job_description.strip():
        return "Please enter a job description.", None
    candidates = parse_resumes(resumes_raw)
    if not candidates:
        return "Please enter at least one resume (separate multiple with a line of `---`).", None

    texts = [job_description] + [c.text for c in candidates]
    embeddings = embed_texts(texts)
    job_vec = embeddings[0:1]
    resume_vecs = embeddings[1:]

    sims = cosine_similarity(job_vec, resume_vecs)[0]
    ranked_idx = np.argsort(-sims)[: min(top_k, len(candidates))]

    rows = []
    details = []
    for rank, idx in enumerate(ranked_idx, start=1):
        cand = candidates[idx]
        score = float(sims[idx])
        rows.append([rank, cand.name, f"{score:.3f}"])
        explanation = explain_match(job_description, cand.text)
        details.append(f"### {rank}. {cand.name} — similarity {score:.3f}\n{explanation}")

    return "\n\n".join(details), rows


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLE_JOB = (
    "Senior Recruiter needed with experience in full-lifecycle international "
    "hiring, ATS management (Greenhouse, Bullhorn), Boolean search, and "
    "exposure to AI-powered sourcing tools. EMEA and APAC coordination a plus."
)

EXAMPLE_RESUMES = """Name: Amina Farouk
5 years as an international recruiter managing full-lifecycle hiring across EMEA. Skilled in Boolean search, Greenhouse ATS, and LinkedIn Recruiter. Coordinated hiring across 4 time zones.
---
Name: Wei Zhang
Backend engineer with 6 years building REST APIs in Node.js and Python. Some exposure to internal hiring panels but no formal recruiting role.
---
Name: Carlos Mendes
Talent acquisition specialist with 3 years experience, APAC-focused sourcing, AI sourcing tool adoption, and Bullhorn ATS documentation."""

with gr.Blocks(title="Candidate-Fit Matcher (RAG demo)") as demo:
    gr.Markdown(
        """
        # 🔍 Candidate-Fit Matcher
        A small **Retrieval-Augmented Generation (RAG)** demo: paste a job description and a batch
        of resumes, and it retrieves the best-matching candidates by semantic similarity, then
        explains the match — grounded only in the text you provide, never invented.

        Separate multiple resumes with a line containing `---`. Optionally start each resume
        block with `Name: <candidate name>`.
        """
    )

    with gr.Row():
        with gr.Column():
            job_input = gr.Textbox(
                label="Job Description",
                lines=6,
                value=EXAMPLE_JOB,
            )
            resumes_input = gr.Textbox(
                label="Resumes (separate with a line of ---)",
                lines=12,
                value=EXAMPLE_RESUMES,
            )
            top_k = gr.Slider(1, 10, value=3, step=1, label="How many top candidates to show")
            run_btn = gr.Button("Find best matches", variant="primary")

        with gr.Column():
            results_table = gr.Dataframe(
                headers=["Rank", "Candidate", "Similarity"],
                label="Ranked candidates",
            )
            explanations = gr.Markdown(label="Why they fit")

    run_btn.click(
        fn=match_candidates,
        inputs=[job_input, resumes_input, top_k],
        outputs=[explanations, results_table],
    )

    gr.Markdown(
        """
        ---
        **How it works:** resumes and the job description are embedded with
        `sentence-transformers/all-MiniLM-L6-v2`, ranked by cosine similarity, and each match is
        explained either by a hosted LLM (if an `HF_TOKEN` secret is configured on this Space) or
        by a transparent keyword-overlap fallback — so the demo is honest about what's a real
        model output versus a heuristic.
        """
    )

if __name__ == "__main__":
    demo.launch()
