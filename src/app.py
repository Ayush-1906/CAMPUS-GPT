
import os
import json
from pathlib import Path

import numpy as np
import pymupdf
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings.npy"

SIMILARITY_THRESHOLD = 0.30
GEMINI_MODEL = "gemini-3.6-flash"


# --------------------------------------------------
# 2. Process PDF files
# --------------------------------------------------

def process_uploaded_pdf(pdf_bytes, filename):
    """Extract searchable text chunks with source and page details."""

    chunks = []
    chunk_size = 500
    overlap = 80

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            current_lines = []
            current_length = 0

            for line in lines:
                if (
                    current_lines
                    and current_length + len(line) + 1 > chunk_size
                ):
                    chunks.append({
                        "source": filename,
                        "page": page_number,
                        "text": "\n".join(current_lines),
                    })

                    # Keep some previous text for context.
                    overlap_lines = []
                    overlap_length = 0

                    for old_line in reversed(current_lines):
                        line_length = len(old_line) + 1

                        if overlap_length + line_length > overlap:
                            break

                        overlap_lines.insert(0, old_line)
                        overlap_length += line_length

                    current_lines = overlap_lines
                    current_length = overlap_length

                current_lines.append(line)
                current_length += len(line) + 1

            if current_lines:
                chunks.append({
                    "source": filename,
                    "page": page_number,
                    "text": "\n".join(current_lines),
                })

    return chunks


# --------------------------------------------------
# 3. Streamlit interface
# --------------------------------------------------

st.set_page_config(
    page_title="CampusGPT",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 CampusGPT")
st.caption(
    "Ask questions about your documents and get source-backed answers."
)

uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select multiple PDF files to search across them together.",
)


# --------------------------------------------------
# 4. Check API key and default search resources
# --------------------------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error(
        "Gemini API key not found. "
        "Check the .env file in your project root."
    )
    st.stop()


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


try:
    embedding_model = load_embedding_model()
except Exception as error:
    st.error(f"Could not load the embedding model: {error}")
    st.stop()


# --------------------------------------------------
# 5. Load default chunks or process uploaded PDFs
# --------------------------------------------------

if uploaded_files:
    chunks = []
    empty_pdfs = []

    try:
        for uploaded_file in uploaded_files:
            pdf_chunks = process_uploaded_pdf(
                uploaded_file.getvalue(),
                uploaded_file.name,
            )

            if pdf_chunks:
                chunks.extend(pdf_chunks)
            else:
                empty_pdfs.append(uploaded_file.name)

        if empty_pdfs:
            st.warning(
                "No readable text was found in: "
                + ", ".join(empty_pdfs)
                + ". These files may be scanned PDFs."
            )

        if not chunks:
            st.error(
                "No searchable text was found in the uploaded PDFs."
            )
            st.stop()

        with st.spinner("Creating searchable document embeddings..."):
            embeddings = embedding_model.encode(
                [chunk["text"] for chunk in chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        source_names = list(dict.fromkeys(
            chunk["source"] for chunk in chunks
        ))

        st.success(
            f"Loaded {len(source_names)} PDF(s) with "
            f"{len(chunks)} searchable text chunks."
        )

        with st.expander("View uploaded documents"):
            for filename in source_names:
                count = sum(
                    1 for chunk in chunks
                    if chunk["source"] == filename
                )
                st.write(f"📄 {filename} — {count} chunks")

    except Exception as error:
        st.error(f"Could not process the uploaded PDFs: {error}")
        st.stop()

else:
    # Use the existing prepared documents when no PDF is uploaded.
    if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        st.info("Upload one or more PDFs to get started.")
        st.stop()

    try:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
            chunks = json.load(file)

        embeddings = np.load(EMBEDDINGS_PATH)

        if len(chunks) != len(embeddings):
            st.error(
                "The number of chunks and embeddings does not match. "
                "Regenerate your chunks and embeddings."
            )
            st.stop()

    except Exception as error:
        st.error(f"Could not load the default search resources: {error}")
        st.stop()


# --------------------------------------------------
# 6. Ask questions across all available documents
# --------------------------------------------------

question = st.text_input(
    "Your question",
    placeholder="e.g. What is a class?",
)

if st.button("Ask CampusGPT", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching all available documents..."):
        query_embedding = embedding_model.encode(
            question,
            normalize_embeddings=True,
        )

        scores = embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:5]

        if len(top_indices) == 0:
            st.warning("No searchable passages were found.")
            st.stop()

        best_score = float(scores[top_indices[0]])

        st.caption(
            f"Best retrieval similarity: {best_score:.3f}"
        )

        if best_score < SIMILARITY_THRESHOLD:
            st.warning(
                "I couldn't find enough relevant information "
                "in the available documents to answer this question."
            )
            st.stop()

        context_parts = []

        for index in top_indices:
            chunk = chunks[index]

            context_parts.append(
                f"Source: {chunk['source']}\n"
                f"Page: {chunk['page']}\n"
                f"Passage: {chunk['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)


    # --------------------------------------------------
    # 7. Generate a source-backed answer
    # --------------------------------------------------

    prompt = f"""
You are CampusGPT, a document question-answering assistant.

Answer the user's question using only the supplied passages.
The passages may come from multiple uploaded PDF documents.

Do not invent facts or use outside knowledge.
If the passages do not contain enough information, say so clearly.

Cite supporting information using this format:
[Source: filename, Page: number]

Use only filenames and page numbers provided in the passages.
Cite each relevant source only once when practical.
Do not repeat identical citations unnecessarily.
Write in clear, beginner-friendly language.

DOCUMENT PASSAGES:
{context}

USER QUESTION:
{question}
"""

    try:
        with st.spinner("Generating your answer..."):
            client = genai.Client(
                api_key=api_key,
                http_options={"timeout": 60000},
            )

            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
            finally:
                client.close()

        if response.text and response.text.strip():
            st.subheader("Answer")
            st.markdown(response.text)
        else:
            st.warning(
                "The model returned an empty answer. Please try again."
            )

        with st.expander("View retrieved passages"):
            for rank, index in enumerate(top_indices, start=1):
                chunk = chunks[index]

                st.markdown(
                    f"**Result {rank} — Score: {scores[index]:.3f}**"
                )

                st.caption(
                    f"Source: {chunk['source']} | "
                    f"Page: {chunk['page']}"
                )

                st.write(chunk["text"])
                st.divider()

    except Exception as error:
        st.error(f"CampusGPT encountered an error: {error}")
