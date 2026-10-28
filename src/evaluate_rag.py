
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer
from google import genai


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

TEST_FILE = PROJECT_ROOT / "evaluation_questions.json"
PDF_FILE = PROJECT_ROOT / "data" / "resume.pdf"

SIMILARITY_THRESHOLD = 0.30
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_TIMEOUT_MS = 60000


# --------------------------------------------------
# 2. Validate files and API key
# --------------------------------------------------

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Evaluation file not found: {TEST_FILE}"
    )

if not PDF_FILE.exists():
    raise FileNotFoundError(
        f"PDF file not found: {PDF_FILE}"
    )

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set. Configure your Gemini API key "
        "in your environment before running this script."
    )

with open(TEST_FILE, "r", encoding="utf-8") as file:
    test_cases = json.load(file)

client = genai.Client(
    api_key=api_key,
    http_options={"timeout": GEMINI_TIMEOUT_MS},
)


# --------------------------------------------------
# 3. Load the embedding model
# --------------------------------------------------

print("Loading embedding model...", flush=True)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 4. Extract text chunks from the PDF
# --------------------------------------------------

def extract_chunks(pdf_path):
    chunks = []
    chunk_size = 500
    overlap = 80

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text").strip()

            if not page_text:
                continue

            lines = [
                line.strip()
                for line in page_text.splitlines()
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
                        "source": pdf_path.name,
                        "page": page_number,
                        "text": "\n".join(current_lines),
                    })

                    # Preserve a small amount of previous context.
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
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": "\n".join(current_lines),
                })

    return chunks


chunks = extract_chunks(PDF_FILE)

if not chunks:
    raise ValueError(
        "No text was extracted from resume.pdf"
    )


# --------------------------------------------------
# 5. Generate document embeddings
# --------------------------------------------------

print("Creating document embeddings...", flush=True)

embeddings = embedding_model.encode(
    [chunk["text"] for chunk in chunks],
    normalize_embeddings=True,
    show_progress_bar=False,
)

print(
    f"\nLoaded {len(test_cases)} tests "
    f"and {len(chunks)} text chunks.\n",
    flush=True,
)


# --------------------------------------------------
# 6. Evaluate retrieval and answer generation
# --------------------------------------------------

correct_retrieval_decisions = 0
total_keyword_checks = 0
correct_keyword_checks = 0

successful_generations = 0
generation_tests = 0

for number, test in enumerate(test_cases, start=1):
    print("=" * 60, flush=True)
    print(f"Test {number}: {test['question']}", flush=True)
    print(
        f"Expected behavior: {test['expected_behavior']}",
        flush=True,
    )

    # Retrieve the most similar chunk.
    query_embedding = embedding_model.encode(
        test["question"],
        normalize_embeddings=True,
    )

    scores = embeddings @ query_embedding
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    best_chunk = chunks[best_index]

    predicted_behavior = (
        "answer"
        if best_score >= SIMILARITY_THRESHOLD
        else "refuse"
    )

    decision_correct = (
        predicted_behavior == test["expected_behavior"]
    )

    if decision_correct:
        correct_retrieval_decisions += 1

    print(f"Best similarity: {best_score:.3f}", flush=True)
    print(f"Best source: {best_chunk['source']}", flush=True)
    print(f"Best page: {best_chunk['page']}", flush=True)
    print(
        f"Passes threshold: "
        f"{best_score >= SIMILARITY_THRESHOLD}",
        flush=True,
    )
    print(f"Predicted behavior: {predicted_behavior}", flush=True)
    print(f"Matches expectation: {decision_correct}", flush=True)

    # Refuse when retrieval similarity is too low.
    if predicted_behavior == "refuse":
        print(
            "Final answer: I cannot find sufficient relevant "
            "information in the document.",
            flush=True,
        )

        if test["expected_behavior"] == "refuse":
            print("Refusal test: PASS", flush=True)
        else:
            print(
                "Refusal test: FAIL - relevant information may "
                "have been missed by retrieval.",
                flush=True,
            )

        continue

    # Check expected keywords in the retrieved chunk.
    chunk_text = best_chunk["text"].lower()

    keywords = [
        word.lower()
        for word in test.get("expected_keywords", [])
    ]

    found_keywords = [
        word for word in keywords
        if word in chunk_text
    ]

    all_keywords_found = len(found_keywords) == len(keywords)

    if keywords:
        total_keyword_checks += 1

        if all_keywords_found:
            correct_keyword_checks += 1

    print(
        f"Expected keywords found: {found_keywords}",
        flush=True,
    )
    print(
        f"All expected keywords found: {all_keywords_found}",
        flush=True,
    )

    # Ask Gemini to answer using only the retrieved text.
    prompt = f"""
You are a document question-answering assistant.

Answer the question using only the document excerpt below.
Do not use outside knowledge or invent missing facts.
If the excerpt does not contain the answer, say:
"I cannot find that information in the provided document."

Document excerpt:
{best_chunk["text"]}

Question:
{test["question"]}
"""

    # Count only requests that actually reach Gemini.
    generation_tests += 1

    print(
        f"Requesting Gemini answer "
        f"(timeout: {GEMINI_TIMEOUT_MS // 1000} seconds)...",
        flush=True,
    )

    try:
        response = None

        # Try the request at most twice.
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
                break

            except Exception as error:
                error_message = str(error)

                is_temporary_error = (
                    "503" in error_message
                    or "UNAVAILABLE" in error_message
                    or "429" in error_message
                    or "RESOURCE_EXHAUSTED" in error_message
                    or "504" in error_message
                    or "DEADLINE_EXCEEDED" in error_message
                )

                if attempt == 0 and is_temporary_error:
                    print(
                        "Temporary Gemini error. Retrying once...",
                        flush=True,
                    )
                    time.sleep(2)
                else:
                    raise

        generated_answer = (response.text or "").strip()

        if generated_answer:
            successful_generations += 1
            print(
                f"Gemini answer: {generated_answer}",
                flush=True,
            )
        else:
            print(
                "Gemini returned an empty answer.",
                flush=True,
            )

    except Exception as error:
        print(
            f"Gemini generation failed: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

    print("-" * 60, flush=True)


# --------------------------------------------------
# 7. Final evaluation summary
# --------------------------------------------------

print("\n" + "=" * 60, flush=True)
print("EVALUATION SUMMARY", flush=True)
print("=" * 60, flush=True)

print(
    f"Successful Gemini generations: "
    f"{successful_generations}/{generation_tests}",
    flush=True,
)

print(
    f"Retrieval decision accuracy: "
    f"{correct_retrieval_decisions}/{len(test_cases)}",
    flush=True,
)

if total_keyword_checks:
    print(
        f"Keyword check pass rate: "
        f"{correct_keyword_checks}/{total_keyword_checks}",
        flush=True,
    )

print(
    "\nNote: Retrieval decision accuracy measures whether the "
    "threshold chose the expected behavior. It does not measure "
    "the factual correctness of Gemini's generated answers.",
    flush=True,
)

client.close()
