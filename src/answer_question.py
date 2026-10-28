import os
from google import genai
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

# Load the Gemini API key from the project root
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

# File paths
chunks_path = Path("data/chunks.json")
embeddings_path = Path("data/embeddings.npy")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# Load saved chunks and embeddings
with open(chunks_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)

embeddings = np.load(embeddings_path)

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ask the user a question
question = input("Ask a question about your PDF: ").strip()

if not question:
    raise ValueError("Please enter a question.")

# Find the five most relevant chunks
query_embedding = model.encode(
    question,
    normalize_embeddings=True
)

scores = embeddings @ query_embedding
top_indices = np.argsort(scores)[::-1][:5]

# Prepare context with source and page information
context_parts = []

for index in top_indices:
    chunk = chunks[index]

    context_parts.append(
        f"Source: {chunk['source']}\n"
        f"Page: {chunk['page']}\n"
        f"Passage: {chunk['text']}"
    )

context = "\n\n---\n\n".join(context_parts)

# Ask Gemini to answer using the retrieved passages
client = genai.Client(api_key=api_key)

prompt = f"""
You are CampusGPT, a college information assistant.

Answer the question using ONLY the provided PDF passages.

Rules:
1. Do not invent facts that are not supported by the passages.
2. If the passages do not contain enough information, clearly say so.
3. Give a clear, beginner-friendly answer.
4. Cite supporting sources using the format [Source: filename, Page: number].
5. Only cite sources and pages included in the passages.

PDF PASSAGES:
{context}

USER QUESTION:
{question}

Write your answer with source citations.
"""

try:
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    print("\n" + "=" * 50)
    print("CAMPUSGPT ANSWER")
    print("=" * 50)
    print(response.text)

finally:
    client.close()
