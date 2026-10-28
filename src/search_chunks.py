
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

chunks_path = Path("data/chunks.json")
embeddings_path = Path("data/embeddings.npy")

# Load the saved chunks and their embeddings
with open(chunks_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)

embeddings = np.load(embeddings_path)

# Load the same model used to create the embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

question = input("Ask a question about your PDF: ")

# Convert the question into an embedding
query_embedding = model.encode(
    question,
    normalize_embeddings=True
)

# Compare the question with every chunk
scores = embeddings @ query_embedding

# Get the 3 most relevant chunks
top_indices = np.argsort(scores)[::-1][:5]

print("\nMost relevant passages:\n")

for rank, index in enumerate(top_indices, start=1):
    chunk = chunks[index]

    print(f"--- Result {rank} ---")
    print(f"Relevance score: {scores[index]:.3f}")
    print(f"Source: {chunk['source']}")
    print(f"Page: {chunk['page']}")
    print(chunk["text"])
    print()
