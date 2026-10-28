
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# File paths
input_path = Path("data/chunks.json")
output_path = Path("data/embeddings.npy")

# Load text chunks
with open(input_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)

if not chunks:
    raise ValueError("No chunks found. Run chunk_text.py first.")

# Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text into numerical vectors
texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

# Save vectors
np.save(output_path, embeddings)

print("\nEmbedding generation complete!")
print("Total chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("Saved to:", output_path)
