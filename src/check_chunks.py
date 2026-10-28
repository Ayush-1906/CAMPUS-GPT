
import json

with open("data/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("Total chunks:", len(chunks))
print("\nChunk lengths:")

for i, chunk in enumerate(chunks, start=1):
    print(
        f"Chunk {i}: {len(chunk['text'])} characters | "
        f"Page {chunk['page']}"
    )
