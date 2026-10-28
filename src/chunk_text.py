
import json
from pathlib import Path

INPUT_FILE = Path("data/extracted_text.json")
OUTPUT_FILE = Path("data/chunks.json")

CHUNK_SIZE = 250
OVERLAP = 40


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        pages = json.load(file)

    chunks = []

    for page in pages:
        text = page.get("text", "").strip()
        if not text:
            continue

        # Split text into paragraphs and sentences.
        paragraphs = [
            part.strip()
            for part in text.replace("\r", "").split("\n")
            if part.strip()
        ]

        current = ""

        for paragraph in paragraphs:
            # Break long paragraphs into smaller pieces.
            while len(paragraph) > CHUNK_SIZE:
                piece = paragraph[:CHUNK_SIZE]
                paragraph = paragraph[CHUNK_SIZE - OVERLAP:]

                if current:
                    chunks.append({
                        "source": page["source"],
                        "page": page["page"],
                        "text": current
                    })
                    current = ""

                chunks.append({
                    "source": page["source"],
                    "page": page["page"],
                    "text": piece
                })

            candidate = (
                current + "\n" + paragraph if current else paragraph
            )

            if len(candidate) <= CHUNK_SIZE:
                current = candidate
            else:
                chunks.append({
                    "source": page["source"],
                    "page": page["page"],
                    "text": current
                })

                current = (
                    current[-OVERLAP:] + " " + paragraph
                ).strip()

        if current:
            chunks.append({
                "source": page["source"],
                "page": page["page"],
                "text": current
            })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print("Total chunks created:", len(chunks))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
