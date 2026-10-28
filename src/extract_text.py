
import pymupdf
import json
from pathlib import Path

pdf_path = Path("data/90 Day AI Engineer Roadmap.pdf")
output_path = Path("data/extracted_text.json")

pages_data = []

with pymupdf.open(pdf_path) as pdf:
    print("Total pages:", len(pdf))

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()

        print(f"Page {page_number}: {len(text)} characters extracted")

        if text:
            pages_data.append({
                "source": pdf_path.name,
                "page": page_number,
                "text": text
            })

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(pages_data, file, ensure_ascii=False, indent=2)

print("Pages containing text:", len(pages_data))
print("Saved extracted text to:", output_path)
