# PDF Outline Extractor (Round 1A)

Extracts a clean, hierarchical outline (Title, H1, H2, H3 + page numbers) from PDFs (≤50 pages), fully offline, CPU-only, under the hackathon constraints.

## What It Does
- Reads every `*.pdf` from `/app/input`.
- Produces matching `*.json` files in `/app/output`:
  ```json
  {
    "title": "Understanding AI",
    "outline": [
      {"level":"H1","text":"Introduction","page":1},
      {"level":"H2","text":"What is AI?","page":2},
      {"level":"H3","text":"History of AI","page":3}
    ]
  }
