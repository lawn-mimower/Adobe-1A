# PDF Outline Extractor (Round 1A)
This is a LightGBM model trained on 13 hand labeled PDFs with multilingual capabilties. 
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
The model is also trained on German, Japanese, Spanish, French and of course English, to make it script and language agnostic, though these languages are RTL (Right-To-Left), the model should do well on these languages.

The preprocessing step consolidates fragmented PDF spans to maintain semantic consistency, and effectively mitigates class imbalance, removing the need for SMOTE. 
