
# LLM-based Hallucination Detection & Knowledge-Grounded Correction

This repository contains a multi-stage pipeline for detecting hallucinations in Urdu–English code-switched inputs, retrieving supporting knowledge from a curated knowledge base, verifying claims, and producing corrected outputs. The pipeline is modular: preprocessing, hallucination detection, retrieval, verification, correction, and validation.

**Quick summary**
- Input: dataset in `data/dataset.json` (mixed Urdu-English examples).
- Output: stage outputs in `data/` (e.g., `stage1_output.json`, `stage5_output.json`).
- Key dirs: `pipeline/` (stages), `knowledge_base/` (KB build & embedding), `pipeline/utils/` (helpers).

**Requirements**
Install Python 3.9+ and then:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- For GPU-enabled `torch`, follow instructions at https://pytorch.org and install the appropriate CUDA build.
- On Windows, prefer installing `faiss-cpu` and some heavy packages via `conda`:

```bash
conda install -c conda-forge faiss-cpu
```

**Post-install setup**
- Download spaCy models (English and multilingual):

```bash
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm
```

- Download NLTK data (used for sentence tokenization):

```python
import nltk
nltk.download('punkt')
```

**Project structure (high level)**
- `main.py` — orchestrator to run the pipeline end-to-end.
- `pipeline/` — modular stage implementations:
  - `stage1_preprocessing.py` — normalization, code-switch tagging, tokenization.
  - `stage2_detection.py` — hallucination detection (embedding/clustering-based signals).
  - `stage3_retrieval.py` — retrieve evidence from the knowledge base using embedding search.
  - `stage4_verification.py` — verify retrieved facts and compute confidence scores.
  - `stage5_correction.py` — generate corrected outputs and DPO-format examples.
  - `stage6_validation.py` — optional validation using a model (e.g., generation-based checks).
- `knowledge_base/` — scripts to build the KB, scrape articles, extract entities, embed the KB, and train TransE/mapper models.
- `data/` — input dataset and stage outputs.

**Typical usage**

1. Build or prepare the knowledge base (if not already provided):

```bash
python knowledge_base/scrape_articles.py   # fetch raw text (uses requests + bs4)
python knowledge_base/extract_entities.py  # extract entities using spaCy
python knowledge_base/build_kb.py          # create triples/json entity files
python knowledge_base/embed_kb.py          # embed KB using sentence-transformers
```

2. Run the full pipeline (from root):

```bash
python main.py
```

3. Inspect outputs in `data/` (stage files):

- `stage1_output.json` — preprocessed texts
- `stage2_output.json` — detection flags/embeddings
- `stage3_output.json` — retrieval results
- `stage4_output.json` — verification + confidence scores
- `stage5_output.json` — corrected outputs

**Developer notes & troubleshooting**
- If you see memory/CUDA errors, reduce batch sizes or run on CPU (set `CUDA_VISIBLE_DEVICES=""`).
- `faiss-cpu` can be large; on Windows use conda for easier installation.
- If `spacy` model loading fails, ensure you downloaded the correct model name.
- Some experimental modules (e.g., `groq`) might be optional or externally provided — if an import fails, either install the package or comment out the dependent code.

**Extending the project**
- Add new knowledge sources under `knowledge_base/data_sources/` and update scraping & build scripts.
- Swap or fine-tune embedding models in `knowledge_base/embed_kb.py` and `pipeline/stage3_retrieval.py` by changing `SentenceTransformer` model names.

**Contact / Attribution**
This repository was created for research on hallucination detection and correction in mixed Urdu–English inputs. For questions or collaboration, open an issue or contact the maintainer.
## Setup Instructions

1. Create and activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

2. Upgrade pip
```bash
python -m pip install --upgrade pip
```

3. Install required packages
```bash
pip install pandas transformers sentencepiece torch

# Or alternatively:
pip install transformers
pip install torch sentencepiece pandas
```

4. Verify installation
```bash
python -c "import pandas, transformers, torch, sentencepiece; print('All packages installed successfully!')"
```

5. Run preprocessing
```bash
python -m pipeline.stage1_preprocessing
```
