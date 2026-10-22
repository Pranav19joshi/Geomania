# GeoMania

Street-image geolocation system using a fine-tuned OpenCLIP ViT-L/14, adaptive GeoCells, embedding retrieval, geographic clustering and similarity-weighted GPS estimation.

## Architecture

The system predicts geographic coordinates from street images through a multi-stage pipeline:

```
Street Image
    ↓
Fine-tuned OpenCLIP ViT-L/14
    ↓
768-D normalized embedding
    ↓
512-class GeoCell classifier
    ↓
Top-10 GeoCells
    ↓
~1,500–2,000 reference candidates
    ↓
Cosine similarity
    ↓
Top-100 reference images
    ↓
Haversine DBSCAN
    ↓
Similarity-weighted spherical centroid
    ↓
Latitude / Longitude
```

### Model Details
- **Base Model:** OpenCLIP ViT-L/14 (DataComp XL — `datacomp_xl_s13b_b90k`)
- **Fine-Tuning:** Partially fine-tuned on geotagged street imagery
- **Classification:** 512 adaptive GeoCells
- **Embedding dimension:** 768-D L2-normalised
- **Reference Database:** 91,178 reference embeddings

### Dataset Split
| Split      | Images  |
|------------|--------:|
| Train      |  91,178 |
| Validation |  10,727 |
| Test       |   5,364 |
| **Total**  | **107,269** |

### GeoCell Classifier Validation Performance
> **Note:** These are GeoCell *classification* metrics, not final GPS distance accuracies.

| Metric | Accuracy |
|--------|----------|
| Top-1  | 42.43%   |
| Top-5  | 83.63%   |
| Top-10 | 91.60%   |

---

## Model Artifacts

The two large trained artifacts are hosted on Hugging Face:

**Repository:** https://huggingface.co/JoshiPranav19/GeoMania

| File | Size | Contents |
|------|------|----------|
| `best_finetuned_geolocation.pt` | ~1.71 GB | Fine-tuned OpenCLIP ViT-L/14 visual encoder + 512-class GeoCell classifier head. Best checkpoint: epoch 3. |
| `final_reference_embeddings.pt` | ~282 MB | 91,178 reference embeddings (768-D), image IDs, GeoCell IDs. |

### Automatic Artifact Resolution

GeoMania resolves artifacts **locally first, Hugging Face only as fallback**:

```
Application starts
       ↓
Check  models/best_finetuned_geolocation.pt
       ↓
Local exists?
  YES → use local file directly (no network activity)
  NO  → download from JoshiPranav19/GeoMania and cache via huggingface_hub
       ↓
Check  data/final_reference_embeddings.pt
       ↓
Local exists?
  YES → use local file directly (no network activity)
  NO  → download from JoshiPranav19/GeoMania and cache via huggingface_hub
       ↓
Load model + embeddings → start application
```

**Development machine:** uses your existing local files immediately — zero download overhead.

**New machine / deployment:** downloads both artifacts from Hugging Face once, caches them in `~/.cache/huggingface/hub/`, and reuses the cache on every subsequent startup.

> You do **not** need to manually download anything. Just run the application and let it handle the rest.

### Private Repository Support

If the Hugging Face repository is made private, set the `HF_TOKEN` environment variable:

```bash
# Windows PowerShell
$env:HF_TOKEN = "hf_your_token_here"

# Linux / macOS
export HF_TOKEN=hf_your_token_here
```

Do **not** hardcode tokens in source code.

---

## Setup and Usage

### 1. Installation

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running the Application

Start the FastAPI server from the repository root:

```bash
uvicorn app.main:app --reload
```

Then open your browser and navigate to:
http://127.0.0.1:8000

On first run on a new machine, the application will automatically download the model and embeddings from Hugging Face (one-time, ~2 GB total). Subsequent startups reuse the local cache.

---

## Project Structure

```
GeoMania/
├── app/
│   ├── main.py              # FastAPI application + lifespan startup
│   └── static/              # Frontend (HTML/CSS/JS)
├── src/
│   ├── artifacts.py         # Artifact resolver (LOCAL FIRST → HF fallback)
│   ├── config.py            # Path constants + ML hyperparameters
│   ├── model.py             # GeolocationModel (OpenCLIP + classifier)
│   ├── retrieval.py         # ReferenceRetriever (cosine similarity search)
│   ├── predictor.py         # GeoPredictor (end-to-end pipeline)
│   └── clustering.py        # Haversine DBSCAN + spherical centroid
├── data/
│   ├── final_metadata.csv       # Image GPS + country metadata
│   ├── location_metadata.csv    # Additional location data
│   ├── geocells.csv             # GeoCell definitions
│   └── geocell_tree.json        # GeoCell spatial tree
├── models/                  # Large artifacts (gitignored; see HF above)
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** `models/*.pt` and `data/*.pt` are excluded from Git. They are either present as local files or downloaded automatically from Hugging Face.
