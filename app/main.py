import io
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.predictor import GeoPredictor
from src.config import (
    MODEL_PATH,
    REFERENCE_EMBEDDINGS_PATH,
    METADATA_PATH,
    TOP_GEOCELLS,
    RETRIEVAL_K,
    DBSCAN_EPS_KM,
    DBSCAN_MIN_SAMPLES,
    SIMILARITY_TEMPERATURE,
    USE_CLASSIFIER_PRIOR,
    ALPHA,
    BETA,
    METHOD
)

# Global Predictor Instance
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("Loading GeoPredictor...")
    predictor = GeoPredictor(
        checkpoint_path=MODEL_PATH,
        embeddings_path=REFERENCE_EMBEDDINGS_PATH,
        metadata_path=METADATA_PATH,
        top_geocells=TOP_GEOCELLS,
        retrieval_k=RETRIEVAL_K,
        method=METHOD,
        temperature=SIMILARITY_TEMPERATURE,
        eps_km=DBSCAN_EPS_KM,
        min_samples=DBSCAN_MIN_SAMPLES,
        use_classifier_prior=USE_CLASSIFIER_PRIOR,
        alpha=ALPHA,
        beta=BETA
    )
    print("GeoPredictor loaded successfully.")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Mount the static directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("app/static/index.html")

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "device": str(predictor._model.device) if predictor else "none"
    }

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model is still loading. Please try again later.")
        
    if not image.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    ext = image.filename.lower().split(".")[-1]
    if ext not in ["jpg", "jpeg", "png", "webp"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a JPG, PNG, or WEBP.")
    
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
    try:
        # Run prediction on existing pipeline
        result = predictor.predict(pil_image)
        
        # Structure the response
        response = {
            "latitude": result["latitude"],
            "longitude": result["longitude"],
            "method": result["method"],
            "top_similarity": result["top_similarity"],
            "candidate_count": result["candidate_count"],
            "retrieval_count": result["retrieval_k"],
            "cluster": {
                "size": result["winning_cluster_size"],
                "score": result["winning_cluster_score"],
                "num_clusters": result["num_clusters"],
                "noise_points": result["num_noise_points"],
                "median_radius_km": result["median_cluster_radius_km"]
            },
            "top_geocells": result["top_geocells"],
            "timing": {
                "model_ms": result["timing"].get("inference_ms", 0),
                "retrieval_ms": result["timing"].get("retrieval_ms", 0),
                "clustering_ms": result["timing"].get("clustering_ms", 0),
                "total_ms": result["timing"].get("total_ms", 0)
            }
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
