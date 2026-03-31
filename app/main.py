from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io
from PIL import Image
import uvicorn
import logging

from app.ml_model import MathOCRModel, HandwritingModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global dictionary to store models, mimicking an MLOps Model Registry in memory
ml_models = {}

@asynccontextmanager
async def lifespan(current_app: FastAPI):
    # Load the ML model gracefully when the server starts up
    logger.info("Starting up MLOps pipeline...")
    try:
        # 1. Initialize Handwriting model first
        handwriting_model = HandwritingModel()
        ml_models["handwriting"] = handwriting_model
        
        # 2. Inject Handwriting reader into MathOCRModel for intelligent fallback
        ml_models["math_ocr"] = MathOCRModel(handwriting_reader=handwriting_model.reader)
        
        logger.info("Intelligent Ensemble Models successfully loaded and registered.")
    except Exception as e:
        logger.error(f"Failed to load ML Models: {e}")
        
    yield # The app runs here
    
    # Clean up the ML model and release resources on shutdown
    ml_models.clear()
    logger.info("Shutting down MLOps pipeline...")

app = FastAPI(
    title="Intelligent Ensemble Math Solver API",
    description="A FastAPI app that uses an ensemble of pix2tex and EasyOCR to solve both printed and handwritten math problems.",
    version="1.2.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    """Endpoint for monitoring pipeline health"""
    math_loaded = "math_ocr" in ml_models
    handwriting_loaded = "handwriting" in ml_models
    return {
        "status": "healthy" if (math_loaded and handwriting_loaded) else "loading/failed",
        "models": {
            "math_ocr": math_loaded,
            "handwriting": handwriting_loaded
        }
    }

@app.post("/solve")
async def solve_math_equation(file: UploadFile = File(...)):
    """
    Endpoint for uploading an image containing a math problem.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a mathematical image.")
    
    try:
        # Load the uploaded bytes into memory and convert to PIL Image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image file: {str(e)}")
    
    # Retrieve the model from our global context (Model Registry)
    model = ml_models.get("math_ocr")
    if not model:
        raise HTTPException(status_code=503, detail="Model is currently loading or unavailable.")
    
    try:
        # Run inference and rule-based logic
        prediction = model.predict(image)
        return {
            "status": "success",
            "filename": file.filename,
            "data": prediction
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
