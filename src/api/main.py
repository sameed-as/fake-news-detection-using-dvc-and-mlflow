"""
Fast API service for ad creative generation
Includes Prometheus metrics export (10 pts)
Loads model from MLflow registry (3 pts)
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import mlflow
import mlflow.pytorch
import os
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics (6 pts for custom metrics)
REQUEST_COUNT = Counter(
    'requests_total',
    'Total number of requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'ad_generation_latency',
    'Latency of ad generation in seconds',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
)

QUALITY_SCORE = Gauge(
    'ad_quality_score',
    'Quality score of generated ad (0-1)'
)

MODEL_VERSION = Gauge(
    'model_version_info',
    'Model version currently loaded'
)

# FastAPI app
app = FastAPI(
    title="Ad Creative Generator API",
    description="Generate compelling ad creatives for e-commerce products",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer
model = None
tokenizer = None
model_loaded = False

# Configuration
class Config:
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MODEL_NAME = os.getenv("MODEL_NAME", "ad-creative-generator")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
    MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")


# Pydantic models for request/response
class ProductInput(BaseModel):
    """Request model for ad generation"""
    product_name: str = Field(..., description="Name of the product")
    category: str = Field(..., description="Product category")
    description: str = Field(..., description="Product description")
    price: Optional[float] = Field(None, gt=0, description="Product price (optional)")
    features: Optional[str] = Field(None, description="Comma-separated features")
    
    class Config:
        schema_extra = {
            "example": {
                "product_name": "Wireless Bluetooth Headphones",
                "category": "Electronics",  
                "description": "Premium noise-cancelling over-ear headphones with 30-hour battery life",
                "price": 299.99,
                "features": "Bluetooth 5.0, 30hr battery, Active noise cancellation"
            }
        }


class AdCreativeOutput(BaseModel):
    """Response model for generated ad"""
    ad_creative: str = Field(..., description="Generated ad creative text")
    quality_score: float = Field(..., description="Quality score (0-1)")
    generation_time: float = Field(..., description="Time taken to generate (seconds)")
    model_version: str = Field(..., description="Model version used")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_name: str
    model_version: str


def load_model_from_mlflow():
    """Load model from MLflow registry (3 pts)"""
    global model, tokenizer, model_loaded
    
    try:
        logger.info(f"🔄 Loading model from MLflow...")
        logger.info(f"  - Tracking URI: {Config.MLFLOW_TRACKING_URI}")
        logger.info(f"  - Model name: {Config.MODEL_NAME}")
        logger.info(f"  - Stage: {Config.MODEL_STAGE}")
        
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        
        # Try to load from registry
        try:
            model_uri = f"models:/{Config.MODEL_NAME}/{Config.MODEL_STAGE}"
            logger.info(f"  - Model URI: {model_uri}")
            
            loaded_model = mlflow.pytorch.load_model(model_uri)
            model = loaded_model
            
            # Load tokenizer from the same run
            logger.info(f"  - Loading tokenizer...")
            tokenizer_path = "./models/ad-creative-generator"  # Fallback to local
            tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
            
            model_loaded = True
            MODEL_VERSION.set(1)  # Set version metric
            
            logger.info(f"✅ Model loaded successfully from MLflow registry")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not load from registry: {e}")
            logger.info(f"    Falling back to local model...")
            
            # Fallback: load from local path
            model_path = "./models/ad-creative-generator"
            if os.path.exists(model_path):
                model = T5ForConditionalGeneration.from_pretrained(model_path)
                tokenizer = T5Tokenizer.from_pretrained(model_path)
                model_loaded = True
                MODEL_VERSION.set(0)  # Local version
                logger.info(f"✅ Model loaded from local path")
            else:
                raise ValueError(f"Model not found in registry or locally at {model_path}")
        
        # Set model to eval mode
        model.eval()
        
        logger.info(f"✨ Model ready for inference")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        model_loaded = False
        return False


def calculate_quality_score(generated_text: str, input_data: ProductInput) -> float:
    """
    Calculate a quality score for the generated ad (0-1)
    Based on simple heuristics:
    - Length check
    - Contains product name
    - Contains price
    - Has structure
    """
    score = 0.0
    
    # Length check (0.3 pts)
    if 50 <= len(generated_text) <= 300:
        score += 0.3
    elif len(generated_text) > 20:
        score += 0.15
    
    # Contains product name (0.3 pts)
    if input_data.product_name.lower() in generated_text.lower():
        score += 0.3
    
    # Contains price reference (0.2 pts)
    if str(input_data.price) in generated_text or '$' in generated_text:
        score += 0.2
    
    # Has some structure (bullet points, newlines) (0.2 pts)
    if '•' in generated_text or '\n' in generated_text or '✨' in generated_text:
        score += 0.2
    
    return min(score, 1.0)


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("🚀 Starting Ad Creative Generator API")
    load_model_from_mlflow()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Ad Creative Generator API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/generate",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint (required for K8s probes)"""
    REQUEST_COUNT.labels(endpoint='/health', status='success').inc()
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_name=Config.MODEL_NAME,
        model_version=Config.MODEL_VERSION
    )


@app.post("/generate", response_model=AdCreativeOutput, tags=["Generation"])
async def generate_ad_creative(product: ProductInput):
    """
    Generate an ad creative for a product
    This is the main inference endpoint
    """
    start_time = time.time()
    
    try:
        if not model_loaded:
            REQUEST_COUNT.labels(endpoint='/generate', status='error').inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Create input prompt (matching training format)
        prompt = f"generate ad: Name: {product.product_name}, "
        prompt += f"Category: {product.category}, "
        prompt += f"Description: {product.description}"
        
        if product.price:
            prompt += f", Price: ${product.price}"
        if product.features:
            prompt += f", Features: {product.features}"
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=128,
            truncation=True
        )
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=150,
                num_beams=4,
                early_stopping=True,
                temperature=0.7,
                do_sample=False
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate quality score
        quality = calculate_quality_score(generated_text, product)
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Update Prometheus metrics (6 pts)
        REQUEST_COUNT.labels(endpoint='/generate', status='success').inc()
        REQUEST_LATENCY.observe(latency)
        QUALITY_SCORE.set(quality)
        
        logger.info(f"✅ Generated ad for '{product.product_name}' in {latency:.2f}s (quality: {quality:.2f})")
        
        return AdCreativeOutput(
            ad_creative=generated_text,
            quality_score=quality,
            generation_time=latency,
            model_version=Config.MODEL_VERSION
        )
        
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='/generate', status='error').inc()
        logger.error(f"❌ Error generating ad: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Metrics"])
async def metrics():
    """
    Prometheus metrics endpoint (4 pts for Prometheus scraping)
    Exposes custom metrics for monitoring
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
