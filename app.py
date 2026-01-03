"""
FastAPI application for loan default prediction service.

Production-ready API with:
- Health check endpoint
- Single prediction endpoint
- Batch prediction endpoint
- Model information endpoint
- Feature importance endpoint
- Comprehensive error handling
- Auto-generated OpenAPI documentation
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from inference.schemas import (
    LoanApplicationInput,
    PredictionResponse,
    BatchPredictionRequest
)
from inference.predict import get_prediction_service

# Initialize FastAPI app
app = FastAPI(
    title="Loan Default Prediction API",
    description="ML-powered API for predicting loan default risk",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service on startup
@app.on_event("startup")
async def startup_event():
    """Initialize prediction service on startup."""
    try:
        service = get_prediction_service()
        print("✓ Prediction service initialized")
    except Exception as e:
        print(f"✗ Failed to initialize prediction service: {e}")
        raise


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Loan Default Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info",
            "feature_importance": "/model/features"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns service health status and performs diagnostic checks.
    """
    try:
        service = get_prediction_service()
        health_status = service.health_check()
        
        status_code = (
            status.HTTP_200_OK 
            if health_status["status"] == "healthy" 
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        
        return JSONResponse(
            status_code=status_code,
            content=health_status
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    status_code=status.HTTP_200_OK
)
async def predict_single(application: LoanApplicationInput):
    """
    Predict default risk for a single loan application.
    
    - **Input**: LoanApplicationInput with all required fields
    - **Output**: PredictionResponse with prediction, probability, and metadata
    
    The model uses the F1-optimized threshold (0.614) by default.
    """
    try:
        service = get_prediction_service()
        
        # Convert Pydantic model to dict
        input_dict = application.dict()
        
        # Make prediction
        result = service.predict(input_dict)
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    tags=["Prediction"],
    status_code=status.HTTP_200_OK
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict default risk for multiple loan applications.
    
    - **Input**: BatchPredictionRequest with list of applications (max 1000)
    - **Output**: List of predictions
    
    Supports custom threshold for all predictions in the batch.
    """
    try:
        service = get_prediction_service()
        
        # Convert applications to dicts
        applications = [app.dict() for app in request.applications]
        
        # Make batch predictions
        results = service.predict_batch(
            applications,
            custom_threshold=request.use_threshold
        )
        
        return {
            "predictions": results,
            "total": len(results),
            "threshold_used": request.use_threshold
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """
    Get comprehensive model information and metadata.
    
    Returns training details, hyperparameters, versions, and more.
    """
    try:
        service = get_prediction_service()
        info = service.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.get("/model/features", tags=["Model"])
async def get_feature_importance(top_n: int = 20):
    """
    Get feature importance from the trained model.
    
    - **top_n**: Number of top features to return (default: 20)
    
    Returns feature names and their importance scores.
    """
    try:
        service = get_prediction_service()
        importance = service.get_feature_importance(top_n=top_n)
        
        return {
            "feature_importance": importance,
            "top_n": len(importance)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feature importance: {str(e)}"
        )


@app.get("/model/thresholds", tags=["Model"])
async def get_threshold_info():
    """
    Get information about available decision thresholds.
    
    Returns threshold recommendations for different business objectives.
    """
    return {
        "default_threshold": 0.614,
        "threshold_options": {
            "f1_optimized": {
                "value": 0.614,
                "description": "Balanced precision and recall (recommended)",
                "use_case": "Standard loan approval"
            },
            "precision_focused": {
                "value": 0.627,
                "description": "Minimize false approvals",
                "use_case": "Conservative lending, minimize defaults"
            },
            "recall_focused": {
                "value": 0.625,
                "description": "Catch more potential defaults",
                "use_case": "Risk-averse lending"
            },
            "balanced_f2": {
                "value": 0.432,
                "description": "Favor recall over precision",
                "use_case": "Regulatory compliance, catch all risks"
            }
        },
        "note": "Threshold can be customized per request"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("="*70)
    print("LOAN DEFAULT PREDICTION API")
    print("="*70)
    print("\nStarting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("API Redoc: http://localhost:8000/redoc")
    print("Health Check: http://localhost:8000/health")
    print("="*70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
