import pickle
import pandas as pd
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to path to allow importing src
sys.path.append(os.getcwd())

from src.features.feature_pipeline import build_features
from src.visualize.api_plots import create_risk_chart_base64

app = FastAPI(title="Loan Default Risk API")

# Load the trained model and columns
MODEL_PATH = "reports/trained_model.pkl"
COLUMNS_PATH = "reports/training_columns.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run training first.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load expected columns
if os.path.exists(COLUMNS_PATH):
    with open(COLUMNS_PATH, "rb") as f:
        training_columns = pickle.load(f)
else:
    training_columns = None
    print("WARNING: Training columns file not found. Prediction might fail if features mismatch.")

class LoanApplication(BaseModel):
    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Loan Default Risk API is running"}

@app.post("/predict")
def predict(application: LoanApplication):
    try:
        # Convert Pydantic model to DataFrame
        data = application.dict()
        df = pd.DataFrame([data])
        
        # Apply the same feature engineering as training
        try:
            df_processed = build_features(df)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Feature engineering failed: {str(e)}")

        # Align columns with training data
        if training_columns:
            # Reindex adds missing columns (zeros) and removes extra columns
            df_processed = df_processed.reindex(columns=training_columns, fill_value=0)
        
        # Make prediction
        prob = model.predict_proba(df_processed)[0][1]
        risk = "HIGH" if prob > 0.6 else "LOW"
        
        # Generate visualization
        image_base64 = create_risk_chart_base64(prob)
        
        return {
            "default_probability": float(prob),
            "risk": risk,
            "visualization_base64": image_base64
        }
    except Exception as e:
        # Log the error for debugging
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
