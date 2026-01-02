import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from fastapi.testclient import TestClient
import pytest

# Try to import app, handling potential path issues
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Loan Default Risk API is running"}

def test_predict_endpoint():
    payload = {
        "person_age": 22,
        "person_income": 59000,
        "person_home_ownership": "RENT",
        "person_emp_length": 123.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "D",
        "loan_amnt": 35000,
        "loan_int_rate": 16.02,
        "loan_percent_income": 0.59,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 3
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert "risk" in data
    assert data["risk"] in ["LOW", "HIGH"]
    assert "visualization_base64" in data
    assert isinstance(data["visualization_base64"], str)
    assert len(data["visualization_base64"]) > 0
