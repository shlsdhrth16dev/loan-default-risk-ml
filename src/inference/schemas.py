"""
Pydantic schemas for loan default prediction API.

Provides:
- Input validation with type checking
- Value constraints (min/max, allowed values)
- Automatic documentation for FastAPI
- Example data for testing
"""
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
from datetime import datetime


# Base features that come from raw data
class LoanApplicationInput(BaseModel):
    """
    Schema for loan application input data.
    
    All feature names match the cleaned data schema.
    """
    # Applicant Demographics
    age: int = Field(
        ..., 
        ge=18, 
        le=100,
        description="Applicant age in years",
        example=35
    )
    
    income: float = Field(
        ...,
        ge=0,
        le=1000000,
        description="Annual income in USD",
        example=55000
    )
    
    education: Literal[
        "High School", 
        "Bachelor", 
        "Master", 
        "PhD",
        "UNKNOWN"
    ] = Field(
        ...,
        description="Education level",
        example="Bachelor"
    )
    
    employmenttype: Literal[
        "Full-time",
        "Part-time", 
        "Self-employed",
        "Unemployed",
        "UNKNOWN"
    ] = Field(
        ...,
        description="Employment type",
        example="Full-time"
    )
    
    monthsemployed: int = Field(
        ...,
        ge=0,
        le=600,
        description="Months at current employer",
        example=48
    )
    
    maritalstatus: Literal[
        "Single",
        "Married", 
        "Divorced",
        "UNKNOWN"
    ] = Field(
        ...,
        description="Marital status",
        example="Married"
    )
    
    hasmortgage: Literal["Yes", "No"] = Field(
        ...,
        description="Has existing mortgage",
        example="Yes"
    )
    
    hasdependents: Literal["Yes", "No"] = Field(
        ...,
        description="Has dependents",
        example="Yes"
    )
    
    # Loan Details
    loanamount: float = Field(
        ...,
        ge=1000,
        le=100000,
        description="Requested loan amount in USD",
        example=25000
    )
    
    loanterm: int = Field(
        ...,
        ge=12,
        le=360,
        description="Loan term in months",
        example=60
    )
    
    interestrate: float = Field(
        ...,
        ge=0.01,
        le=0.35,
        description="Interest rate (decimal, e.g., 0.05 = 5%)",
        example=0.085
    )
    
    loanpurpose: Literal[
        "DebtConsolidation",
        "HomeImprovement",
        "Medical",
        "Education",
        "Business",
        "Other",
        "UNKNOWN"
    ] = Field(
        ...,
        description="Purpose of the loan",
        example="DebtConsolidation"
    )
    
    # Credit Profile
    creditscore: int = Field(
        ...,
        ge=300,
        le=850,
        description="Credit score (FICO)",
        example=680
    )
    
    numcreditlines: int = Field(
        ...,
        ge=0,
        le=50,
        description="Number of active credit lines",
        example=5
    )
    
    dtiratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Debt-to-income ratio (decimal)",
        example=0.35
    )
    
    hascosigner: Literal["Yes", "No"] = Field(
        ...,
        description="Has co-signer on loan",
        example="No"
    )
    
    @validator('dtiratio')
    def validate_dti(cls, v):
        """Ensure DTI ratio is reasonable."""
        if v > 0.6:
            raise ValueError('DTI ratio > 60% is extremely high and risky')
        return v
    
    @validator('interestrate')
    def validate_interest(cls, v):
        """Ensure interest rate is in decimal format."""
        if v > 1.0:
            raise ValueError('Interest rate should be decimal (e.g., 0.05 not 5.0)')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "income": 55000,
                "education": "Bachelor",
                "employmenttype": "Full-time",
                "monthsemployed": 48,
                "maritalstatus": "Married",
                "hasmortgage": "Yes",
                "hasdependents": "Yes",
                "loanamount": 25000,
                "loanterm": 60,
                "interestrate": 0.085,
                "loanpurpose": "DebtConsolidation",
                "creditscore": 680,
                "numcreditlines": 5,
                "dtiratio": 0.35,
                "hascosigner": "No"
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction API response."""
    
    prediction: Literal["approved", "rejected"] = Field(
        ...,
        description="Loan decision"
    )
    
    default_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of default (0-1)"
    )
    
    confidence: Literal["low", "medium", "high"] = Field(
        ...,
        description="Model confidence level"
    )
    
    risk_category: Literal["low", "medium", "high", "very_high"] = Field(
        ...,
        description="Risk category"
    )
    
    threshold_used: float = Field(
        ...,
        description="Decision threshold applied"
    )
    
    model_version: str = Field(
        ...,
        description="Model version used for prediction"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Prediction timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": "approved",
                "default_probability": 0.23,
                "confidence": "high",
                "risk_category": "low",
                "threshold_used": 0.614,
                "model_version": "xgboost_v1.0",
                "timestamp": "2026-01-03T14:57:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests."""
    
    applications: list[LoanApplicationInput] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of loan applications (max 1000)"
    )
    
    use_threshold: Optional[float] = Field(
        0.614,
        ge=0.0,
        le=1.0,
        description="Custom decision threshold (default: 0.614)"
    )


# Feature names mapping for reference
REQUIRED_FEATURES = [
    "age",
    "income", 
    "education",
    "employmenttype",
    "monthsemployed",
    "maritalstatus",
    "hasmortgage",
    "hasdependents",
    "loanamount",
    "loanterm",
    "interestrate",
    "loanpurpose",
    "creditscore",
    "numcreditlines",
    "dtiratio",
    "hascosigner"
]

# Allowed categorical values
CATEGORICAL_VALUES = {
    "education": ["High School", "Bachelor", "Master", "PhD", "UNKNOWN"],
    "employmenttype": ["Full-time", "Part-time", "Self-employed", "Unemployed", "UNKNOWN"],
    "maritalstatus": ["Single", "Married", "Divorced", "UNKNOWN"],
    "hasmortgage": ["Yes", "No"],
    "hasdependents": ["Yes", "No"],
    "loanpurpose": ["DebtConsolidation", "HomeImprovement", "Medical", "Education", "Business", "Other", "UNKNOWN"],
    "hascosigner": ["Yes", "No"]
}
