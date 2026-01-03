"""
Pytest configuration and shared fixtures.

Provides:
- Sample data fixtures
- Mock models
- Test configurations
- Common utilities
"""
import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path


@pytest.fixture
def sample_raw_data():
    """Create sample raw loan data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'LoanID': range(1, n_samples + 1),
        'Age': np.random.randint(18, 70, n_samples),
        'Income': np.random.randint(20000, 150000, n_samples),
        'LoanAmount': np.random.randint(5000, 50000, n_samples),
        'CreditScore': np.random.randint(300, 850, n_samples),
        'MonthsEmployed': np.random.randint(0, 300, n_samples),
        'NumCreditLines': np.random.randint(0, 20, n_samples),
        'InterestRate': np.random.uniform(0.03, 0.25, n_samples),
        'LoanTerm': np.random.choice([36, 60, 84, 120], n_samples),
        'DTIRatio': np.random.uniform(0.1, 0.6, n_samples),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'EmploymentType': np.random.choice(['Full-time', 'Part-time', 'Self-employed'], n_samples),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'HasMortgage': np.random.choice(['Yes', 'No'], n_samples),
        'HasDependents': np.random.choice(['Yes', 'No'], n_samples),
        'LoanPurpose': np.random.choice(['DebtConsolidation', 'HomeImprovement', 'Medical'], n_samples),
        'HasCosigner': np.random.choice(['Yes', 'No'], n_samples),
        'Default': np.random.choice(['Y', 'N'], n_samples, p=[0.15, 0.85])
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_clean_data(sample_raw_data):
    """Create sample cleaned data."""
    df = sample_raw_data.copy()
    
    # Lowercase columns
    df.columns = df.columns.str.lower()
    
    # Map default to binary
    df['default'] = df['default'].map({'Y': 1, 'N': 0})
    
    return df


@pytest.fixture
def sample_loan_application():
    """Create a valid loan application for testing."""
    return {
        "age": 35,
        "income": 55000,
        "loanamount": 25000,
        "creditscore": 680,
        "monthsemployed": 48,
        "numcreditlines": 5,
        "interestrate": 0.085,
        "loanterm": 60,
        "dtiratio": 0.35,
        "education": "Bachelor",
        "employmenttype": "Full-time",
        "maritalstatus": "Married",
        "hasmortgage": "Yes",
        "hasdependents": "Yes",
        "loanpurpose": "DebtConsolidation",
        "hascosigner": "No"
    }


@pytest.fixture
def test_data_path(tmp_path):
    """Create temporary data directory for tests."""
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def test_model_path(tmp_path):
    """Create temporary model directory for tests."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    return model_dir


# Test configuration
TEST_RANDOM_SEED = 42
TEST_SAMPLE_SIZE = 100
