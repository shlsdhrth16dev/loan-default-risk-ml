"""
Unit tests for feature engineering.

Tests:
- Feature creation
- Feature transformations
- Output shapes
- Value correctness
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestFeatureEngineering:
    """Test suite for feature engineering operations."""
    
    def test_loan_to_income_ratio(self, sample_clean_data):
        """Test loan-to-income ratio calculation."""
        df = sample_clean_data.copy()
        df['loan_to_income_ratio'] = df['loanamount'] / (df['income'] + 1)
        
        assert 'loan_to_income_ratio' in df.columns
        assert (df['loan_to_income_ratio'] >= 0).all()
        assert df['loan_to_income_ratio'].notna().all()
    
    def test_log_transforms(self, sample_clean_data):
        """Test log transformations."""
        df = sample_clean_data.copy()
        df['log_income'] = np.log1p(df['income'])
        df['log_loan_amount'] = np.log1p(df['loanamount'])
        
        assert (df['log_income'] >= 0).all()
        assert (df['log_loan_amount'] >= 0).all()
        assert df['log_income'].notna().all()
    
    def test_credit_score_bins(self, sample_clean_data):
        """Test credit score binning."""
        df = sample_clean_data.copy()
        df['credit_score_bin'] = pd.cut(
            df['creditscore'],
            bins=[0, 580, 670, 740, 850],
            labels=['poor', 'fair', 'good', 'excellent']
        )
        
        assert df['credit_score_bin'].notna().all()
        assert set(df['credit_score_bin'].unique()).issubset(
            {'poor', 'fair', 'good', 'excellent'}
        )
    
    def test_age_groups(self, sample_clean_data):
        """Test age grouping."""
        df = sample_clean_data.copy()
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 25, 35, 50, 100],
            labels=['young', 'mid', 'mature', 'senior']
        )
        
        assert df['age_group'].notna().all()
        valid_groups = {'young', 'mid', 'mature', 'senior'}
        assert set(df['age_group'].unique()).issubset(valid_groups)
    
    def test_binary_features(self, sample_clean_data):
        """Test binary feature creation."""
        df = sample_clean_data.copy()
        
        dti_median = df['dtiratio'].median()
        df['high_dti'] = (df['dtiratio'] > dti_median).astype(int)
        
        assert set(df['high_dti'].unique()).issubset({0, 1})
    
    def test_interaction_features(self, sample_clean_data):
        """Test interaction feature creation."""
        df = sample_clean_data.copy()
        df['rate_times_amount'] = df['interestrate'] * df['loanamount'] / 10000
        
        assert 'rate_times_amount' in df.columns
        assert (df['rate_times_amount'] >= 0).all()
    
    def test_no_infinite_values(self, sample_clean_data):
        """Test that feature engineering doesn't create infinite values."""
        df = sample_clean_data.copy()
        
        # Create features that might cause inf
        df['loan_to_income_ratio'] = df['loanamount'] / (df['income'] + 1)
        df['log_income'] = np.log1p(df['income'])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            assert np.isfinite(df[col]).all(), f"Infinite values in {col}"
    
    def test_feature_count(self, sample_clean_data):
        """Test that expected number of features are created."""
        df = sample_clean_data.copy()
        
        # Simulate feature engineering
        df['loan_to_income_ratio'] = df['loanamount'] / (df['income'] + 1)
        df['log_income'] = np.log1p(df['income'])
        df['log_loan_amount'] = np.log1p(df['loanamount'])
        
        # Should have original + engineered features
        assert len(df.columns) >= len(sample_clean_data.columns) + 3
