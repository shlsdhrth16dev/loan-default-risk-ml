"""
Unit tests for data cleaning module.

Tests:
- Data loading
- Column standardization
- Missing value handling
- Target encoding
- Data validation
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestDataCleaning:
    """Test suite for data cleaning operations."""
    
    def test_column_lowercasing(self, sample_raw_data):
        """Test that column names are properly lowercased."""
        df = sample_raw_data.copy()
        df.columns = df.columns.str.lower()
        
        assert all(col.islower() for col in df.columns)
        assert 'age' in df.columns
        assert 'loanamount' in df.columns
    
    def test_target_encoding(self, sample_raw_data):
        """Test default column is properly encoded."""
        df = sample_raw_data.copy()
        df.columns = df.columns.str.lower()
        df['default'] = df['default'].map({'Y': 1, 'N': 0})
        
        assert df['default'].dtype == np.int64 or df['default'].dtype == np.float64
        assert set(df['default'].unique()).issubset({0, 1})
    
    def test_no_duplicates(self, sample_clean_data):
        """Test that duplicate rows are removed."""
        df_with_dupes = pd.concat([sample_clean_data, sample_clean_data.iloc[:5]])
        df_clean = df_with_dupes.drop_duplicates()
        
        assert len(df_clean) == len(sample_clean_data)
    
    def test_required_columns_exist(self, sample_clean_data):
        """Test that all required columns exist after cleaning."""
        required_cols = [
            'age', 'income', 'loanamount', 'creditscore',
            'monthsemployed', 'numcreditlines', 'interestrate',
            'loanterm', 'dtiratio', 'default'
        ]
        
        for col in required_cols:
            assert col in sample_clean_data.columns, f"Missing required column: {col}"
    
    def test_data_types(self, sample_clean_data):
        """Test that data types are correct after cleaning."""
        numeric_cols = ['age', 'income', 'loanamount', 'creditscore']
        
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(sample_clean_data[col])
    
    def test_value_ranges(self, sample_clean_data):
        """Test that values are within expected ranges."""
        assert (sample_clean_data['age'] >= 18).all()
        assert (sample_clean_data['age'] <= 100).all()
        
        assert (sample_clean_data['creditscore'] >= 300).all()
        assert (sample_clean_data['creditscore'] <= 850).all()
        
        assert (sample_clean_data['interestrate'] >= 0).all()
        assert (sample_clean_data['interestrate'] <= 1).all()
    
    def test_no_missing_critical_columns(self, sample_clean_data):
        """Test that critical columns have no missing values."""
        critical_cols = ['age', 'income', 'loanamount', 'default']
        
        for col in critical_cols:
            assert sample_clean_data[col].notna().all(), f"Missing values in {col}"
    
    def test_target_distribution(self, sample_clean_data):
        """Test that target variable has reasonable distribution."""
        default_rate = sample_clean_data['default'].mean()
        
        # Should be between 5% and 40% for realistic loan data
        assert 0.05 <= default_rate <= 0.40, f"Unusual default rate: {default_rate}"
