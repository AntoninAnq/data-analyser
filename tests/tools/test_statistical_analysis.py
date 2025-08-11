#!/usr/bin/env python3
"""
Unit tests for statistical analysis tools
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from tools.statistical_analysis import (
    calculate_column_mean,
    calculate_column_std,
    analyze_column_skewness
)


@pytest.fixture
def sample_numeric_data():
    """Create sample numeric data for testing"""
    data = {
        'normal_dist': np.random.normal(100, 15, 1000),
        'skewed_right': np.random.exponential(50, 1000),
        'skewed_left': 200 - np.random.exponential(50, 1000),
        'uniform': np.random.uniform(0, 100, 1000),
        'categorical': ['A', 'B', 'C'] * 333 + ['A']
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(sample_numeric_data):
    """Create a temporary CSV file with sample data"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_numeric_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


class TestCalculateColumnMean:
    """Test cases for calculate_column_mean function"""
    
    def test_calculate_mean_normal_distribution(self, temp_csv_file):
        """Test mean calculation on normal distribution"""
        result = calculate_column_mean(temp_csv_file, 'normal_dist')
        
        assert "Mean Analysis: normal_dist" in result
        assert "Mean Value:" in result
        assert "Dataset Information" in result
        assert "Additional Statistics" in result
        assert "Insights" in result
        
        # Check that mean is calculated (should be close to 100 for normal dist)
        lines = result.split('\n')
        mean_line = [line for line in lines if "Mean Value:" in line][0]
        # Handle markdown formatting: "** 99.1788" -> "99.1788"
        mean_value_str = mean_line.split(':')[1].strip().replace('**', '').strip()
        mean_value = float(mean_value_str)
        assert 95 <= mean_value <= 105  # Allow some variance
        
    def test_calculate_mean_skewed_distribution(self, temp_csv_file):
        """Test mean calculation on skewed distribution"""
        result = calculate_column_mean(temp_csv_file, 'skewed_right')
        
        assert "Mean Analysis: skewed_right" in result
        assert "Mean Value:" in result
        
        # Check that mean is calculated (should be close to 50 for exponential)
        lines = result.split('\n')
        mean_line = [line for line in lines if "Mean Value:" in line][0]
        mean_value_str = mean_line.split(':')[1].strip().replace('**', '').strip()
        mean_value = float(mean_value_str)
        assert 45 <= mean_value <= 55  # Allow some variance
        
    def test_calculate_mean_invalid_column(self, temp_csv_file):
        """Test mean calculation with non-existent column"""
        result = calculate_column_mean(temp_csv_file, 'non_existent_column')
        
        assert "❌ **Error**" in result
        assert "not found in the dataset" in result
        
    def test_calculate_mean_categorical_column(self, temp_csv_file):
        """Test mean calculation with categorical column"""
        result = calculate_column_mean(temp_csv_file, 'categorical')
        
        assert "❌ **Error**" in result
        assert "not numeric" in result
        
    def test_calculate_mean_file_not_found(self):
        """Test mean calculation with non-existent file"""
        result = calculate_column_mean('non_existent_file.csv', 'column')
        
        assert "❌ **Error**" in result
        assert "not found" in result


class TestCalculateColumnStd:
    """Test cases for calculate_column_std function"""
    
    def test_calculate_std_normal_distribution(self, temp_csv_file):
        """Test standard deviation calculation on normal distribution"""
        result = calculate_column_std(temp_csv_file, 'normal_dist')
        
        assert "Standard Deviation Analysis: normal_dist" in result
        assert "Standard Deviation:" in result
        assert "Coefficient of Variation:" in result
        assert "Additional Statistics" in result
        assert "Insights" in result
        
        # Check that std is calculated (should be close to 15 for normal dist)
        lines = result.split('\n')
        std_line = [line for line in lines if "Standard Deviation:" in line][0]
        std_value_str = std_line.split(':')[1].strip().replace('**', '').strip()
        std_value = float(std_value_str)
        assert 13 <= std_value <= 17  # Allow some variance
        
    def test_calculate_std_skewed_distribution(self, temp_csv_file):
        """Test standard deviation calculation on skewed distribution"""
        result = calculate_column_std(temp_csv_file, 'skewed_right')
        
        assert "Standard Deviation Analysis: skewed_right" in result
        assert "Standard Deviation:" in result
        assert "Coefficient of Variation:" in result
        
        # Check coefficient of variation (should be high for exponential)
        lines = result.split('\n')
        cv_line = [line for line in lines if "Coefficient of Variation:" in line][0]
        cv_value_str = cv_line.split(':')[1].strip().replace('**', '').strip().replace('%', '')
        cv_value = float(cv_value_str)
        assert cv_value > 80  # CV should be high for exponential distribution
        
    def test_calculate_std_uniform_distribution(self, temp_csv_file):
        """Test standard deviation calculation on uniform distribution"""
        result = calculate_column_std(temp_csv_file, 'uniform')
        
        assert "Standard Deviation Analysis: uniform" in result
        assert "Standard Deviation:" in result
        
        # Check coefficient of variation (should be moderate for uniform)
        lines = result.split('\n')
        cv_line = [line for line in lines if "Coefficient of Variation:" in line][0]
        cv_value_str = cv_line.split(':')[1].strip().replace('**', '').strip().replace('%', '')
        cv_value = float(cv_value_str)
        assert 50 <= cv_value <= 70  # CV should be moderate for uniform distribution
        
    def test_calculate_std_invalid_column(self, temp_csv_file):
        """Test standard deviation calculation with non-existent column"""
        result = calculate_column_std(temp_csv_file, 'non_existent_column')
        
        assert "❌ **Error**" in result
        assert "not found in the dataset" in result
        
    def test_calculate_std_categorical_column(self, temp_csv_file):
        """Test standard deviation calculation with categorical column"""
        result = calculate_column_std(temp_csv_file, 'categorical')
        
        assert "❌ **Error**" in result
        assert "not numeric" in result


class TestAnalyzeColumnSkewness:
    """Test cases for analyze_column_skewness function"""
    
    def test_analyze_skewness_normal_distribution(self, temp_csv_file):
        """Test skewness analysis on normal distribution"""
        result = analyze_column_skewness(temp_csv_file, 'normal_dist')
        
        assert "Distribution Skewness Analysis: normal_dist" in result
        assert "Skewness:" in result
        assert "Kurtosis:" in result
        assert "Distribution Classification" in result
        assert "Central Tendency Comparison" in result
        
        # Check that skewness is calculated (should be close to 0 for normal)
        lines = result.split('\n')
        skewness_line = [line for line in lines if "Skewness:" in line][0]
        skewness_value_str = skewness_line.split(':')[1].strip().replace('**', '').strip()
        skewness_value = float(skewness_value_str)
        assert abs(skewness_value) < 0.5  # Should be approximately symmetric
        
    def test_analyze_skewness_right_skewed_distribution(self, temp_csv_file):
        """Test skewness analysis on right-skewed distribution"""
        result = analyze_column_skewness(temp_csv_file, 'skewed_right')
        
        assert "Distribution Skewness Analysis: skewed_right" in result
        assert "Skewness:" in result
        assert "Kurtosis:" in result
        
        # Check that skewness is positive (right-skewed)
        lines = result.split('\n')
        skewness_line = [line for line in lines if "Skewness:" in line][0]
        skewness_value_str = skewness_line.split(':')[1].strip().replace('**', '').strip()
        skewness_value = float(skewness_value_str)
        assert skewness_value > 0.5  # Should be right-skewed
        
        # Check classification
        assert "Right-skewed" in result
        
    def test_analyze_skewness_left_skewed_distribution(self, temp_csv_file):
        """Test skewness analysis on left-skewed distribution"""
        result = analyze_column_skewness(temp_csv_file, 'skewed_left')
        
        assert "Distribution Skewness Analysis: skewed_left" in result
        assert "Skewness:" in result
        assert "Kurtosis:" in result
        
        # Check that skewness is negative (left-skewed)
        lines = result.split('\n')
        skewness_line = [line for line in lines if "Skewness:" in line][0]
        skewness_value_str = skewness_line.split(':')[1].strip().replace('**', '').strip()
        skewness_value = float(skewness_value_str)
        assert skewness_value < -0.5  # Should be left-skewed
        
        # Check classification
        assert "Left-skewed" in result
        
    def test_analyze_skewness_uniform_distribution(self, temp_csv_file):
        """Test skewness analysis on uniform distribution"""
        result = analyze_column_skewness(temp_csv_file, 'uniform')
        
        assert "Distribution Skewness Analysis: uniform" in result
        assert "Skewness:" in result
        assert "Kurtosis:" in result
        
        # Check that skewness is close to 0 (uniform)
        lines = result.split('\n')
        skewness_line = [line for line in lines if "Skewness:" in line][0]
        skewness_value_str = skewness_line.split(':')[1].strip().replace('**', '').strip()
        skewness_value = float(skewness_value_str)
        assert abs(skewness_value) < 0.5  # Should be approximately symmetric
        
    def test_analyze_skewness_invalid_column(self, temp_csv_file):
        """Test skewness analysis with non-existent column"""
        result = analyze_column_skewness(temp_csv_file, 'non_existent_column')
        
        assert "❌ **Error**" in result
        assert "not found in the dataset" in result
        
    def test_analyze_skewness_categorical_column(self, temp_csv_file):
        """Test skewness analysis with categorical column"""
        result = analyze_column_skewness(temp_csv_file, 'categorical')
        
        assert "❌ **Error**" in result
        assert "not numeric" in result


class TestStatisticalAnalysisEdgeCases:
    """Test edge cases for statistical analysis tools"""
    
    def test_empty_dataframe(self):
        """Test with empty dataframe"""
        # Create empty dataframe
        df = pd.DataFrame({'numeric_col': []})
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            result = calculate_column_mean(temp_path, 'numeric_col')
            assert "❌ **Error**" in result or "nan" in result.lower()
            
            result = calculate_column_std(temp_path, 'numeric_col')
            assert "❌ **Error**" in result or "nan" in result.lower()
            
            result = analyze_column_skewness(temp_path, 'numeric_col')
            assert "❌ **Error**" in result or "nan" in result.lower()
        finally:
            os.unlink(temp_path)
    
    def test_single_value_dataframe(self):
        """Test with dataframe containing only one value"""
        df = pd.DataFrame({'numeric_col': [42.0]})
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            result = calculate_column_mean(temp_path, 'numeric_col')
            assert "Mean Value:" in result
            # Check that the value 42.0000 appears in the result
            assert "42.0000" in result
            
            result = calculate_column_std(temp_path, 'numeric_col')
            assert "Standard Deviation:" in result
            assert "0.0000" in result
            
            result = analyze_column_skewness(temp_path, 'numeric_col')
            assert "Skewness:" in result
        finally:
            os.unlink(temp_path)
    
    def test_all_null_values(self):
        """Test with dataframe containing only null values"""
        df = pd.DataFrame({'numeric_col': [np.nan, np.nan, np.nan]})
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            result = calculate_column_mean(temp_path, 'numeric_col')
            assert "nan" in result.lower() or "❌ **Error**" in result
            
            result = calculate_column_std(temp_path, 'numeric_col')
            assert "nan" in result.lower() or "❌ **Error**" in result
            
            result = analyze_column_skewness(temp_path, 'numeric_col')
            assert "nan" in result.lower() or "❌ **Error**" in result
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])
