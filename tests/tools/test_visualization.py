#!/usr/bin/env python3
"""
Unit tests for visualization tools
"""

import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import Mock, patch
from tools.visualization import (
    distribution_plot,
    correlation_heatmap,
    pair_plot,
    scatter_plot,
    bar_plot
)


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing"""
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [30000, 45000, 55000, 65000, 75000],
        'category': ['A', 'B', 'A', 'C', 'B']
    })


@pytest.fixture
def temp_csv_file(sample_dataframe):
    """Create a temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_dataframe.to_csv(f.name, index=False)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


class TestVisualizationTools:
    """Test visualization tools with proper mocking"""
    
    @patch('tools.visualization.load_dataset')
    @patch('tools.visualization.validate_column_exists')
    @patch('tools.visualization.plt')
    @patch('tools.visualization.save_plot')
    def test_distribution_plot_success(self, mock_save_plot, mock_plt, 
                                     mock_validate, mock_load_dataset, 
                                     sample_dataframe, temp_csv_file):
        """Test successful distribution plot creation"""
        # Arrange
        mock_load_dataset.return_value = sample_dataframe
        mock_validate.return_value = True
        mock_save_plot.return_value = "plots/distribution_age_20240101_120000.png"
        
        # Mock subplots to return a figure and axes
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_plt.gcf.return_value = mock_fig
        
        # Act
        result = distribution_plot(temp_csv_file, "age")
        
        # Assert
        assert "✅ **Distribution plot created successfully!**" in result
        assert "plots/distribution_age_20240101_120000.png" in result
    
    @patch('tools.visualization.load_dataset')
    def test_distribution_plot_file_not_found(self, mock_load_dataset):
        """Test distribution plot with file not found"""
        # Arrange
        mock_load_dataset.return_value = "❌ **Error**: File not found"
        
        # Act
        result = distribution_plot("nonexistent.csv", "age")
        
        # Assert
        assert "❌ **Error**" in result
    
    @patch('tools.visualization.load_dataset')
    @patch('tools.visualization.plt')
    @patch('tools.visualization.sns')
    @patch('tools.visualization.save_plot')
    def test_correlation_heatmap_success(self, mock_save_plot, mock_sns, 
                                       mock_plt, mock_load_dataset, 
                                       sample_dataframe, temp_csv_file):
        """Test successful correlation heatmap creation"""
        # Arrange
        mock_load_dataset.return_value = sample_dataframe
        mock_save_plot.return_value = "plots/correlation_heatmap_20240101_120000.png"
        
        # Act
        result = correlation_heatmap(temp_csv_file)
        
        # Assert
        assert "✅ **Correlation heatmap created successfully!**" in result
        assert "plots/correlation_heatmap_20240101_120000.png" in result
    
    @patch('tools.visualization.load_dataset')
    def test_correlation_heatmap_no_numerical_columns(self, mock_load_dataset):
        """Test correlation heatmap with no numerical columns"""
        # Arrange
        df_no_numeric = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'text': ['hello', 'world', 'test']
        })
        mock_load_dataset.return_value = df_no_numeric
        
        # Act
        result = correlation_heatmap("test.csv")
        
        # Assert
        assert "❌ **Error**: No numerical columns found in the dataset." in result
    
    @patch('tools.visualization.load_dataset')
    @patch('tools.visualization.plt')
    @patch('tools.visualization.sns')
    @patch('tools.visualization.save_plot')
    def test_pair_plot_success(self, mock_save_plot, mock_sns, 
                             mock_plt, mock_load_dataset, 
                             sample_dataframe, temp_csv_file):
        """Test successful pair plot creation"""
        # Arrange
        mock_load_dataset.return_value = sample_dataframe
        mock_save_plot.return_value = "plots/pair_plot_20240101_120000.png"
        
        # Mock pairplot
        mock_pairplot = Mock()
        mock_pairplot.fig = Mock()
        mock_sns.pairplot.return_value = mock_pairplot
        
        # Act
        result = pair_plot(temp_csv_file)
        
        # Assert
        assert "✅ **Pair plot created successfully!**" in result
        assert "plots/pair_plot_20240101_120000.png" in result
    
    @patch('tools.visualization.load_dataset')
    @patch('tools.visualization.validate_column_exists')
    @patch('tools.visualization.plt')
    @patch('tools.visualization.save_plot')
    def test_scatter_plot_success(self, mock_save_plot, mock_plt, 
                                mock_validate, mock_load_dataset, 
                                sample_dataframe, temp_csv_file):
        """Test successful scatter plot creation"""
        # Arrange
        mock_load_dataset.return_value = sample_dataframe
        mock_validate.return_value = True
        mock_save_plot.return_value = "plots/scatter_age_vs_income_20240101_120000.png"
        
        # Act
        result = scatter_plot(temp_csv_file, "age", "income")
        
        # Assert
        assert "✅ **Scatter plot created successfully!**" in result
        assert "plots/scatter_age_vs_income_20240101_120000.png" in result
    
    @patch('tools.visualization.load_dataset')
    @patch('tools.visualization.validate_column_exists')
    @patch('tools.visualization.plt')
    @patch('tools.visualization.save_plot')
    def test_bar_plot_success(self, mock_save_plot, mock_plt, 
                            mock_validate, mock_load_dataset, 
                            sample_dataframe, temp_csv_file):
        """Test successful bar plot creation"""
        # Arrange
        mock_load_dataset.return_value = sample_dataframe
        mock_validate.return_value = True
        mock_save_plot.return_value = "plots/bar_plot_category_20240101_120000.png"
        
        # Act
        result = bar_plot(temp_csv_file, "category")
        
        # Assert
        assert "✅ **Bar plot created successfully!**" in result
        assert "plots/bar_plot_category_20240101_120000.png" in result
