#!/usr/bin/env python3
"""
Fast unit tests for tools - no LLM calls required
These tests focus on tool functionality without calling agents
"""

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch
from tools.dataset_summary import dataset_summary
from tools.column_analysis import analyze_column_unique_values


@pytest.mark.unit
def test_dataset_summary_tool_returns_string(temp_test_file):
    """Test that dataset summary tool returns a string"""
    # Act
    result = dataset_summary(temp_test_file)
    
    # Assert
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.unit
def test_dataset_summary_tool_with_real_data(test_dataset_path):
    """Test dataset summary tool with real dataset"""
    # Act
    result = dataset_summary(test_dataset_path)
    
    # Assert
    assert isinstance(result, str)
    assert "rows" in result.lower() or "columns" in result.lower()


@pytest.mark.unit
def test_dataset_summary_tool_file_not_found():
    """Test dataset summary tool with non-existent file"""
    # Act
    result = dataset_summary("non_existent_file.csv")
    
    # Assert
    assert isinstance(result, str)
    assert "❌ **Error**" in result
    assert "not found" in result.lower()


@pytest.mark.unit
def test_dataset_summary_tool_empty_file():
    """Test dataset summary tool with empty file"""
    # Arrange
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("")  # Empty file
        temp_path = f.name
    
    try:
        # Act
        result = dataset_summary(temp_path)
        
        # Assert
        assert isinstance(result, str)
    finally:
        # Cleanup
        os.unlink(temp_path)


@pytest.mark.unit
def test_column_unique_values_tool_returns_string(temp_test_file):
    """Test that column analysis tool returns a string"""
    # Act
    result = analyze_column_unique_values(temp_test_file, "column1")
    
    # Assert
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.unit
def test_column_unique_values_tool_with_real_data(test_dataset_path):
    """Test column analysis tool with real dataset"""
    # Act
    result = analyze_column_unique_values(test_dataset_path, "SEX")
    
    # Assert
    assert isinstance(result, str)
    assert "unique" in result.lower()


@pytest.mark.unit
def test_column_unique_values_tool_invalid_column(temp_test_file):
    """Test column analysis tool with non-existent column"""
    # Act
    result = analyze_column_unique_values(temp_test_file, "non_existent_column")
    
    # Assert
    assert isinstance(result, str)
    assert "❌ **Error**" in result
    assert "not found" in result.lower()
    assert "column1" in result  # Should mention available columns


@pytest.mark.unit
def test_column_unique_values_tool_file_not_found():
    """Test column analysis tool with non-existent file"""
    # Act
    result = analyze_column_unique_values("non_existent_file.csv", "column1")
    
    # Assert
    assert isinstance(result, str)
    assert "❌ **Error**" in result
    assert "not found" in result.lower()


@pytest.mark.integration
@pytest.mark.parametrize("csv_content,column_name", [
    ("col1,col2\n1,2\n3,4\n", "col1"),
    ("col1;col2\n1;2\n3;4\n", "col1"),  # Semicolon separated
    ("col1\tcol2\n1\t2\n3\t4\n", "col1"),  # Tab separated
])
def test_tools_with_different_file_formats(csv_content, column_name):
    """Test tools with different CSV formats"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        temp_path = f.name
    
    try:
        # Test dataset summary
        summary_result = dataset_summary(temp_path)
        assert isinstance(summary_result, str)
        
        # Test column analysis
        column_result = analyze_column_unique_values(temp_path, column_name)
        assert isinstance(column_result, str)
        
    finally:
        os.unlink(temp_path)


@pytest.mark.integration
def test_tools_with_large_numbers():
    """Test tools with large numbers in data"""
    # Arrange
    csv_content = "id,value\n1,1000000\n2,2000000\n3,3000000\n"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        temp_path = f.name
    
    try:
        # Act
        summary_result = dataset_summary(temp_path)
        column_result = analyze_column_unique_values(temp_path, "value")
        
        # Assert
        assert isinstance(summary_result, str)
        assert isinstance(column_result, str)
        assert "1000000" in column_result or "2000000" in column_result
        
    finally:
        os.unlink(temp_path)


@pytest.mark.integration
def test_tools_with_special_characters():
    """Test tools with special characters in data"""
    # Arrange
    csv_content = 'name,description\n"John Doe","Contains, comma"\n"Jane Smith","Contains ""quotes"""\n'
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        temp_path = f.name
    
    try:
        # Act
        summary_result = dataset_summary(temp_path)
        column_result = analyze_column_unique_values(temp_path, "name")
        
        # Assert
        assert isinstance(summary_result, str)
        assert isinstance(column_result, str)
        
    finally:
        os.unlink(temp_path)


@pytest.mark.performance
def test_tool_performance_with_large_dataset():
    """Test tool performance with larger dataset"""
    # Arrange - Create a larger test dataset
    large_csv_content = "id,name,value\n"
    for i in range(1000):  # 1000 rows
        large_csv_content += f"{i},name_{i},{i * 10}\n"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(large_csv_content)
        temp_path = f.name
    
    try:
        # Act & Assert - Should complete within reasonable time
        import time
        start_time = time.time()
        
        summary_result = dataset_summary(temp_path)
        column_result = analyze_column_unique_values(temp_path, "name")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert
        assert isinstance(summary_result, str)
        assert isinstance(column_result, str)
        assert execution_time < 5.0  # Should complete within 5 seconds
        
    finally:
        os.unlink(temp_path)


# Performance benchmarks
@pytest.mark.benchmark
def test_dataset_summary_benchmark(benchmark, test_dataset_path):
    """Benchmark dataset summary tool"""
    result = benchmark(dataset_summary, test_dataset_path)
    assert isinstance(result, str)


@pytest.mark.benchmark
def test_column_analysis_benchmark(benchmark, test_dataset_path):
    """Benchmark column analysis tool"""
    result = benchmark(analyze_column_unique_values, test_dataset_path, "SEX")
    assert isinstance(result, str)
