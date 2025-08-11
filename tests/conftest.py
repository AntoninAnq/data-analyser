"""
Pytest configuration and shared fixtures for testing agents
"""

import pytest
from unittest.mock import Mock, patch
import os
import tempfile


@pytest.fixture(scope="session")
def test_dataset_path():
    """Fixture providing path to test dataset"""
    return "dataset/DD_EEC_ANNUEL_2024_data.csv"


@pytest.fixture(scope="session")
def mock_llm_response():
    """Fixture providing common mocked LLM responses"""
    return {
        "dataset_overview": "This dataset contains 1000 rows and 5 columns with demographic information.",
        "data_types": "The columns have the following data types: object, int64, float64.",
        "missing_values": "Found 3 missing values in the AGE column.",
        "unique_values": "The SEX column has 2 unique values: M (60%) and F (40%).",
        "error_response": "I encountered an error while analyzing the dataset."
    }


@pytest.fixture
def mock_llm():
    """Fixture for mocked LLM with proper attributes"""
    with patch('main.LLM') as mock_llm_class:
        mock_llm_instance = Mock()
        # Add required methods that CrewAI expects
        mock_llm_instance.supports_stop_words.return_value = True
        mock_llm_instance.llm_type = "mock"
        mock_llm_class.return_value = mock_llm_instance
        yield mock_llm_instance


@pytest.fixture
def mock_crew():
    """Fixture for mocked Crew"""
    with patch('main.Crew') as mock_crew_class:
        mock_crew_instance = Mock()
        mock_crew_class.return_value = mock_crew_instance
        yield mock_crew_instance


@pytest.fixture
def mock_crew_setup():
    """Fixture for mocking crew setup with proper LLM mocking"""
    with patch('main.LLM') as mock_llm_class, \
         patch('main.Crew') as mock_crew_class:
        
        # Create a proper mock LLM instance with required attributes
        mock_llm_instance = Mock()
        mock_llm_instance.supports_stop_words.return_value = True
        mock_llm_instance.llm_type = "mock"
        mock_llm_class.return_value = mock_llm_instance
        
        mock_crew_instance = Mock()
        mock_crew_class.return_value = mock_crew_instance
        
        yield {
            'llm': mock_llm_instance,
            'crew': mock_crew_instance,
            'mock_llm_class': mock_llm_class,
            'mock_crew_class': mock_crew_class
        }


@pytest.fixture
def mock_dataset_summary_tool():
    """Fixture for mocked dataset summary tool"""
    with patch('tools.dataset_summary.dataset_summary') as mock:
        yield mock


@pytest.fixture
def mock_column_analysis_tool():
    """Fixture for mocked column analysis tool"""
    with patch('tools.column_analysis.analyze_column_unique_values') as mock:
        yield mock


@pytest.fixture
def temp_test_file():
    """Fixture providing a temporary test file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("column1,column2,column3\n1,2,3\n4,5,6\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


# Mock benchmark fixture for tests that use pytest-benchmark
@pytest.fixture
def benchmark():
    """Mock benchmark fixture for tests that don't have pytest-benchmark installed"""
    class MockBenchmark:
        def __call__(self, func, *args, **kwargs):
            return func(*args, **kwargs)
    return MockBenchmark()


# Test markers for different test categories
def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (should be skipped in CI)"
    )
    config.addinivalue_line(
        "markers", "llm: mark test as requiring LLM (should be mocked)"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark test"
    )


# Skip slow tests in CI environment
def pytest_collection_modifyitems(config, items):
    """Skip slow tests in CI environment"""
    if os.getenv('CI'):
        skip_slow = pytest.mark.skip(reason="Skipping slow tests in CI")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
