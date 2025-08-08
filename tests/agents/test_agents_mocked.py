#!/usr/bin/env python3
"""
Test script for agents using proper mocking techniques
This avoids calling real LLMs and makes tests fast and reliable
"""

import pytest
from unittest.mock import Mock, patch
from agents.data_agent import create_data_agent
from chat_main import analyze_dataset_chat, setup_crew


@pytest.mark.unit
@patch('agents.data_agent.Agent')
def test_create_data_agent(mock_agent_class, mock_llm):
    """Test that data agent is created with correct configuration"""
    # Arrange
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent
    
    # Act
    agent = create_data_agent(llm=mock_llm_instance)
    
    # Assert
    mock_agent_class.assert_called_once()
    call_args = mock_agent_class.call_args
    assert call_args[1]['role'] == 'Data Analysis Assistant'
    assert 'data analyst' in call_args[1]['backstory'].lower()
    assert len(call_args[1]['tools']) == 2  # dataset_summary_tool and column_unique_values_tool
    assert call_args[1]['verbose'] is True
    assert call_args[1]['allow_delegation'] is False
    assert call_args[1]['memory'] is True
    
@pytest.mark.integration
def test_analyze_dataset_chat_success(mock_crew, mock_llm):
    """Test successful dataset analysis with mocked LLM responses"""
    # Arrange
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    mock_crew_instance = Mock()
    mock_crew.return_value = mock_crew_instance
    mock_crew_instance.kickoff.return_value = "Mocked analysis result"
    
    # Act
    result = analyze_dataset_chat("What are the data types?", "test_dataset.csv")
    
    # Assert
    assert result == "Mocked analysis result"
    mock_crew_instance.kickoff.assert_called_once_with(inputs={'file_path': 'test_dataset.csv'})


@pytest.mark.integration
def test_analyze_dataset_chat_error_handling(mock_crew, mock_llm):
    """Test error handling when LLM calls fail"""
    # Arrange
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    mock_crew_instance = Mock()
    mock_crew.return_value = mock_crew_instance
    mock_crew_instance.kickoff.side_effect = Exception("LLM API error")
    
    # Act & Assert
    with pytest.raises(Exception, match="LLM API error"):
        analyze_dataset_chat("What are the data types?", "test_dataset.csv")


@pytest.mark.unit
def test_dataset_summary_tool_integration(mock_dataset_summary_tool):
    """Test that dataset summary tool is properly integrated"""
    # Arrange
    mock_dataset_summary_tool.return_value = "Mocked dataset summary"
    
    # Act
    agent = create_data_agent()
    
    # Assert
    assert any('dataset_summary' in str(tool) for tool in agent.tools)


@pytest.mark.unit
def test_column_analysis_tool_integration(mock_column_analysis_tool):
    """Test that column analysis tool is properly integrated"""
    # Arrange
    mock_column_analysis_tool.return_value = "Mocked column analysis"
    
    # Act
    agent = create_data_agent()
    
    # Assert
    assert any('column_unique_values' in str(tool) for tool in agent.tools)


@pytest.mark.unit
@patch('agents.data_agent.Agent')
@patch('chat_main.LLM')
def test_setup_crew_returns_correct_components(mock_llm_class, mock_agent_class):
    """Test that setup_crew returns expected components"""
    # Arrange
    mock_llm_instance = Mock()
    mock_llm_class.return_value = mock_llm_instance
    mock_agent = Mock()
    mock_agent_class.return_value = mock_agent
    
    # Act
    llm, agent = setup_crew()
    
    # Assert
    # The setup_crew function should return the LLM instance
    assert llm == mock_llm_instance
    assert agent == mock_agent
    # Verify that LLM was called with correct parameters
    mock_llm_class.assert_called_once_with(
        model="ollama/qwen3:8b",
        base_url="http://localhost:11434"
    )
    mock_agent_class.assert_called_once()


@pytest.mark.integration
@pytest.mark.parametrize("response", [
    "The dataset contains 1000 rows and 5 columns with the following data types...",
    "Analysis complete. Found 3 missing values in column 'AGE'.",
    "The SEX column has 2 unique values: M (60%) and F (40%)."
])
def test_mocked_successful_analysis(mock_crew, mock_llm, response):
    """Test with mocked successful analysis response"""
    # Arrange
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    mock_crew_instance = Mock()
    mock_crew.return_value = mock_crew_instance
    mock_crew_instance.kickoff.return_value = response
    
    # Act
    result = analyze_dataset_chat("Test query", "test.csv")
    
    # Assert
    assert result == response


@pytest.mark.integration
@pytest.mark.parametrize("error", [
    Exception("API rate limit exceeded"),
    Exception("Invalid file path"),
    Exception("Network timeout"),
    Exception("Model not available")
])
def test_mocked_error_responses(mock_crew, mock_llm, error):
    """Test with mocked error responses"""
    # Arrange
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    mock_crew_instance = Mock()
    mock_crew.return_value = mock_crew_instance
    mock_crew_instance.kickoff.side_effect = error
    
    # Act & Assert
    with pytest.raises(Exception):
        analyze_dataset_chat("Test query", "test.csv")


# Integration test with fixtures
@pytest.mark.integration
def test_integration_with_fixtures(mock_llm, mock_crew):
    """Integration test using fixtures"""
    # Arrange
    mock_crew.kickoff.return_value = "Integration test result"
    
    # Act
    result = analyze_dataset_chat("Integration test query", "test.csv")
    
    # Assert
    assert result == "Integration test result"
    mock_crew.kickoff.assert_called_once()
