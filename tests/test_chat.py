#!/usr/bin/env python3
"""
Test script for the chat-based data analysis approach
"""

import pytest
from main import analyze_dataset_chat

#Test should be mocked
@pytest.mark.skip(reason="Full working test using llm for queries. Long to run")
def test_chat_analysis():
    """Test the chat-based analysis with different queries"""
    
    test_queries = [
        "Give me a general overview of this dataset",
        "What are the data types of the columns?",
        "How many rows and columns does this dataset have?",
        "Are there any missing values in the dataset?",
        "What are the unique values in the EEC_MEASURE column?",
        "Show me the unique values and their percentages for the SEX column",
        "Analyze the AGE column and show me the distribution of values",
        "What are the most common values in the TIME_PERIOD column?"
    ]
    
    file_path = "dataset/DD_EEC_ANNUEL_2024_data.csv"
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {query}")
        print(f"{'='*60}")
        
        try:
            result = analyze_dataset_chat(query, file_path)
            print(f"✅ SUCCESS!")
            print(f"Result: {result}")
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*60}\n")

if __name__ == "__main__":
    test_chat_analysis() 