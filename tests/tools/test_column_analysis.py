#!/usr/bin/env python3
"""
Test script for the column analysis tool
"""

from tools.column_analysis import analyze_column_unique_values

def test_column_analysis():
    """Test the column analysis tool with different columns"""
    
    file_path = "dataset/DD_EEC_ANNUEL_2024_data.csv"
    
    test_columns = [
        "EEC_MEASURE",
        "SEX", 
        "AGE",
        "TIME_PERIOD"
    ]
    
    for column in test_columns:
        print(f"\n{'='*60}")
        print(f"Testing column: {column}")
        print(f"{'='*60}")
        
        try:
            result = analyze_column_unique_values(file_path, column)
            print("✅ SUCCESS!")
            print(result)
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*60}\n")

if __name__ == "__main__":
    test_column_analysis() 