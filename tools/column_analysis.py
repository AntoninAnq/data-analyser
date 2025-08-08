from crewai.tools import tool
import pandas as pd
from .utils import load_dataset, validate_column_exists, get_dataset_info

@tool("Analyze unique values and their percentage representation in a specific column")
def column_unique_values_tool(file_path: str, column_name: str) -> str:
    """Analyze unique values in a specific column and provide their count and percentage representation.
    
    Args:
        file_path (str): Path to the dataset file (CSV or Parquet format)
        column_name (str): Name of the column to analyze
        
    Returns:
        str: A formatted markdown string containing unique values, counts, and percentages
    """
    return analyze_column_unique_values(file_path, column_name)

def analyze_column_unique_values(file_path: str, column_name: str) -> str:
    """
    Analyze unique values in a specific column and provide their count and percentage representation.
    
    Args:
        file_path (str): Path to the dataset file (CSV or Parquet format)
        column_name (str): Name of the column to analyze
        
    Returns:
        str: A formatted markdown string containing unique values, counts, and percentages
    """
    # Load dataset using utils
    result = load_dataset(file_path)
    if isinstance(result, str):
        return result  # Return error message
    
    df = result
    
    # Validate column exists using utils
    validation_result = validate_column_exists(df, column_name)
    if validation_result != True:
        return validation_result  # Return error message
    
    # Get dataset information using utils
    info = get_dataset_info(df, file_path)
    
    # Get unique values and their counts
    value_counts = df[column_name].value_counts()
    total_rows = info['total_rows']
    
    # Calculate percentages
    percentages = (value_counts / total_rows * 100).round(2)
    
    # Format the output as markdown
    output = f"""# Column Analysis: {column_name}

## Dataset Information
- **Dataset:** {info['file_name']}
- **Column:** {column_name}
- **Total Rows:** {total_rows:,}
- **Unique Values:** {len(value_counts):,}
- **Data Type:** {df[column_name].dtype}

## Unique Values Analysis
| Value | Count | Percentage | Cumulative % |
|-------|-------|------------|--------------|"""
    
    cumulative_pct = 0
    for value, count in value_counts.items():
        pct = percentages[value]
        cumulative_pct += pct
        
        # Handle different data types for display
        if pd.isna(value):
            display_value = "NULL/NaN"
        elif isinstance(value, (int, float)):
            display_value = str(value)
        else:
            display_value = str(value)
        
        output += f"\n| {display_value} | {count:,} | {pct:.2f}% | {cumulative_pct:.2f}% |"
    
    # Add summary statistics
    output += f"\n\n## Summary Statistics"
    output += f"\n- **Most Common Value:** {value_counts.index[0]} ({percentages.iloc[0]:.2f}%)"
    output += f"\n- **Least Common Value:** {value_counts.index[-1]} ({percentages.iloc[-1]:.2f}%)"
    
    # Add data quality insights
    null_count = df[column_name].isnull().sum()
    null_pct = (null_count / total_rows * 100)
    
    if null_count > 0:
        output += f"\n- **Missing Values:** {null_count:,} ({null_pct:.2f}%)"
    else:
        output += f"\n- **Missing Values:** None"
    
    # Add distribution insights
    if len(value_counts) <= 10:
        output += f"\n- **Distribution:** {len(value_counts)} unique values (good variety)"
    elif len(value_counts) <= 50:
        output += f"\n- **Distribution:** {len(value_counts)} unique values (moderate variety)"
    else:
        output += f"\n- **Distribution:** {len(value_counts)} unique values (high variety)"
    
    # Add top 5 and bottom 5 if there are many values
    if len(value_counts) > 10:
        output += f"\n\n## Top 5 Most Common Values"
        output += f"\n| Rank | Value | Count | Percentage |"
        output += f"\n|------|-------|-------|------------|"
        for i, (value, count) in enumerate(value_counts.head().items(), 1):
            pct = percentages[value]
            display_value = "NULL/NaN" if pd.isna(value) else str(value)
            output += f"\n| {i} | {display_value} | {count:,} | {pct:.2f}% |"
    
    return output 