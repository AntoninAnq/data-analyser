from crewai.tools import tool
import pandas as pd
from .utils import load_dataset, get_dataset_info

@tool("Describe a dataset with types, missing values, and basic stats")
def dataset_summary_tool(file_path: str) -> str:
    """Analyze a dataset and provide information about data types, missing values, and basic statistics.
    
    Args:
        file_path (str): Path to the dataset file (CSV or Parquet format)
        
    Returns:
        str: A formatted markdown string containing data types, missing values count, and descriptive statistics
    """
    return dataset_summary(file_path)

def dataset_summary(file_path: str) -> str:
    """
    Analyze a dataset and provide information about data types, missing values, and basic statistics.
    
    Args:
        file_path (str): Path to the dataset file (CSV or Parquet format)
        
    Returns:
        str: A formatted markdown string containing data types, missing values count, and descriptive statistics
    """
    # Load dataset using utils
    result = load_dataset(file_path)
    if isinstance(result, str):
        return result  # Return error message
    
    df = result
    
    # Get dataset information using utils
    info = get_dataset_info(df, file_path)
    
    # Basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        stats_info = df[numeric_cols].describe()
    else:
        stats_info = None
    
    # Format the output as markdown
    output = f"""# Dataset Summary: {info['file_name']}

## Basic Information
- **Total Rows:** {info['total_rows']:,}
- **Total Columns:** {info['total_cols']}

## Data Types
| Column | Data Type |
|--------|-----------|"""
    
    for col, dtype in info['dtypes'].items():
        output += f"\n| {col} | {dtype} |"
    
    output += "\n\n## Missing Values"
    output += "\n| Column | Missing Count | Missing % |"
    output += "\n|--------|---------------|-----------|"
    
    for col, missing_count in info['missing_counts'].items():
        missing_pct = (missing_count / info['total_rows']) * 100
        output += f"\n| {col} | {missing_count:,} | {missing_pct:.2f}% |"
    
    if stats_info is not None and len(stats_info) > 0:
        output += "\n\n## Numeric Column Statistics"
        output += "\n| Column | Count | Mean | Std | Min | 25% | 50% | 75% | Max |"
        output += "\n|--------|-------|------|-----|-----|-----|-----|-----|-----|"
        
        for col in numeric_cols:
            stats = stats_info[col]
            output += f"\n| {col} | {stats['count']:,.0f} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | {stats['25%']:.2f} | {stats['50%']:.2f} | {stats['75%']:.2f} | {stats['max']:.2f} |"
    
    # Add sample data
    output += "\n\n## Sample Data (First 5 rows)"
    output += "\n```"
    output += f"\n{df.head().to_string()}"
    output += "\n```"
    
    return output