from crewai.tools import tool
import pandas as pd
import numpy as np
from .utils import load_dataset, validate_column_exists, get_dataset_info

@tool("Calculate the mean (average) value of a numeric column")
def column_mean_tool(file_path: str, column_name: str) -> str:
    """Calculate the mean (average) value of a numeric column.
    
    Args:
        file_path (str): Path to the dataset file (CSV or Parquet format)
        column_name (str): Name of the numeric column to analyze
        
    Returns:
        str: A formatted markdown string containing the mean calculation and insights
    """
    return calculate_column_mean(file_path, column_name)

def calculate_column_mean(file_path: str, column_name: str) -> str:
    """
    Calculate the mean (average) value of a numeric column.
    
    Args:
        file_path (str): Path to the dataset file (CSV or Parquet format)
        column_name (str): Name of the numeric column to analyze
        
    Returns:
        str: A formatted markdown string containing the mean calculation and insights
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
    
    # Check if column is numeric
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        return f"❌ **Error**: Column '{column_name}' is not numeric. Data type: {df[column_name].dtype}"
    
    # Get dataset information using utils
    info = get_dataset_info(df, file_path)
    
    # Calculate mean
    mean_value = df[column_name].mean()
    total_rows = info['total_rows']
    null_count = df[column_name].isnull().sum()
    valid_rows = total_rows - null_count
    
    # Format the output as markdown
    output = f"""# Mean Analysis: {column_name}

## Dataset Information
- **Dataset:** {info['file_name']}
- **Column:** {column_name}
- **Total Rows:** {total_rows:,}
- **Valid Rows:** {valid_rows:,}
- **Missing Values:** {null_count:,}
- **Data Type:** {df[column_name].dtype}

## Mean Calculation
- **Mean Value:** {mean_value:.4f}

## Additional Statistics
- **Minimum:** {df[column_name].min():.4f}
- **Maximum:** {df[column_name].max():.4f}
- **Range:** {df[column_name].max() - df[column_name].min():.4f}
- **Median:** {df[column_name].median():.4f}

## Insights
"""
    
    # Add insights based on the data
    if null_count > 0:
        output += f"- **Data Quality:** {null_count:,} missing values ({null_count/total_rows*100:.2f}%) were excluded from calculation\n"
    
    if mean_value == df[column_name].median():
        output += "- **Distribution:** Mean equals median, suggesting a symmetric distribution\n"
    elif mean_value > df[column_name].median():
        output += "- **Distribution:** Mean is greater than median, suggesting right-skewed distribution\n"
    else:
        output += "- **Distribution:** Mean is less than median, suggesting left-skewed distribution\n"
    
    # Add percentile information
    percentiles = df[column_name].quantile([0.25, 0.5, 0.75])
    output += f"- **25th Percentile:** {percentiles[0.25]:.4f}\n"
    output += f"- **50th Percentile (Median):** {percentiles[0.5]:.4f}\n"
    output += f"- **75th Percentile:** {percentiles[0.75]:.4f}\n"
    
    return output


@tool("Calculate the standard deviation of a numeric column")
def column_std_tool(file_path: str, column_name: str) -> str:
    """Calculate the standard deviation of a numeric column.
    
    Args:
        file_path (str): Path to the dataset file (CSV or Parquet format)
        column_name (str): Name of the numeric column to analyze
        
    Returns:
        str: A formatted markdown string containing the standard deviation calculation and insights
    """
    return calculate_column_std(file_path, column_name)

def calculate_column_std(file_path: str, column_name: str) -> str:
    """
    Calculate the standard deviation of a numeric column.
    
    Args:
        file_path (str): Path to the dataset file (CSV or Parquet format)
        column_name (str): Name of the numeric column to analyze
        
    Returns:
        str: A formatted markdown string containing the standard deviation calculation and insights
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
    
    # Check if column is numeric
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        return f"❌ **Error**: Column '{column_name}' is not numeric. Data type: {df[column_name].dtype}"
    
    # Get dataset information using utils
    info = get_dataset_info(df, file_path)
    
    # Calculate standard deviation
    std_value = df[column_name].std()
    mean_value = df[column_name].mean()
    total_rows = info['total_rows']
    null_count = df[column_name].isnull().sum()
    valid_rows = total_rows - null_count
    
    # Calculate coefficient of variation
    cv = (std_value / mean_value) * 100 if mean_value != 0 else float('inf')
    
    # Format the output as markdown
    output = f"""# Standard Deviation Analysis: {column_name}

## Dataset Information
- **Dataset:** {info['file_name']}
- **Column:** {column_name}
- **Total Rows:** {total_rows:,}
- **Valid Rows:** {valid_rows:,}
- **Missing Values:** {null_count:,}
- **Data Type:** {df[column_name].dtype}

## Standard Deviation Calculation
- **Standard Deviation:** {std_value:.4f}
- **Mean:** {mean_value:.4f}
- **Coefficient of Variation:** {cv:.2f}%

## Additional Statistics
- **Variance:** {df[column_name].var():.4f}
- **Minimum:** {df[column_name].min():.4f}
- **Maximum:** {df[column_name].max():.4f}
- **Range:** {df[column_name].max() - df[column_name].min():.4f}

## Insights
"""
    
    # Add insights based on the data
    if null_count > 0:
        output += f"- **Data Quality:** {null_count:,} missing values ({null_count/total_rows*100:.2f}%) were excluded from calculation\n"
    
    # Interpret coefficient of variation
    if cv < 15:
        output += "- **Variability:** Low variability (CV < 15%)\n"
    elif cv < 35:
        output += "- **Variability:** Moderate variability (CV 15-35%)\n"
    else:
        output += "- **Variability:** High variability (CV > 35%)\n"
    
    # Add percentile information
    percentiles = df[column_name].quantile([0.25, 0.5, 0.75])
    iqr = percentiles[0.75] - percentiles[0.25]
    output += f"- **25th Percentile:** {percentiles[0.25]:.4f}\n"
    output += f"- **50th Percentile (Median):** {percentiles[0.5]:.4f}\n"
    output += f"- **75th Percentile:** {percentiles[0.75]:.4f}\n"
    output += f"- **Interquartile Range (IQR):** {iqr:.4f}\n"
    
    # Compare std with IQR
    if std_value > iqr:
        output += "- **Spread:** Standard deviation is greater than IQR, indicating potential outliers\n"
    else:
        output += "- **Spread:** Standard deviation is less than IQR, indicating relatively normal distribution\n"
    
    return output


@tool("Analyze the distribution shape and skewness of a numeric column")
def column_skewness_tool(file_path: str, column_name: str) -> str:
    """Analyze the distribution shape and skewness of a numeric column.
    
    Args:
        file_path (str): Path to the dataset file (CSV or Parquet format)
        column_name (str): Name of the numeric column to analyze
        
    Returns:
        str: A formatted markdown string containing skewness analysis and distribution insights
    """
    return analyze_column_skewness(file_path, column_name)

def analyze_column_skewness(file_path: str, column_name: str) -> str:
    """
    Analyze the distribution shape and skewness of a numeric column.
    
    Args:
        file_path (str): Path to the dataset file (CSV or Parquet format)
        column_name (str): Name of the numeric column to analyze
        
    Returns:
        str: A formatted markdown string containing skewness analysis and distribution insights
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
    
    # Check if column is numeric
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        return f"❌ **Error**: Column '{column_name}' is not numeric. Data type: {df[column_name].dtype}"
    
    # Get dataset information using utils
    info = get_dataset_info(df, file_path)
    
    # Calculate skewness and kurtosis
    skewness = df[column_name].skew()
    kurtosis = df[column_name].kurtosis()
    mean_value = df[column_name].mean()
    median_value = df[column_name].median()
    mode_value = df[column_name].mode().iloc[0] if len(df[column_name].mode()) > 0 else "No unique mode"
    
    total_rows = info['total_rows']
    null_count = df[column_name].isnull().sum()
    valid_rows = total_rows - null_count
    
    # Format the output as markdown
    output = f"""# Distribution Skewness Analysis: {column_name}

## Dataset Information
- **Dataset:** {info['file_name']}
- **Column:** {column_name}
- **Total Rows:** {total_rows:,}
- **Valid Rows:** {valid_rows:,}
- **Missing Values:** {null_count:,}
- **Data Type:** {df[column_name].dtype}

## Distribution Shape Analysis
- **Skewness:** {skewness:.4f}
- **Kurtosis:** {kurtosis:.4f}
- **Mean:** {mean_value:.4f}
- **Median:** {median_value:.4f}
- **Mode:** {mode_value}

## Distribution Classification
"""
    
    # Classify skewness
    if abs(skewness) < 0.5:
        output += "- **Skewness:** Approximately symmetric (|skewness| < 0.5)\n"
    elif skewness > 0.5:
        output += "- **Skewness:** Right-skewed (positive skewness > 0.5)\n"
    else:
        output += "- **Skewness:** Left-skewed (negative skewness < -0.5)\n"
    
    # Classify kurtosis
    if abs(kurtosis) < 2:
        output += "- **Kurtosis:** Mesokurtic (normal-like peaks, |kurtosis| < 2)\n"
    elif kurtosis > 2:
        output += "- **Kurtosis:** Leptokurtic (sharp peaks, kurtosis > 2)\n"
    else:
        output += "- **Kurtosis:** Platykurtic (flat peaks, kurtosis < -2)\n"
    
    # Compare mean, median, mode
    output += "\n## Central Tendency Comparison\n"
    if abs(mean_value - median_value) < 0.01:
        output += "- **Mean vs Median:** Very close, suggesting symmetric distribution\n"
    elif mean_value > median_value:
        output += "- **Mean vs Median:** Mean > Median, indicating right skew\n"
    else:
        output += "- **Mean vs Median:** Mean < Median, indicating left skew\n"
    
    # Add percentile information
    percentiles = df[column_name].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    output += f"\n## Percentile Analysis\n"
    output += f"- **10th Percentile:** {percentiles[0.1]:.4f}\n"
    output += f"- **25th Percentile:** {percentiles[0.25]:.4f}\n"
    output += f"- **50th Percentile (Median):** {percentiles[0.5]:.4f}\n"
    output += f"- **75th Percentile:** {percentiles[0.75]:.4f}\n"
    output += f"- **90th Percentile:** {percentiles[0.9]:.4f}\n"
    
    # Calculate and interpret quartile skewness
    q1, q2, q3 = percentiles[0.25], percentiles[0.5], percentiles[0.75]
    quartile_skewness = (q3 - q2) - (q2 - q1)
    output += f"- **Quartile Skewness:** {quartile_skewness:.4f}\n"
    
    if abs(quartile_skewness) < 0.1:
        output += "  - **Interpretation:** Symmetric distribution around median\n"
    elif quartile_skewness > 0.1:
        output += "  - **Interpretation:** Right-skewed distribution\n"
    else:
        output += "  - **Interpretation:** Left-skewed distribution\n"
    
    # Add insights
    output += "\n## Insights\n"
    if null_count > 0:
        output += f"- **Data Quality:** {null_count:,} missing values ({null_count/total_rows*100:.2f}%) were excluded from analysis\n"
    
    # Distribution recommendations
    if abs(skewness) < 0.5 and abs(kurtosis) < 2:
        output += "- **Statistical Tests:** Data appears approximately normal, parametric tests may be appropriate\n"
    else:
        output += "- **Statistical Tests:** Data is non-normal, consider non-parametric tests or data transformation\n"
    
    # Transformation suggestions
    if skewness > 1:
        output += "- **Transformations:** Consider log transformation or square root transformation to reduce right skew\n"
    elif skewness < -1:
        output += "- **Transformations:** Consider square transformation to reduce left skew\n"
    
    return output
