import pandas as pd
from typing import Union

def load_dataset(file_path: str) -> Union[pd.DataFrame, str]:
    """
    Load a dataset from file with automatic format detection.
    
    Args:
        file_path (str): Path to the dataset file (CSV or Parquet format)
        
    Returns:
        Union[pd.DataFrame, str]: DataFrame if successful, error message if failed
    """
    try:
        # Handle CSV files with automatic separator detection
        if file_path.endswith('.csv'):
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                # Check if semicolon is used as separator
                if ';' in first_line:
                    df = pd.read_csv(file_path, sep=';')
                else:
                    df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            return f"❌ **Error**: Unsupported file format. Supported formats: CSV, Parquet"
        
        return df
        
    except FileNotFoundError:
        return f"❌ **Error**: File '{file_path}' not found"
    except PermissionError:
        return f"❌ **Error**: Permission denied accessing '{file_path}'"
    except Exception as e:
        return f"❌ **Error loading dataset**: {e}"

def validate_column_exists(df: pd.DataFrame, column_name: str) -> Union[bool, str]:
    """
    Validate if a column exists in the dataset.
    
    Args:
        df (pd.DataFrame): The dataset
        column_name (str): Name of the column to check
        
    Returns:
        Union[bool, str]: True if column exists, error message if not
    """
    if column_name not in df.columns:
        available_columns = ', '.join(df.columns.tolist())
        return f"❌ **Error**: Column '{column_name}' not found in the dataset.\n\n**Available columns:** {available_columns}"
    
    return True

def get_dataset_info(df: pd.DataFrame, file_path: str) -> dict:
    """
    Get basic information about a dataset.
    
    Args:
        df (pd.DataFrame): The dataset
        file_path (str): Path to the dataset file
        
    Returns:
        dict: Dictionary containing dataset information
    """
    return {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'file_name': file_path.split('/')[-1],
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_counts': df.isnull().sum().to_dict()
    } 