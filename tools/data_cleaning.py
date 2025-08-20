from crewai.tools import tool
import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional
from .utils import load_dataset, validate_column_exists, get_dataset_info
import tempfile
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
import json
from .dataset_manager import DatasetManager
from datetime import datetime

class DataCleaningReport:
    """Class to track and report data cleaning operations"""
    
    def __init__(self):
        self.operations = []
        self.rows_modified = 0
        self.columns_modified = []
        self.original_shape = None
        self.final_shape = None
    
    def add_operation(self, operation_type: str, details: str, rows_affected: int = 0, columns_affected: List[str] = None):
        """Add a cleaning operation to the report"""
        self.operations.append({
            'type': operation_type,
            'details': details,
            'rows_affected': rows_affected,
            'columns_affected': columns_affected or []
        })
        self.rows_modified += rows_affected
        if columns_affected:
            self.columns_modified.extend(columns_affected)
    
    def generate_markdown_report(self) -> str:
        """Generate a markdown report of all cleaning operations"""
        if not self.operations:
            return "## Data Cleaning Report\nNo cleaning operations were performed."
        
        report = "## Data Cleaning Report\n\n"
        report += f"**Summary:**\n"
        report += f"- Total operations performed: {len(self.operations)}\n"
        report += f"- Total rows modified: {self.rows_modified:,}\n"
        report += f"- Columns modified: {len(set(self.columns_modified))}\n"
        
        if self.original_shape and self.final_shape:
            report += f"- Original dataset shape: {self.original_shape}\n"
            report += f"- Final dataset shape: {self.final_shape}\n"
        
        report += "\n**Detailed Operations:**\n\n"
        
        for i, op in enumerate(self.operations, 1):
            report += f"### Operation {i}: {op['type']}\n"
            report += f"- **Details:** {op['details']}\n"
            if op['rows_affected'] > 0:
                report += f"- **Rows affected:** {op['rows_affected']:,}\n"
            if op['columns_affected']:
                report += f"- **Columns affected:** {', '.join(op['columns_affected'])}\n"
            report += "\n"
        
        return report

def save_cleaned_dataset(df: pd.DataFrame, original_file_path: str, suffix: str = "_cleaned") -> str:
    """
    Save the cleaned dataset to a temporary file.
    
    Args:
        df (pd.DataFrame): The cleaned dataset
        original_file_path (str): Path to the original file
        suffix (str): Suffix to add to the filename
        
    Returns:
        str: Path to the saved cleaned dataset
    """
    # Create a temporary directory for cleaned datasets
    temp_dir = tempfile.mkdtemp(prefix="cleaned_data_")
    
    # Determine file extension and save accordingly
    if original_file_path.endswith('.csv'):
        # Check if original file uses semicolon separator
        with open(original_file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            separator = ';' if ';' in first_line else ','
        
        output_path = os.path.join(temp_dir, f"cleaned_dataset{suffix}.csv")
        df.to_csv(output_path, sep=separator, index=False)
    elif original_file_path.endswith('.parquet'):
        output_path = os.path.join(temp_dir, f"cleaned_dataset{suffix}.parquet")
        df.to_parquet(output_path, index=False)
    else:
        # Default to CSV
        output_path = os.path.join(temp_dir, f"cleaned_dataset{suffix}.csv")
        df.to_csv(output_path, index=False)
    
    return output_path

@tool("Save the current working dataset to a permanent file")
def save_current_dataset_tool(output_path: str = None, format: str = "auto") -> str:
    """Save the current working dataset to a permanent file.
    
    Args:
        output_path (str): Path where to save the dataset (if None, auto-generate)
        format (str): Output format ('csv', 'parquet', or 'auto' to match original)
        
    Returns:
        str: A formatted markdown string with the save results
    """
    return save_current_dataset(output_path, format)

def save_current_dataset(output_path: str = None, format: str = "auto") -> str:
    """
    Save the current working dataset to a permanent file.
    
    Args:
        output_path (str): Path where to save the dataset
        format (str): Output format
        
    Returns:
        str: A formatted markdown string with the save results
    """
    manager = DatasetManager()
    current_df = manager.get_current_dataset()
    
    if current_df is None:
        return "❌ **Error**: No current dataset available. Please load a dataset first."
    
    current_file_path = manager.get_current_file_path()
    
    # Auto-generate output path if not provided
    if output_path is None:
        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(manager.original_file_path))[0]
        
        if format == "auto":
            # Match original format
            if current_file_path.endswith('.parquet'):
                format = "parquet"
            else:
                format = "csv"
        
        if format == "parquet":
            output_path = f"{base_name}_cleaned_{timestamp}.parquet"
        else:
            output_path = f"{base_name}_cleaned_{timestamp}.csv"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the dataset
    try:
        if format == "parquet" or output_path.endswith('.parquet'):
            current_df.to_parquet(output_path, index=False)
        else:
            # For CSV, check if original used semicolon separator
            if manager.original_file_path and manager.original_file_path.endswith('.csv'):
                try:
                    with open(manager.original_file_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        separator = ';' if ';' in first_line else ','
                except:
                    separator = ','
            else:
                separator = ','
            
            current_df.to_csv(output_path, sep=separator, index=False)
        
        # Get dataset info
        info = get_dataset_info(current_df, output_path)
        
        # Generate cleaning history summary
        history = manager.get_cleaning_history()
        history_summary = ""
        if history:
            history_summary = "\n\n## Cleaning History\n"
            for i, op in enumerate(history, 1):
                history_summary += f"{i}. **{op.get('type', 'Unknown')}** - {op.get('details', 'No details')}\n"
                if 'timestamp' in op:
                    history_summary += f"   Timestamp: {op['timestamp']}\n"
        
        return f"""# Dataset Saved Successfully

## File Information
- **Saved to:** {output_path}
- **Format:** {format.upper()}
- **Rows:** {info['total_rows']:,}
- **Columns:** {info['total_cols']:,}
- **File Size:** {os.path.getsize(output_path):,} bytes

## Dataset Summary
- **Original File:** {manager.original_file_path}
- **Current File:** {current_file_path}
- **Total Cleaning Operations:** {len(history)}

{history_summary}

## Next Steps
Your cleaned dataset is now saved at: `{output_path}`
You can use this file for further analysis or share it with others.
"""
    
    except Exception as e:
        return f"❌ **Error saving dataset**: {e}"

@tool("Handle missing values in a dataset by replacing them with mean, median, mode, or dropping rows")
def handle_missing_values_tool(file_path: str, strategy: str = "mean", columns: Optional[str] = None) -> str:
    """Handle missing values in the dataset using various strategies.
    
    Args:
        file_path (str): Path to the dataset file
        strategy (str): Strategy to handle missing values ('mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill')
        columns (str): Comma-separated list of columns to process (if None, process all columns)
        
    Returns:
        str: A formatted markdown string with the cleaning results and report
    """
    return handle_missing_values(file_path, strategy, columns)

def handle_missing_values(file_path: str, strategy: str = "mean", columns: Optional[str] = None) -> str:
    """
    Handle missing values in the dataset using various strategies.
    
    Args:
        file_path (str): Path to the dataset file
        strategy (str): Strategy to handle missing values
        columns (str): Comma-separated list of columns to process
        
    Returns:
        str: A formatted markdown string with the cleaning results and report
    """
    # Load dataset
    result = load_dataset(file_path)
    if isinstance(result, str):
        return result
    
    df = result.copy()
    report = DataCleaningReport()
    report.original_shape = df.shape
    
    # Initialize dataset manager
    manager = DatasetManager()
    manager.set_current_dataset(df, file_path)
    
    # Parse columns parameter
    target_columns = []
    if columns:
        target_columns = [col.strip() for col in columns.split(',')]
        # Validate columns exist
        for col in target_columns:
            validation = validate_column_exists(df, col)
            if validation != True:
                return validation
    else:
        target_columns = df.columns.tolist()
    
    # Count initial missing values
    initial_missing = df[target_columns].isnull().sum().sum()
    
    if initial_missing == 0:
        return "## Missing Values Analysis\n\nNo missing values found in the specified columns."
    
    # Apply strategy
    rows_affected = 0
    columns_affected = []
    
    if strategy == "drop":
        # Drop rows with missing values
        original_rows = len(df)
        df = df.dropna(subset=target_columns)
        rows_affected = original_rows - len(df)
        columns_affected = target_columns
        
        report.add_operation(
            "Drop Missing Values",
            f"Dropped {rows_affected:,} rows with missing values in columns: {', '.join(target_columns)}",
            rows_affected,
            target_columns
        )
    
    else:
        # Fill missing values
        for column in target_columns:
            if df[column].isnull().sum() > 0:
                original_missing = df[column].isnull().sum()
                
                if strategy == "mean" and pd.api.types.is_numeric_dtype(df[column]):
                    fill_value = df[column].mean()
                    df.loc[:, column] = df[column].fillna(fill_value)
                    operation_details = f"Filled {original_missing:,} missing values with mean ({fill_value:.4f})"
                
                elif strategy == "median" and pd.api.types.is_numeric_dtype(df[column]):
                    fill_value = df[column].median()
                    df.loc[:, column] = df[column].fillna(fill_value)
                    operation_details = f"Filled {original_missing:,} missing values with median ({fill_value:.4f})"
                
                elif strategy == "mode":
                    fill_value = df[column].mode().iloc[0] if not df[column].mode().empty else "Unknown"
                    df.loc[:, column] = df[column].fillna(fill_value)
                    operation_details = f"Filled {original_missing:,} missing values with mode ({fill_value})"
                
                elif strategy == "forward_fill":
                    df.loc[:, column] = df[column].fillna(method='ffill')
                    operation_details = f"Filled {original_missing:,} missing values using forward fill"
                
                elif strategy == "backward_fill":
                    df.loc[:, column] = df[column].fillna(method='bfill')
                    operation_details = f"Filled {original_missing:,} missing values using backward fill"
                
                else:
                    # Skip non-numeric columns for mean/median strategies
                    if strategy in ["mean", "median"]:
                        continue
                    else:
                        return f"❌ **Error**: Strategy '{strategy}' not supported for column '{column}' of type {df[column].dtype}"
                
                rows_affected += original_missing
                columns_affected.append(column)
                
                report.add_operation(
                    f"Fill Missing Values ({strategy})",
                    f"Column '{column}': {operation_details}",
                    original_missing,
                    [column]
                )
    
    # Save cleaned dataset and update manager
    cleaned_file_path = save_cleaned_dataset(df, file_path)
    manager.set_current_dataset(df, cleaned_file_path)
    manager.add_cleaning_operation({
        'type': f'Missing Values ({strategy})',
        'details': f'Processed {len(target_columns)} columns, {rows_affected:,} values modified',
        'file_path': cleaned_file_path
    })
    report.final_shape = df.shape
    
    # Generate output
    output = f"""# Missing Values Handling

## Strategy Applied
- **Strategy:** {strategy}
- **Columns Processed:** {', '.join(target_columns)}
- **Initial Missing Values:** {initial_missing:,}
- **Rows/Values Modified:** {rows_affected:,}

## Results
- **Cleaned Dataset Saved:** {cleaned_file_path}
- **Final Missing Values:** {df[target_columns].isnull().sum().sum():,}

{report.generate_markdown_report()}

## Next Steps
You can now use the cleaned dataset at: `{cleaned_file_path}`
Use "save current dataset" to save it permanently.
"""
    
    return output

@tool("Correct data types of columns, especially converting object columns to numeric when possible")
def correct_data_types_tool(file_path: str, columns: Optional[str] = None, force_numeric: bool = False) -> str:
    """Correct data types of columns, especially converting object columns to numeric when possible.
    
    Args:
        file_path (str): Path to the dataset file
        columns (str): Comma-separated list of columns to process (if None, process all object columns)
        force_numeric (bool): Whether to force conversion to numeric even if some values fail
        
    Returns:
        str: A formatted markdown string with the type correction results and report
    """
    return correct_data_types(file_path, columns, force_numeric)

def correct_data_types(file_path: str, columns: Optional[str] = None, force_numeric: bool = False) -> str:
    """
    Correct data types of columns, especially converting object columns to numeric when possible.
    
    Args:
        file_path (str): Path to the dataset file
        columns (str): Comma-separated list of columns to process
        force_numeric (bool): Whether to force conversion to numeric even if some values fail
        
    Returns:
        str: A formatted markdown string with the type correction results and report
    """
    # Load dataset
    result = load_dataset(file_path)
    if isinstance(result, str):
        return result
    
    df = result.copy()
    report = DataCleaningReport()
    report.original_shape = df.shape
    
    # Initialize dataset manager
    manager = DatasetManager()
    manager.set_current_dataset(df, file_path)
    
    # Parse columns parameter
    target_columns = []
    if columns:
        target_columns = [col.strip() for col in columns.split(',')]
        # Validate columns exist
        for col in target_columns:
            validation = validate_column_exists(df, col)
            if validation != True:
                return validation
    else:
        # Default to object columns
        target_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if not target_columns:
        return "## Data Type Correction\n\nNo object columns found that need type correction."
    
    # Track conversions
    conversions_made = []
    columns_affected = []
    
    for column in target_columns:
        original_dtype = df[column].dtype
        original_sample = df[column].dropna().head(3).tolist()
        
        # Try to convert to numeric
        try:
            # First, try to convert directly
            converted_series = pd.to_numeric(df[column], errors='coerce' if force_numeric else 'raise')
            new_dtype = converted_series.dtype
            
            # Check if conversion was successful (at least some values converted)
            if new_dtype != original_dtype and converted_series.notna().sum() > 0:
                df.loc[:, column] = converted_series
                conversions_made.append({
                    'column': column,
                    'from': str(original_dtype),
                    'to': str(new_dtype),
                    'sample_before': original_sample,
                    'sample_after': df[column].dropna().head(3).tolist()
                })
                columns_affected.append(column)
                
                # Count non-null values that were successfully converted
                non_null_count = df[column].notna().sum()
                
                report.add_operation(
                    "Data Type Conversion",
                    f"Column '{column}': {original_dtype} → {new_dtype} ({non_null_count:,} values converted)",
                    0,
                    [column]
                )
            elif force_numeric and converted_series.notna().sum() == 0:
                # Direct conversion failed (all NaN), try cleaning
                # Remove common non-numeric characters and try again
                cleaned_series = df[column].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                cleaned_series = cleaned_series.replace(['', 'nan', 'None'], np.nan)
                
                try:
                    converted_series = pd.to_numeric(cleaned_series, errors='coerce')
                    new_dtype = converted_series.dtype
                    
                    if new_dtype != original_dtype and converted_series.notna().sum() > 0:
                        df.loc[:, column] = converted_series
                        conversions_made.append({
                            'column': column,
                            'from': str(original_dtype),
                            'to': str(new_dtype),
                            'sample_before': original_sample,
                            'sample_after': df[column].dropna().head(3).tolist(),
                            'cleaned': True
                        })
                        columns_affected.append(column)
                        
                        non_null_count = df[column].notna().sum()
                        
                        report.add_operation(
                            "Data Type Conversion (with cleaning)",
                            f"Column '{column}': {original_dtype} → {new_dtype} after cleaning ({non_null_count:,} values converted)",
                            0,
                            [column]
                        )
                except:
                    pass

        
        except (ValueError, TypeError):
            # If direct conversion fails, try to clean the data first
            if force_numeric:
                # Remove common non-numeric characters and try again
                cleaned_series = df[column].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                cleaned_series = cleaned_series.replace(['', 'nan', 'None'], np.nan)
                
                try:
                    converted_series = pd.to_numeric(cleaned_series, errors='coerce')
                    new_dtype = converted_series.dtype
                    
                    if new_dtype != original_dtype and converted_series.notna().sum() > 0:
                        df.loc[:, column] = converted_series
                        conversions_made.append({
                            'column': column,
                            'from': str(original_dtype),
                            'to': str(new_dtype),
                            'sample_before': original_sample,
                            'sample_after': df[column].dropna().head(3).tolist(),
                            'cleaned': True
                        })
                        columns_affected.append(column)
                        
                        non_null_count = df[column].notna().sum()
                        
                        report.add_operation(
                            "Data Type Conversion (with cleaning)",
                            f"Column '{column}': {original_dtype} → {new_dtype} after cleaning ({non_null_count:,} values converted)",
                            0,
                            [column]
                        )
                except:
                    pass
    
    if not conversions_made:
        return "## Data Type Correction\n\nNo data type conversions were possible for the specified columns."
    
    # Save cleaned dataset and update manager
    cleaned_file_path = save_cleaned_dataset(df, file_path)
    manager.set_current_dataset(df, cleaned_file_path)
    manager.add_cleaning_operation({
        'type': 'Data Type Correction',
        'details': f'Converted {len(conversions_made)} columns to numeric',
        'file_path': cleaned_file_path
    })
    report.final_shape = df.shape
    
    # Generate output
    output = f"""# Data Type Correction

## Summary
- **Columns Processed:** {len(target_columns)}
- **Conversions Made:** {len(conversions_made)}
- **Force Numeric:** {force_numeric}

## Conversions Details
"""
    
    for conv in conversions_made:
        output += f"\n### {conv['column']}\n"
        output += f"- **From:** {conv['from']} → **To:** {conv['to']}\n"
        output += f"- **Sample Before:** {conv['sample_before']}\n"
        output += f"- **Sample After:** {conv['sample_after']}\n"
        if conv.get('cleaned'):
            output += f"- **Note:** Data was cleaned before conversion\n"
    
    output += f"\n## Results\n"
    output += f"- **Cleaned Dataset Saved:** {cleaned_file_path}\n"
    
    output += f"\n{report.generate_markdown_report()}\n"
    
    output += f"\n## Next Steps\n"
    output += f"You can now use the cleaned dataset at: `{cleaned_file_path}`\n"
    output += f"Use \"save current dataset\" to save it permanently.\n"
    
    return output

@tool("Normalize numerical columns using StandardScaler or RobustScaler")
def normalize_data_tool(file_path: str, method: str = "standard", columns: Optional[str] = None) -> str:
    """Normalize numerical columns using StandardScaler or RobustScaler.
    
    Args:
        file_path (str): Path to the dataset file
        method (str): Normalization method ('standard' or 'robust')
        columns (str): Comma-separated list of numerical columns to normalize
        
    Returns:
        str: A formatted markdown string with the normalization results and report
    """
    return normalize_data(file_path, method, columns)

def normalize_data(file_path: str, method: str = "standard", columns: Optional[str] = None) -> str:
    """
    Normalize numerical columns using StandardScaler or RobustScaler.
    
    Args:
        file_path (str): Path to the dataset file
        method (str): Normalization method ('standard' or 'robust')
        columns (str): Comma-separated list of numerical columns to normalize
        
    Returns:
        str: A formatted markdown string with the normalization results and report
    """
    # Load dataset
    result = load_dataset(file_path)
    if isinstance(result, str):
        return result
    
    df = result.copy()
    report = DataCleaningReport()
    report.original_shape = df.shape
    
    # Parse columns parameter
    target_columns = []
    if columns:
        target_columns = [col.strip() for col in columns.split(',')]
        # Validate columns exist and are numeric
        for col in target_columns:
            validation = validate_column_exists(df, col)
            if validation != True:
                return validation
            if not pd.api.types.is_numeric_dtype(df[col]):
                return f"❌ **Error**: Column '{col}' is not numeric (dtype: {df[col].dtype})"
    else:
        # Default to all numeric columns
        target_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not target_columns:
        return "## Data Normalization\n\nNo numerical columns found for normalization."
    
    # Choose scaler
    if method == "standard":
        scaler = StandardScaler()
        method_name = "StandardScaler (Z-score normalization)"
    elif method == "robust":
        scaler = RobustScaler()
        method_name = "RobustScaler (robust to outliers)"
    else:
        return f"❌ **Error**: Unsupported normalization method '{method}'. Use 'standard' or 'robust'."
    
    # Store original statistics
    original_stats = {}
    for col in target_columns:
        original_stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    # Apply normalization
    try:
        # Fit and transform the data
        normalized_data = scaler.fit_transform(df[target_columns])
        
        # Create new DataFrame with normalized values
        normalized_df = pd.DataFrame(normalized_data, columns=target_columns, index=df.index)
        
        # Replace original columns with normalized ones
        for col in target_columns:
            df[col] = normalized_df[col]
        
        # Calculate new statistics
        new_stats = {}
        for col in target_columns:
            new_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        # Add to report
        report.add_operation(
            f"Data Normalization ({method})",
            f"Applied {method_name} to {len(target_columns)} columns: {', '.join(target_columns)}",
            len(df),
            target_columns
        )
        
    except Exception as e:
        return f"❌ **Error during normalization**: {e}"
    
    # Save normalized dataset
    normalized_file_path = save_cleaned_dataset(df, file_path, "_normalized")
    report.final_shape = df.shape
    
    # Generate output
    output = f"""# Data Normalization

## Method Applied
- **Method:** {method_name}
- **Columns Normalized:** {len(target_columns)}
- **Columns:** {', '.join(target_columns)}

## Statistics Comparison
"""
    
    for col in target_columns:
        output += f"\n### {col}\n"
        output += f"| Statistic | Before | After |\n"
        output += f"|-----------|--------|-------|\n"
        output += f"| Mean | {original_stats[col]['mean']:.4f} | {new_stats[col]['mean']:.4f} |\n"
        output += f"| Std | {original_stats[col]['std']:.4f} | {new_stats[col]['std']:.4f} |\n"
        output += f"| Min | {original_stats[col]['min']:.4f} | {new_stats[col]['min']:.4f} |\n"
        output += f"| Max | {original_stats[col]['max']:.4f} | {new_stats[col]['max']:.4f} |\n"
    
    output += f"\n## Results\n"
    output += f"- **Normalized Dataset Saved:** {normalized_file_path}\n"
    
    output += f"\n{report.generate_markdown_report()}\n"
    
    output += f"\n## Next Steps\n"
    output += f"You can now use the normalized dataset at: `{normalized_file_path}`\n"
    
    return output

@tool("Remove duplicate rows from the dataset")
def remove_duplicates_tool(file_path: str, subset: Optional[str] = None, keep: str = "first") -> str:
    """Remove duplicate rows from the dataset.
    
    Args:
        file_path (str): Path to the dataset file
        subset (str): Comma-separated list of columns to consider for duplicates (if None, consider all columns)
        keep (str): Which duplicates to keep ('first', 'last', or False to drop all)
        
    Returns:
        str: A formatted markdown string with the deduplication results and report
    """
    return remove_duplicates(file_path, subset, keep)

def remove_duplicates(file_path: str, subset: Optional[str] = None, keep: str = "first") -> str:
    """
    Remove duplicate rows from the dataset.
    
    Args:
        file_path (str): Path to the dataset file
        subset (str): Comma-separated list of columns to consider for duplicates
        keep (str): Which duplicates to keep
        
    Returns:
        str: A formatted markdown string with the deduplication results and report
    """
    # Load dataset
    result = load_dataset(file_path)
    if isinstance(result, str):
        return result
    
    df = result.copy()
    report = DataCleaningReport()
    report.original_shape = df.shape
    
    # Parse subset parameter
    subset_columns = None
    if subset:
        subset_columns = [col.strip() for col in subset.split(',')]
        # Validate columns exist
        for col in subset_columns:
            validation = validate_column_exists(df, col)
            if validation != True:
                return validation
    
    # Count duplicates
    if subset_columns:
        duplicate_count = df.duplicated(subset=subset_columns).sum()
        subset_info = f" considering columns: {', '.join(subset_columns)}"
    else:
        duplicate_count = df.duplicated().sum()
        subset_info = " considering all columns"
    
    if duplicate_count == 0:
        return f"## Duplicate Removal\n\nNo duplicates found{subset_info}."
    
    # Remove duplicates
    original_rows = len(df)
    df = df.drop_duplicates(subset=subset_columns, keep=keep)
    rows_removed = original_rows - len(df)
    
    # Add to report
    report.add_operation(
        "Remove Duplicates",
        f"Removed {rows_removed:,} duplicate rows{subset_info} (keeping: {keep})",
        rows_removed,
        subset_columns or df.columns.tolist()
    )
    
    # Save deduplicated dataset
    deduplicated_file_path = save_cleaned_dataset(df, file_path, "_deduplicated")
    report.final_shape = df.shape
    
    # Generate output
    output = f"""# Duplicate Removal

## Summary
- **Original Rows:** {original_rows:,}
- **Duplicate Rows Found:** {duplicate_count:,}
- **Rows Removed:** {rows_removed:,}
- **Final Rows:** {len(df):,}
- **Keep Strategy:** {keep}
"""
    
    if subset_columns:
        output += f"- **Columns Considered:** {', '.join(subset_columns)}\n"
    else:
        output += f"- **Columns Considered:** All columns\n"
    
    output += f"\n## Results\n"
    output += f"- **Deduplicated Dataset Saved:** {deduplicated_file_path}\n"
    
    output += f"\n{report.generate_markdown_report()}\n"
    
    output += f"\n## Next Steps\n"
    output += f"You can now use the deduplicated dataset at: `{deduplicated_file_path}`\n"
    
    return output

@tool("Remove outliers from numerical columns using IQR method")
def remove_outliers_tool(file_path: str, columns: Optional[str] = None, method: str = "iqr", threshold: float = 1.5) -> str:
    """Remove outliers from numerical columns using IQR method.
    
    Args:
        file_path (str): Path to the dataset file
        columns (str): Comma-separated list of numerical columns to process
        method (str): Outlier detection method ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection
        
    Returns:
        str: A formatted markdown string with the outlier removal results and report
    """
    return remove_outliers(file_path, columns, method, threshold)

def remove_outliers(file_path: str, columns: Optional[str] = None, method: str = "iqr", threshold: float = 1.5) -> str:
    """
    Remove outliers from numerical columns using IQR method.
    
    Args:
        file_path (str): Path to the dataset file
        columns (str): Comma-separated list of numerical columns to process
        method (str): Outlier detection method
        threshold (float): Threshold for outlier detection
        
    Returns:
        str: A formatted markdown string with the outlier removal results and report
    """
    # Load dataset
    result = load_dataset(file_path)
    if isinstance(result, str):
        return result
    
    df = result.copy()
    report = DataCleaningReport()
    report.original_shape = df.shape
    
    # Parse columns parameter
    target_columns = []
    if columns:
        target_columns = [col.strip() for col in columns.split(',')]
        # Validate columns exist and are numeric
        for col in target_columns:
            validation = validate_column_exists(df, col)
            if validation != True:
                return validation
            if not pd.api.types.is_numeric_dtype(df[col]):
                return f"❌ **Error**: Column '{col}' is not numeric (dtype: {df[col].dtype})"
    else:
        # Default to all numeric columns
        target_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not target_columns:
        return "## Outlier Removal\n\nNo numerical columns found for outlier removal."
    
    # Track outliers per column
    outliers_info = {}
    total_outliers = 0
    
    for column in target_columns:
        if method == "iqr":
            # IQR method
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            outliers_info[column] = {
                'method': 'IQR',
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers_count': outliers_count
            }
            
        elif method == "zscore":
            # Z-score method
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers_mask = z_scores > threshold
            outliers_count = outliers_mask.sum()
            
            outliers_info[column] = {
                'method': 'Z-score',
                'threshold': threshold,
                'outliers_count': outliers_count
            }
        
        else:
            return f"❌ **Error**: Unsupported outlier detection method '{method}'. Use 'iqr' or 'zscore'."
        
        total_outliers += outliers_count
    
    if total_outliers == 0:
        return f"## Outlier Removal\n\nNo outliers found using {method} method with threshold {threshold}."
    
    # Remove outliers
    original_rows = len(df)
    outlier_mask = pd.Series([False] * len(df), index=df.index)
    
    for column in target_columns:
        if method == "iqr":
            Q1 = outliers_info[column]['Q1']
            Q3 = outliers_info[column]['Q3']
            IQR = outliers_info[column]['IQR']
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            column_outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        else:  # zscore
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            column_outliers = z_scores > threshold
        
        outlier_mask = outlier_mask | column_outliers
    
    df_cleaned = df[~outlier_mask]
    rows_removed = original_rows - len(df_cleaned)
    
    # Add to report
    report.add_operation(
        f"Remove Outliers ({method})",
        f"Removed {rows_removed:,} rows with outliers from {len(target_columns)} columns using {method} method (threshold: {threshold})",
        rows_removed,
        target_columns
    )
    
    # Save cleaned dataset
    cleaned_file_path = save_cleaned_dataset(df_cleaned, file_path, "_outliers_removed")
    report.final_shape = df_cleaned.shape
    
    # Generate output
    output = f"""# Outlier Removal

## Method Applied
- **Method:** {method.title()}
- **Threshold:** {threshold}
- **Columns Processed:** {len(target_columns)}
- **Total Outliers Removed:** {total_outliers:,}
- **Rows Removed:** {rows_removed:,}

## Outlier Details by Column
"""
    
    for column in target_columns:
        info = outliers_info[column]
        output += f"\n### {column}\n"
        output += f"- **Method:** {info['method']}\n"
        output += f"- **Outliers Found:** {info['outliers_count']:,}\n"
        
        if info['method'] == 'IQR':
            output += f"- **Q1:** {info['Q1']:.4f}\n"
            output += f"- **Q3:** {info['Q3']:.4f}\n"
            output += f"- **IQR:** {info['IQR']:.4f}\n"
            output += f"- **Lower Bound:** {info['lower_bound']:.4f}\n"
            output += f"- **Upper Bound:** {info['upper_bound']:.4f}\n"
        else:
            output += f"- **Z-score Threshold:** {info['threshold']}\n"
    
    output += f"\n## Results\n"
    output += f"- **Original Rows:** {original_rows:,}\n"
    output += f"- **Final Rows:** {len(df_cleaned):,}\n"
    output += f"- **Cleaned Dataset Saved:** {cleaned_file_path}\n"
    
    output += f"\n{report.generate_markdown_report()}\n"
    
    output += f"\n## Next Steps\n"
    output += f"You can now use the cleaned dataset at: `{cleaned_file_path}`\n"
    
    return output
