from crewai import Task
from typing import Optional

def create_data_quality_assessment_task(cleaning_agent, file_path: str) -> Task:
    """Create a task for assessing data quality and identifying cleaning needs."""
    return Task(
        description=f"""Analyze the dataset at '{file_path}' to assess data quality and identify potential issues that need cleaning.
        
        **Your Analysis Should Include:**
        1. **Dataset Overview**: Basic information about the dataset (rows, columns, data types)
        2. **Missing Values Analysis**: Identify columns with missing values and their percentages
        3. **Data Type Issues**: Identify columns that might have incorrect data types (e.g., numeric data stored as objects)
        4. **Duplicate Detection**: Check for duplicate rows in the dataset
        5. **Outlier Detection**: Identify potential outliers in numerical columns
        6. **Data Quality Score**: Provide a qualitative assessment of overall data quality
        
        **Output Format:**
        - Provide a comprehensive markdown report
        - Include specific recommendations for cleaning operations
        - Prioritize the most critical issues that should be addressed first
        - Suggest appropriate cleaning strategies for each identified issue
        
        **Important**: This is a diagnostic task - do not modify the data, only analyze and report findings.""",
        agent=cleaning_agent,
        expected_output="A comprehensive data quality assessment report with specific cleaning recommendations",
        context=f"Dataset file: {file_path}"
    )

def create_missing_values_cleaning_task(cleaning_agent, file_path: str, strategy: str = "auto", columns: Optional[str] = None) -> Task:
    """Create a task for handling missing values in the dataset."""
    strategy_description = {
        "auto": "automatically determine the best strategy based on data characteristics",
        "mean": "replace missing values with the mean (for numerical columns)",
        "median": "replace missing values with the median (for numerical columns)",
        "mode": "replace missing values with the mode (for categorical columns)",
        "drop": "remove rows with missing values",
        "forward_fill": "fill missing values using forward fill method",
        "backward_fill": "fill missing values using backward fill method"
    }
    
    return Task(
        description=f"""Clean missing values in the dataset at '{file_path}' using the {strategy} strategy.
        
        **Strategy Details:**
        - **Method**: {strategy_description.get(strategy, strategy)}
        - **Target Columns**: {columns if columns else "All columns"}
        
        **Your Process Should Include:**
        1. **Initial Assessment**: Analyze the current state of missing values
        2. **Strategy Selection**: Choose appropriate method for each column type
        3. **Cleaning Execution**: Apply the selected strategy
        4. **Validation**: Verify that missing values have been handled appropriately
        5. **Reporting**: Provide detailed report of changes made
        
        **Important Requirements:**
        - Do not modify the original dataset - create a cleaned copy
        - Provide detailed explanation of why each strategy was chosen
        - Report the number of values modified for each column
        - Include the path to the cleaned dataset file
        - Suggest next steps for further analysis""",
        agent=cleaning_agent,
        expected_output="A detailed report of missing values cleaning with the path to the cleaned dataset",
        context=f"Dataset: {file_path}, Strategy: {strategy}, Columns: {columns or 'All'}"
    )

def create_data_type_correction_task(cleaning_agent, file_path: str, columns: Optional[str] = None, force_numeric: bool = False) -> Task:
    """Create a task for correcting data types in the dataset."""
    return Task(
        description=f"""Correct data types in the dataset at '{file_path}', focusing on converting object columns to appropriate numeric types when possible.
        
        **Parameters:**
        - **Target Columns**: {columns if columns else "All object columns"}
        - **Force Numeric**: {force_numeric} (attempt aggressive cleaning for numeric conversion)
        
        **Your Process Should Include:**
        1. **Data Type Analysis**: Identify columns with potentially incorrect data types
        2. **Conversion Strategy**: Plan appropriate conversions for each column
        3. **Safe Conversion**: Attempt conversions while preserving data integrity
        4. **Validation**: Verify that conversions were successful and meaningful
        5. **Reporting**: Document all conversions and their impact
        
        **Focus Areas:**
        - Convert object columns to numeric when the data is actually numeric
        - Handle common issues like currency symbols, commas, or text in numeric fields
        - Preserve categorical data that should remain as objects
        - Report any columns that couldn't be converted and why
        
        **Important Requirements:**
        - Do not modify the original dataset - create a cleaned copy
        - Provide before/after examples for each converted column
        - Explain the rationale behind each conversion decision
        - Include the path to the cleaned dataset file
        - Suggest next steps for analysis with the corrected data types""",
        agent=cleaning_agent,
        expected_output="A detailed report of data type corrections with examples and the path to the cleaned dataset",
        context=f"Dataset: {file_path}, Columns: {columns or 'All objects'}, Force Numeric: {force_numeric}"
    )

def create_data_normalization_task(cleaning_agent, file_path: str, method: str = "standard", columns: Optional[str] = None) -> Task:
    """Create a task for normalizing numerical data."""
    method_description = {
        "standard": "StandardScaler (Z-score normalization) - centers data around mean=0, std=1",
        "robust": "RobustScaler - similar to StandardScaler but robust to outliers"
    }
    
    return Task(
        description=f"""Normalize numerical data in the dataset at '{file_path}' using {method} normalization.
        
        **Normalization Method:**
        - **Type**: {method_description.get(method, method)}
        - **Target Columns**: {columns if columns else "All numerical columns"}
        
        **Your Process Should Include:**
        1. **Numerical Column Identification**: Identify all numerical columns for normalization
        2. **Pre-normalization Analysis**: Document original statistics (mean, std, min, max)
        3. **Normalization Application**: Apply the selected normalization method
        4. **Post-normalization Validation**: Verify the normalization was successful
        5. **Statistical Comparison**: Compare before/after statistics
        
        **Expected Results:**
        - For StandardScaler: Mean ≈ 0, Standard Deviation ≈ 1
        - For RobustScaler: Median ≈ 0, IQR-based scaling
        
        **Important Requirements:**
        - Do not modify the original dataset - create a normalized copy
        - Provide detailed statistical comparison for each normalized column
        - Explain when and why this normalization is useful
        - Include the path to the normalized dataset file
        - Suggest appropriate next steps (e.g., "Data is now ready for machine learning")""",
        agent=cleaning_agent,
        expected_output="A detailed normalization report with statistical comparisons and the path to the normalized dataset",
        context=f"Dataset: {file_path}, Method: {method}, Columns: {columns or 'All numerical'}"
    )

def create_duplicate_removal_task(cleaning_agent, file_path: str, subset: Optional[str] = None, keep: str = "first") -> Task:
    """Create a task for removing duplicate rows from the dataset."""
    return Task(
        description=f"""Remove duplicate rows from the dataset at '{file_path}'.
        
        **Parameters:**
        - **Subset Columns**: {subset if subset else "All columns"}
        - **Keep Strategy**: {keep} (which duplicate to keep: first, last, or drop all)
        
        **Your Process Should Include:**
        1. **Duplicate Analysis**: Identify and count duplicate rows
        2. **Strategy Selection**: Choose appropriate columns for duplicate detection
        3. **Duplicate Removal**: Remove duplicates according to the specified strategy
        4. **Validation**: Verify that duplicates have been removed appropriately
        5. **Impact Assessment**: Report the impact on dataset size and quality
        
        **Considerations:**
        - Determine if all columns should be considered or only specific ones
        - Choose the most appropriate 'keep' strategy based on data characteristics
        - Ensure that important information is not lost during deduplication
        - Consider the business context when deciding what constitutes a duplicate
        
        **Important Requirements:**
        - Do not modify the original dataset - create a deduplicated copy
        - Provide detailed statistics on duplicates found and removed
        - Explain the rationale behind the deduplication strategy
        - Include the path to the deduplicated dataset file
        - Suggest next steps for analysis with the clean dataset""",
        agent=cleaning_agent,
        expected_output="A detailed deduplication report with statistics and the path to the deduplicated dataset",
        context=f"Dataset: {file_path}, Subset: {subset or 'All columns'}, Keep: {keep}"
    )

def create_outlier_removal_task(cleaning_agent, file_path: str, method: str = "iqr", threshold: float = 1.5, columns: Optional[str] = None) -> Task:
    """Create a task for removing outliers from numerical columns."""
    method_description = {
        "iqr": "Interquartile Range method - identifies outliers beyond Q1 - 1.5*IQR and Q3 + 1.5*IQR",
        "zscore": "Z-score method - identifies outliers beyond the specified threshold standard deviations"
    }
    
    return Task(
        description=f"""Remove outliers from numerical columns in the dataset at '{file_path}' using the {method} method.
        
        **Parameters:**
        - **Method**: {method_description.get(method, method)}
        - **Threshold**: {threshold}
        - **Target Columns**: {columns if columns else "All numerical columns"}
        
        **Your Process Should Include:**
        1. **Outlier Analysis**: Identify outliers in each numerical column
        2. **Method Application**: Apply the selected outlier detection method
        3. **Impact Assessment**: Evaluate the impact of outlier removal on data distribution
        4. **Validation**: Ensure that outlier removal doesn't remove legitimate data points
        5. **Reporting**: Document the number of outliers removed from each column
        
        **Considerations:**
        - Outliers might be legitimate data points - consider the business context
        - Different columns might need different outlier detection strategies
        - Consider the impact on statistical analysis and machine learning models
        - Provide justification for outlier removal decisions
        
        **Important Requirements:**
        - Do not modify the original dataset - create a cleaned copy
        - Provide detailed statistics on outliers found and removed per column
        - Explain the rationale behind outlier detection and removal
        - Include the path to the cleaned dataset file
        - Suggest next steps for analysis with the outlier-free dataset""",
        agent=cleaning_agent,
        expected_output="A detailed outlier removal report with statistics and the path to the cleaned dataset",
        context=f"Dataset: {file_path}, Method: {method}, Threshold: {threshold}, Columns: {columns or 'All numerical'}"
    )

def create_comprehensive_cleaning_task(cleaning_agent, file_path: str, cleaning_steps: Optional[str] = None) -> Task:
    """Create a comprehensive cleaning task that applies multiple cleaning operations."""
    return Task(
        description=f"""Perform comprehensive data cleaning on the dataset at '{file_path}'.
        
        **Cleaning Steps to Apply:**
        {cleaning_steps if cleaning_steps else "Apply all standard cleaning operations in logical order"}
        
        **Standard Cleaning Pipeline:**
        1. **Data Quality Assessment**: Analyze the dataset to identify issues
        2. **Missing Values**: Handle missing values using appropriate strategies
        3. **Data Type Correction**: Convert object columns to numeric when possible
        4. **Duplicate Removal**: Remove duplicate rows
        5. **Outlier Detection**: Identify and handle outliers appropriately
        6. **Data Validation**: Verify the quality of cleaned data
        
        **Your Process Should Include:**
        1. **Initial Assessment**: Comprehensive analysis of current data quality
        2. **Strategic Planning**: Determine the most appropriate cleaning sequence
        3. **Iterative Cleaning**: Apply cleaning operations step by step
        4. **Quality Validation**: Verify that each cleaning step improves data quality
        5. **Comprehensive Reporting**: Document all changes and their impact
        
        **Important Requirements:**
        - Do not modify the original dataset - create cleaned copies at each step
        - Provide detailed reports for each cleaning operation
        - Explain the rationale behind each cleaning decision
        - Include paths to all intermediate and final cleaned datasets
        - Suggest next steps for analysis with the fully cleaned dataset
        - Provide a summary of overall data quality improvements""",
        agent=cleaning_agent,
        expected_output="A comprehensive cleaning report with detailed documentation of all operations and the path to the final cleaned dataset",
        context=f"Dataset: {file_path}, Steps: {cleaning_steps or 'Standard pipeline'}"
    )
