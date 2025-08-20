# Data Analyzer 

A flexible data analysis system using chat-based approaches for dataset analysis with advanced data cleaning and management capabilities.

## Features

- **Chat-Based Analysis**: Dynamic analysis based on user queries
- **Interactive Chat**: Real-time conversation with the data analysis agent
- **Data Visualization**: Create insightful plots and charts
- **Data Cleaning**: Advanced data cleaning operations with non-destructive processing
- **Dataset Management**: Track and save cleaned datasets across multiple operations
- **Markdown Output**: Well-formatted, readable analysis results
- **Multiple Dataset Support**: Works with CSV and Parquet files
- **Column Analysis**: Detailed analysis of unique values and their distributions
- **Modular Architecture**: Shared utilities for consistent data handling

## Installation

1. Install dependencies:
```bash
poetry install
```

2. Make sure Ollama is running with the required model:
```bash
ollama run qwen3:8b
```

## Usage

### 1. Chat-Based Analysis 

Ask specific questions about your dataset through Interactive Chat :

```bash
# Start interactive chat
poetry run python main.py
```
it use defaut dataset 'dataset/DD_EEC_ANNUEL_2024_data.csv'
Then ask for example :
```bash
# General overview
"Give me a general overview of this dataset"

# Specific questions
"What are the data types of the columns?"
"How many missing values are there?"
"What are the unique values in the EEC_MEASURE column?"

# Column analysis
"Show me the unique values and their percentages for the SEX column"
"Analyze the AGE column and show me the distribution of values"
```

### 2. Data Visualization

The system includes a dedicated visualization agent that can create various types of plots and visualizations. You can request visualizations through the interactive chat:

#### Through Interactive Chat

```bash
# Start interactive chat
poetry run python main.py
```

Then ask for visualizations like:
- "Create a correlation heatmap for all numerical columns"
- "Show me the distribution of the AGE column"
- "Create a scatter plot between AGE and INCOME"
- "Make a pair plot for the numerical columns"
- "Create a bar plot showing the top 10 categories"
- "Generate a comprehensive visualization of the dataset"

#### Available Visualization Types

1. **Distribution Plots**: Histograms and box plots for understanding data distributions
2. **Correlation Heatmaps**: Visualize relationships between numerical variables
3. **Pair Plots**: Scatter plot matrices showing all pairwise relationships
4. **Scatter Plots**: Analyze relationships between two specific variables
5. **Bar Plots**: Visualize categorical data and frequency distributions
6. **Comprehensive Analysis**: Multiple plots for complete dataset overview

All plots are automatically saved to the `plots/` directory with timestamps.

### 3. Interactive Chat Mode

Start an interactive conversation with the data analysis agent:

```bash
poetry run python main.py
```

Then ask questions like:
- "What's the structure of this dataset?"
- "Are there any data quality issues?"
- "What are the most common values in each column?"
- "Show me the statistical summary"
- "Analyze the unique values in the SEX column"
- "What's the distribution of values in the AGE column?"

### 4. Data Cleaning

The system includes a dedicated data cleaning agent that can perform various data cleaning operations while maintaining the original dataset intact:

#### Through Interactive Chat

```bash
# Start interactive chat
poetry run python main.py
```

Then ask for cleaning operations like:
- "Clean missing values in the dataset using mean strategy"
- "Fix data types for numerical columns that are stored as objects"
- "Normalize the AGE column using standard scaling"
- "Remove duplicate rows from the dataset"
- "Remove outliers from the INCOME column using IQR method"
- "Perform comprehensive data cleaning on the entire dataset"

#### Available Cleaning Operations

1. **Missing Values Handling**: 
   - Replace with mean, median, or mode
   - Forward fill (ffill) or backward fill (bfill)
   - Drop rows with missing values
   - Column-specific strategies

2. **Data Type Correction**:
   - Convert object columns to numeric
   - Force numeric conversion with data cleaning
   - Handle mixed data types intelligently

3. **Data Normalization**:
   - Standard scaling (z-score normalization)
   - Robust scaling (median-based scaling)
   - Column-specific normalization

4. **Duplicate Removal**:
   - Remove exact duplicates
   - Subset-based duplicate removal
   - Configurable duplicate detection

5. **Outlier Removal**:
   - IQR method (Interquartile Range)
   - Z-score method
   - Column-specific outlier detection

### 5. Dataset Management and Saving

After performing multiple cleaning operations, you can save the current working dataset:

#### Save Current Dataset

```bash
# Through interactive chat
"Save the current dataset to a new file"
"Save the cleaned dataset as 'cleaned_data.csv'"
"Save the current dataset in parquet format"
```

#### Dataset State Management

The system automatically tracks:
- Current working dataset state
- Original dataset (never modified)
- Cleaning operation history
- File paths for all intermediate datasets

#### Non-Destructive Operations

- Original dataset files are never modified
- All cleaning operations create new temporary files
- Complete operation history is maintained
- Easy rollback to previous states

### 4. Testing

Run tests to verify functionality:

```bash
poetry run pytest tests/
```

## Available Tools

### 1. Dataset Summary Tool
- **Purpose**: Provides comprehensive dataset overview
- **Features**: Data types, missing values, basic statistics, sample data
- **Usage**: Automatically used for general dataset analysis

### 2. Column Analysis Tool
- **Purpose**: Analyzes unique values and their distributions in specific columns
- **Features**: 
  - Unique value counts and percentages
  - Cumulative percentage analysis
  - Data quality insights (missing values)
  - Distribution analysis
  - Top values ranking
- **Usage**: Automatically used when asking about specific column values

### 3. Data Cleaning Tools
- **Purpose**: Perform comprehensive data cleaning operations
- **Features**:
  - Missing values handling (mean, median, mode, drop, ffill, bfill)
  - Data type correction (object to numeric conversion)
  - Data normalization (StandardScaler, RobustScaler)
  - Duplicate removal (exact and subset-based)
  - Outlier removal (IQR, Z-score methods)
  - Non-destructive operations with detailed reporting
- **Usage**: Available through interactive chat - automatically delegates to cleaning agent when data quality issues are detected

### 4. Dataset Management Tools
- **Purpose**: Track and manage dataset state across operations
- **Features**:
  - Current dataset state tracking
  - Operation history management
  - Dataset saving with custom formats
  - Original dataset preservation
  - State rollback capabilities
- **Usage**: Automatically manages dataset state during cleaning operations

### 5. Visualization Tools
- **Purpose**: Create insightful plots and charts for data analysis
- **Features**:
  - Distribution plots (histograms, box plots)
  - Correlation heatmaps
  - Pair plots (scatter plot matrices)
  - Scatter plots with trend lines
  - Bar plots for categorical data
  - Comprehensive visualization analysis
- **Usage**: Available through interactive chat - the system intelligently selects the right agent for visualization tasks

## Architecture

### Components

- **Agents**: 
  - `agents/data_agent.py` - Data analysis assistant
  - `agents/visualizer_agent.py` - Data visualization specialist
  - `agents/cleaning_agent.py` - Data cleaning specialist
- **Tasks**: 
  - `tasks/data_tasks.py` - Task definitions for data analysis
  - `tasks/visualization_tasks.py` - Task definitions for visualization
  - `tasks/cleaning_tasks.py` - Task definitions for data cleaning
- **Tools**: 
  - `tools/dataset_summary.py` - Dataset analysis tools
  - `tools/column_analysis.py` - Column-specific analysis tools
  - `tools/data_cleaning.py` - Data cleaning tools
  - `tools/dataset_manager.py` - Dataset state management
  - `tools/visualization.py` - Data visualization tools
  - `tools/utils.py` - Shared utilities for data handling
- **Main**: `main.py` - Chat-based analysis, visualization, and cleaning functions

### Agent Delegation

The system implements intelligent delegation between agents:

1. **Data Agent** → **Visualization Agent**: For chart and plot creation
2. **Data Agent** → **Cleaning Agent**: When data quality issues are detected
3. **Visualization Agent** → **Cleaning Agent**: When visualization fails due to data issues
4. **Cleaning Agent** → **Data Agent**: After successful data cleaning for further analysis

### Shared Utilities (`tools/utils.py`)

The refactored system now includes shared utilities:

- **`load_dataset()`**: Centralized dataset loading with format detection
- **`validate_column_exists()`**: Column validation with helpful error messages
- **`get_dataset_info()`**: Consistent dataset information extraction

### Dataset State Management (`tools/dataset_manager.py`)

The system includes a singleton DatasetManager for tracking dataset state:

- **Current Dataset Tracking**: Maintains the current working dataset
- **Operation History**: Records all cleaning and analysis operations
- **File Path Management**: Tracks original and temporary file paths
- **State Persistence**: Maintains state across multiple operations

## Example Queries

Here are some example queries you can try:

### Data Analysis Queries
- **General Analysis**: "Give me a comprehensive overview of this dataset"
- **Data Quality**: "Are there any missing values or data quality issues?"
- **Structure**: "What are the column names and their data types?"
- **Statistics**: "What are the basic statistics for numeric columns?"
- **Specific Columns**: "Tell me about the EEC_MEASURE column"
- **Column Analysis**: "Show me the unique values and their percentages for the SEX column"
- **Distribution**: "Analyze the AGE column and show me the distribution of values"
- **Patterns**: "What patterns do you see in the data?"

### Data Cleaning Queries
- **Missing Values**: "Clean missing values in the dataset using mean strategy"
- **Data Types**: "Fix data types for numerical columns that are stored as objects"
- **Normalization**: "Normalize the AGE column using standard scaling"
- **Duplicates**: "Remove duplicate rows from the dataset"
- **Outliers**: "Remove outliers from the INCOME column using IQR method"
- **Comprehensive Cleaning**: "Perform comprehensive data cleaning on the entire dataset"

### Dataset Management Queries
- **Save Dataset**: "Save the current dataset to a new file"
- **Custom Save**: "Save the cleaned dataset as 'my_cleaned_data.csv'"
- **Format Save**: "Save the current dataset in parquet format"
- **State Check**: "What is the current state of the dataset?"

### Visualization Queries
- **Correlation Analysis**: "Create a correlation heatmap for all numerical columns"
- **Distribution Visualization**: "Show me the distribution of the AGE column with a plot"
- **Relationship Analysis**: "Create a scatter plot between AGE and INCOME"
- **Multi-variable Analysis**: "Make a pair plot for the numerical columns"
- **Categorical Analysis**: "Create a bar plot showing the top 10 categories"
- **Comprehensive Visualization**: "Generate a comprehensive visualization of the dataset"
- **Custom Plots**: "Create a scatter plot of AGE vs INCOME colored by SEX"

## Configuration

The system uses Ollama with the `qwen3:8b` model by default. You can modify the LLM configuration in `main.py` if needed.

## File Structure

```
data-analyser/
├── agents/
│   ├── data_agent.py          # Data analysis agent
│   ├── visualizer_agent.py    # Data visualization specialist
│   └── cleaning_agent.py      # Data cleaning specialist
├── tasks/
│   ├── data_tasks.py          # Task definitions for data analysis
│   ├── visualization_tasks.py # Task definitions for visualization
│   └── cleaning_tasks.py      # Task definitions for data cleaning
├── tools/
│   ├── dataset_summary.py     # Dataset analysis tools
│   ├── column_analysis.py     # Column analysis tools
│   ├── data_cleaning.py       # Data cleaning tools
│   ├── dataset_manager.py     # Dataset state management
│   ├── visualization.py       # Data visualization tools
│   └── utils.py              # Shared utilities
├── dataset/
│   └── DD_EEC_ANNUEL_2024_data.csv
├── plots/                    # Generated visualization files
├── main.py                   # Main entry point with analysis, visualization, and cleaning
├── tests/
│   └── *.py                  # Test scripts
└── README.md                 # This file
``` 