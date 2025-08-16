# Data Analyzer 

A flexible data analysis system using chat-based approaches for dataset analysis.

## Features

- **Chat-Based Analysis**: Dynamic analysis based on user queries
- **Interactive Chat**: Real-time conversation with the data analysis agent
- **Data Visualization**: Create insightful plots and charts
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

Ask specific questions about your dataset:

```bash
# General overview
poetry run python main.py dataset/DD_EEC_ANNUEL_2024_data.csv "Give me a general overview of this dataset"

# Specific questions
poetry run python main.py dataset/DD_EEC_ANNUEL_2024_data.csv "What are the data types of the columns?"
poetry run python main.py dataset/DD_EEC_ANNUEL_2024_data.csv "How many missing values are there?"
poetry run python main.py dataset/DD_EEC_ANNUEL_2024_data.csv "What are the unique values in the EEC_MEASURE column?"

# Column analysis
poetry run python main.py dataset/DD_EEC_ANNUEL_2024_data.csv "Show me the unique values and their percentages for the SEX column"
poetry run python main.py dataset/DD_EEC_ANNUEL_2024_data.csv "Analyze the AGE column and show me the distribution of values"
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

### 4. Testing

Run tests to verify functionality:

```bash
poetry run python tests/test_chat.py
poetry run python test_column_analysis.py
poetry run python test_refactored_tools.py
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

### 3. Visualization Tools
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
- **Tasks**: 
  - `tasks/data_tasks.py` - Task definitions for data analysis
  - `tasks/visualization_tasks.py` - Task definitions for visualization
- **Tools**: 
  - `tools/dataset_summary.py` - Dataset analysis tools
  - `tools/column_analysis.py` - Column-specific analysis tools
  - `tools/visualization.py` - Data visualization tools
  - `tools/utils.py` - Shared utilities for data handling
- **Main**: `main.py` - Chat-based analysis and visualization functions


### Key Improvements

1. **Flexible Query System**: Ask any question about your dataset
2. **Markdown Output**: Better formatted, readable results
3. **Semicolon CSV Support**: Properly handles European CSV format
4. **Memory**: Agent remembers previous interactions
5. **Error Handling**: Robust error handling and user feedback
6. **Column Analysis**: Dedicated tool for detailed column analysis
7. **Modular Design**: Shared utilities for consistent data handling
8. **Code Reusability**: Eliminated code duplication between tools

### Shared Utilities (`tools/utils.py`)

The refactored system now includes shared utilities:

- **`load_dataset()`**: Centralized dataset loading with format detection
- **`validate_column_exists()`**: Column validation with helpful error messages
- **`get_dataset_info()`**: Consistent dataset information extraction

Benefits:
- **Consistency**: All tools use the same data loading logic
- **Maintainability**: Changes to data handling only need to be made in one place
- **Error Handling**: Centralized error handling for file operations
- **Extensibility**: Easy to add new tools that use the same utilities

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

### Visualization Queries
- **Correlation Analysis**: "Create a correlation heatmap for all numerical columns"
- **Distribution Visualization**: "Show me the distribution of the AGE column with a plot"
- **Relationship Analysis**: "Create a scatter plot between AGE and INCOME"
- **Multi-variable Analysis**: "Make a pair plot for the numerical columns"
- **Categorical Analysis**: "Create a bar plot showing the top 10 categories"
- **Comprehensive Visualization**: "Generate a comprehensive visualization of the dataset"
- **Custom Plots**: "Create a scatter plot of AGE vs INCOME colored by SEX"

## Configuration

The system uses Ollama with the `qwen3:8b` model by default. You can modify the LLM configuration in `crew.py` and `chat_main.py` if needed.

## File Structure

```
data-analyser/
├── agents/
│   ├── data_agent.py          # Data analysis agent
│   └── visualizer_agent.py    # Data visualization specialist
├── tasks/
│   ├── data_tasks.py          # Task definitions for data analysis
│   └── visualization_tasks.py # Task definitions for visualization
├── tools/
│   ├── dataset_summary.py     # Dataset analysis tools
│   ├── column_analysis.py     # Column analysis tools
│   ├── visualization.py       # Data visualization tools
│   └── utils.py              # Shared utilities
├── dataset/
│   └── DD_EEC_ANNUEL_2024_data.csv
├── plots/                    # Generated visualization files
├── main.py                   # Main entry point with analysis and visualization
├── tests/
│   └── *.py                  # Test scripts
└── README.md                 # This file
``` 