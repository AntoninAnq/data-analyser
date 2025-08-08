# Data Analyzer with CrewAI

A flexible data analysis system using CrewAI that supports both hardcoded and chat-based approaches for dataset analysis.

## Features

- **Hardcoded Analysis**: Traditional approach with predefined analysis tasks
- **Chat-Based Analysis**: Dynamic analysis based on user queries
- **Interactive Chat**: Real-time conversation with the data analysis agent
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

### 1. Hardcoded Analysis (Original Approach)

Run a predefined analysis on your dataset:

```bash
poetry run python main.py dataset/DD_EEC_ANNUEL_2024_data.csv
```

### 2. Chat-Based Analysis (New Approach)

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

### 3. Interactive Chat Mode

Start an interactive conversation with the data analysis agent:

```bash
poetry run python chat_main.py
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

## Architecture

### Components

- **Agents**: `agents/data_agent.py` - Data analysis assistant
- **Tasks**: `tasks/data_tasks.py` - Task definitions for both approaches
- **Tools**: 
  - `tools/dataset_summary.py` - Dataset analysis tools
  - `tools/column_analysis.py` - Column-specific analysis tools
  - `tools/utils.py` - Shared utilities for data handling
- **Crew**: `crew.py` - Hardcoded crew setup
- **Chat**: `chat_main.py` - Chat-based analysis interface

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

- **General Analysis**: "Give me a comprehensive overview of this dataset"
- **Data Quality**: "Are there any missing values or data quality issues?"
- **Structure**: "What are the column names and their data types?"
- **Statistics**: "What are the basic statistics for numeric columns?"
- **Specific Columns**: "Tell me about the EEC_MEASURE column"
- **Column Analysis**: "Show me the unique values and their percentages for the SEX column"
- **Distribution**: "Analyze the AGE column and show me the distribution of values"
- **Patterns**: "What patterns do you see in the data?"

## Configuration

The system uses Ollama with the `qwen3:8b` model by default. You can modify the LLM configuration in `crew.py` and `chat_main.py` if needed.

## File Structure

```
data-analyser/
├── agents/
│   └── data_agent.py          # Data analysis agent
├── tasks/
│   └── data_tasks.py          # Task definitions
├── tools/
│   ├── dataset_summary.py     # Dataset analysis tools
│   ├── column_analysis.py     # Column analysis tools
│   └── utils.py              # Shared utilities
├── dataset/
│   └── DD_EEC_ANNUEL_2024_data.csv
├── crew.py                    # Hardcoded crew setup
├── chat_main.py              # Chat-based analysis
├── main.py                   # Main entry point
├── tests/
│   └── *.py          # Test scripts
├── test_column_analysis.py   # Column analysis test
└── README.md                 # This file
``` 