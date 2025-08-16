from crewai import Task
from typing import Optional

def create_distribution_analysis_task(agent, column_name: str, file_path: str = "dataset/DD_EEC_ANNUEL_2024_data.csv"):
    """
    Create a task for analyzing the distribution of a specific column.
    
    Args:
        agent: The visualizer agent
        column_name (str): Name of the column to analyze
        file_path (str): Path to the dataset file
        
    Returns:
        Task: The distribution analysis task
    """
    return Task(
        description=f"""Analyze the distribution of the '{column_name}' column in the dataset.
        
        Your task is to:
        1. First, get a summary of the dataset to understand its structure
        2. Create a distribution plot for the '{column_name}' column that shows:
           - Histogram of the values
           - Box plot to identify outliers and quartiles
        3. Provide insights about the distribution, including:
           - Whether it's normal, skewed, or has multiple modes
           - Presence of outliers
           - Key statistics (mean, median, standard deviation)
        
        Use the distribution_plot_tool to create the visualization.
        Make sure to explain what the plot reveals about the data distribution.
        """,
        agent=agent,
        expected_output=f"""A comprehensive analysis of the '{column_name}' column distribution including:
        - The generated plot file path
        - Statistical summary of the column
        - Interpretation of the distribution shape and characteristics
        - Insights about outliers or unusual patterns
        """,
        context=f"Analyzing distribution of column '{column_name}' in dataset at {file_path}",
        input_data={'file_path': file_path, 'column_name': column_name}
    )

def create_correlation_analysis_task(agent, columns: Optional[str] = None, file_path: str = "dataset/DD_EEC_ANNUEL_2024_data.csv"):
    """
    Create a task for analyzing correlations between numerical columns.
    
    Args:
        agent: The visualizer agent
        columns (str, optional): Comma-separated list of columns to analyze. If None, uses all numerical columns.
        file_path (str): Path to the dataset file
        
    Returns:
        Task: The correlation analysis task
    """
    columns_desc = f"columns: {columns}" if columns else "all numerical columns"
    
    return Task(
        description=f"""Analyze correlations between {columns_desc} in the dataset.
        
        Your task is to:
        1. First, get a summary of the dataset to understand its structure
        2. Create a correlation heatmap that shows:
           - Correlation coefficients between all numerical variables
           - Color-coded strength of correlations
           - Clear annotations of correlation values
        3. Identify and explain:
           - Strong positive correlations (close to 1.0)
           - Strong negative correlations (close to -1.0)
           - Weak correlations (close to 0.0)
           - Any surprising or interesting correlation patterns
        
        Use the correlation_heatmap_tool to create the visualization.
        Focus on the most meaningful correlations and explain their business or analytical significance.
        """,
        agent=agent,
        expected_output="""A comprehensive correlation analysis including:
        - The generated correlation heatmap file path
        - Summary of the strongest correlations found
        - Interpretation of correlation patterns
        - Business insights from the correlation analysis
        """,
        context=f"Analyzing correlations between {columns_desc} in dataset at {file_path}",
        input_data={'file_path': file_path, 'columns': columns}
    )

def create_pair_plot_analysis_task(agent, columns: Optional[str] = None, file_path: str = "dataset/DD_EEC_ANNUEL_2024_data.csv"):
    """
    Create a task for creating a pair plot to show relationships between variables.
    
    Args:
        agent: The visualizer agent
        columns (str, optional): Comma-separated list of columns to include. If None, uses all numerical columns.
        file_path (str): Path to the dataset file
        
    Returns:
        Task: The pair plot analysis task
    """
    columns_desc = f"columns: {columns}" if columns else "all numerical columns"
    
    return Task(
        description=f"""Create a comprehensive pair plot analysis using {columns_desc} in the dataset.
        
        Your task is to:
        1. First, get a summary of the dataset to understand its structure
        2. Create a pair plot that shows:
           - Scatter plots for all pairs of variables
           - Kernel density estimation (KDE) plots on the diagonal
           - Clear visualization of relationships between variables
        3. Analyze the pair plot to identify:
           - Linear relationships between variables
           - Non-linear patterns or clusters
           - Outliers or unusual data points
           - Distribution shapes of individual variables
        
        Use the pair_plot_tool to create the visualization.
        Explain the key patterns and relationships you observe in the data.
        """,
        agent=agent,
        expected_output="""A comprehensive pair plot analysis including:
        - The generated pair plot file path
        - Summary of key relationships and patterns observed
        - Identification of interesting data clusters or outliers
        - Insights about variable distributions and interactions
        """,
        context=f"Creating pair plot analysis for {columns_desc} in dataset at {file_path}",
        input_data={'file_path': file_path, 'columns': columns}
    )

def create_scatter_plot_analysis_task(agent, x_column: str, y_column: str, color_column: Optional[str] = None, file_path: str = "dataset/DD_EEC_ANNUEL_2024_data.csv"):
    """
    Create a task for creating a scatter plot between two specific columns.
    
    Args:
        agent: The visualizer agent
        x_column (str): Name of the column for x-axis
        y_column (str): Name of the column for y-axis
        color_column (str, optional): Name of the column to use for color coding
        file_path (str): Path to the dataset file
        
    Returns:
        Task: The scatter plot analysis task
    """
    color_desc = f" and colored by '{color_column}'" if color_column else ""
    
    return Task(
        description=f"""Create a detailed scatter plot analysis between '{x_column}' and '{y_column}'{color_desc}.
        
        Your task is to:
        1. First, get a summary of the dataset to understand its structure
        2. Create a scatter plot that shows:
           - Relationship between the two variables
           - Trend line to show the overall direction
           - Color coding if a third variable is specified
        3. Analyze the scatter plot to identify:
           - Strength and direction of the relationship
           - Presence of outliers or unusual points
           - Any clustering or grouping patterns
           - Whether the relationship is linear or non-linear
        
        Use the scatter_plot_tool to create the visualization.
        Calculate and interpret the correlation coefficient between the variables.
        """,
        agent=agent,
        expected_output="""A comprehensive scatter plot analysis including:
        - The generated scatter plot file path
        - Correlation coefficient and interpretation
        - Description of the relationship between variables
        - Identification of patterns, clusters, or outliers
        - Business insights from the relationship analysis
        """,
        context=f"Creating scatter plot analysis between '{x_column}' and '{y_column}'{color_desc} in dataset at {file_path}",
        input_data={'file_path': file_path, 'x_column': x_column, 'y_column': y_column, 'color_column': color_column}
    )

def create_bar_plot_analysis_task(agent, column_name: str, top_n: int = 10, file_path: str = "dataset/DD_EEC_ANNUEL_2024_data.csv"):
    """
    Create a task for creating a bar plot for categorical or discrete data.
    
    Args:
        agent: The visualizer agent
        column_name (str): Name of the column to plot
        top_n (int): Number of top categories to show
        file_path (str): Path to the dataset file
        
    Returns:
        Task: The bar plot analysis task
    """
    return Task(
        description=f"""Create a bar plot analysis for the '{column_name}' column showing the top {top_n} most frequent values.
        
        Your task is to:
        1. First, get a summary of the dataset to understand its structure
        2. Create a bar plot that shows:
           - Top {top_n} most frequent values in the column
           - Clear count labels on each bar
           - Proper formatting and readability
        3. Analyze the bar plot to identify:
           - Most common categories or values
           - Distribution of frequencies
           - Any unusual patterns or imbalances
           - Total number of unique values vs. top {top_n}
        
        Use the bar_plot_tool to create the visualization.
        Explain what the frequency distribution reveals about the data.
        """,
        agent=agent,
        expected_output="""A comprehensive bar plot analysis including:
        - The generated bar plot file path
        - Summary of the most frequent values
        - Analysis of the frequency distribution
        - Insights about data balance and patterns
        - Total unique values and missing data information
        """,
        context=f"Creating bar plot analysis for top {top_n} values in '{column_name}' column in dataset at {file_path}",
        input_data={'file_path': file_path, 'column_name': column_name, 'top_n': top_n}
    )

def create_comprehensive_visualization_task(agent, file_path: str = "dataset/DD_EEC_ANNUEL_2024_data.csv"):
    """
    Create a comprehensive visualization task that creates multiple plots for a complete data overview.
    
    Args:
        agent: The visualizer agent
        file_path (str): Path to the dataset file
        
    Returns:
        Task: The comprehensive visualization task
    """
    return Task(
        description="""Create a comprehensive visualization analysis of the dataset to provide a complete overview.
        
        Your task is to:
        1. First, get a detailed summary of the dataset to understand its structure
        2. Create multiple visualizations to provide a complete picture:
           - Distribution plots for key numerical columns
           - Correlation heatmap for all numerical variables
           - Pair plot for the most important numerical columns
           - Bar plots for key categorical columns
        3. For each visualization, provide:
           - Clear explanation of what it shows
           - Key insights and patterns identified
           - Business or analytical significance
        
        Choose the most appropriate visualization tools based on the data types and your analysis goals.
        Focus on creating insights that would be valuable for understanding the dataset.
        """,
        agent=agent,
        expected_output="""A comprehensive visualization analysis including:
        - Multiple generated plot file paths
        - Summary of key insights from each visualization
        - Overall data patterns and characteristics
        - Recommendations for further analysis
        - Business insights and implications
        """,
        context=f"Creating comprehensive visualization analysis for dataset at {file_path}",
        input_data={'file_path': file_path}
    )
