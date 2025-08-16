from crewai import Task
from agents.data_agent import create_data_agent

def create_describe_dataset_task(agent):
    return Task(
        description='Analyze the file at {{file_path}} and provide a summary of its structure and quality.',
        expected_output='Column names, types, missing data report, and statistics.',
        agent=agent
    )

def create_chat_data_task(agent, user_query: str, file_path: str):
    """
    Create a dynamic chat-based task for data analysis.
    
    Args:
        agent: The data analysis agent
        user_query: The user's specific question or request
        file_path: Path to the dataset file
    """
    return Task(
        description=f"""Analyze the dataset at {{file_path}} and answer the user's question: "{user_query}"
        
        You have access to tools that can:
        - Analyze dataset structure and statistics (dataset_summary_tool)
        - Analyze unique values and their percentage representation in specific columns (column_unique_values_tool)
        - Calculate the mean (average) of numeric columns (column_mean_tool)
        - Calculate the standard deviation of numeric columns (column_std_tool)
        - Analyze distribution shape and skewness of numeric columns (column_skewness_tool)
        - Provide detailed summaries with markdown formatting
        
        Use the appropriate tools to answer the user's specific question about the dataset.
        If the user asks for a general description, provide a comprehensive overview.
        If they ask for specific information about a column's unique values, use the column_unique_values_tool.
        If they ask for statistical measures like mean, standard deviation, or distribution analysis, use the appropriate statistical tools.
        If they ask for specific information, focus on that aspect.
        
        **IMPORTANT**: If the user asks for visualizations, plots, charts, or graphs, you should delegate this task to the "Data Visualization Specialist" agent who has the appropriate tools for creating plots and charts. When delegating, specify the exact task and provide the file path and column name if needed. Do not attempt to create visualizations yourself - delegate to the expert.
        
        **SPECIFIC DELEGATION RULES**:
        - If they ask about "correlations" or "relationships between columns" → Delegate to create a correlation heatmap
        - If they ask about "distributions" → Delegate to create distribution plots
        - If they ask about "patterns" or "trends" → Delegate to create appropriate visualizations
        - If they ask for any type of "plot", "graph", "chart", or "visualization" → Delegate to the visualizer
        
        **DELEGATION FORMAT**: When delegating, provide:
        - Task: The specific visualization task (e.g., "create a correlation heatmap")
        - Context: File path and any specific columns or parameters needed
        - Coworker: "Data Visualization Specialist"
        
        **IMPORTANT**: Keep delegation messages concise to avoid truncation. Include only essential information.""",
        expected_output=f"A detailed response addressing: {user_query}",
        agent=agent,
        inputs={
            "file_path": file_path,
            "user_query": user_query
        }
    )