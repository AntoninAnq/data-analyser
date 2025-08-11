from crewai import Agent
from tools.dataset_summary import dataset_summary_tool
from tools.column_analysis import column_unique_values_tool
from tools.statistical_analysis import column_mean_tool, column_std_tool, column_skewness_tool

def create_data_agent(llm=None):
    return Agent(
        role='Data Analysis Assistant',
        goal='Help users understand and analyze datasets by answering their specific questions and providing insights.',
        backstory="""You are an expert data analyst with deep knowledge of statistical analysis, 
        data visualization, and machine learning. You excel at understanding user needs and providing 
        clear, actionable insights from data. You can analyze datasets, identify patterns, and explain 
        complex data concepts in simple terms.""",
        tools=[
            dataset_summary_tool, 
            column_unique_values_tool,
            column_mean_tool,
            column_std_tool,
            column_skewness_tool
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=True
    )