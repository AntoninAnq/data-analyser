from crewai import Crew, LLM, Process
from agents.data_agent import create_data_agent
from agents.visualizer_agent import create_visualizer_agent
from agents.cleaning_agent import create_cleaning_agent
from tasks.data_tasks import create_chat_data_task


def setup_crew():
    """Setup the crew with LLM and agents"""
    # Configure Ollama LLM
    llm = LLM(
        model="ollama/qwen3:8b",
        base_url="http://localhost:11434"
    )
    
    # Create agents
    data_agent = create_data_agent(llm=llm)
    visualizer_agent = create_visualizer_agent(llm=llm)
    cleaning_agent = create_cleaning_agent(llm=llm)
    
    return llm, data_agent, visualizer_agent, cleaning_agent

def analyze_dataset_chat(user_query: str, file_path: str = "dataset/DD_EEC_ANNUEL_2024_data.csv"):
    """
    Analyze a dataset using a chat-based approach with multi-agent crew.
    
    Args:
        user_query: The user's specific question about the dataset
        file_path: Path to the dataset file
    """
    llm, data_agent, visualizer_agent, cleaning_agent = setup_crew()
    
    # Create dynamic task based on user query
    # The task can be delegated to either agent based on the content
    chat_task = create_chat_data_task(data_agent, user_query, file_path)
    
    # Create crew with all agents - they can collaborate and delegate tasks
    # The cleaning agent can be called when data quality issues are detected
    crew = Crew(
        agents=[data_agent, visualizer_agent, cleaning_agent],
        tasks=[chat_task],
        llm=llm,
        verbose=True,
        process=Process.sequential  # Ensure proper delegation flow
    )
    
    # Execute the task with the file_path as input
    result = crew.kickoff(inputs={'file_path': file_path})
    return result

def interactive_chat():
    """Interactive chat interface for dataset analysis"""
    print("🤖 Data Analysis Chat Assistant")
    print("=" * 50)
    print("Ask me anything about your dataset!")
    print("Type 'quit' to exit")
    print()
    
    file_path = "dataset/DD_EEC_ANNUEL_2024_data.csv"
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not user_input:
            continue
        
        print("\n🤖 Analyzing...")
        try:
            result = analyze_dataset_chat(user_input, file_path)
            print(f"\n🤖 Assistant: {result}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    # # Example usage
    # print("Example 1: General dataset description")
    # result1 = analyze_dataset_chat("Give me a general overview of this dataset")
    # print(result1)
    # print("\n" + "="*50 + "\n")
    
    # print("Example 2: Specific question")
    # result2 = analyze_dataset_chat("What are the data types of the columns?")
    # print(result2)
    # print("\n" + "="*50 + "\n")
    
    # print("Example 3: Visualization request")
    # result3 = analyze_dataset_chat("Create a correlation heatmap for all numerical columns")
    # print(result3)
    # print("\n" + "="*50 + "\n")
    
    print("Interactive mode")
    interactive_chat() 