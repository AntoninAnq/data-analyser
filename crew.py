from crewai import Crew, LLM
from agents.data_agent import create_data_agent
from tasks.data_tasks import create_describe_dataset_task

# Configure Ollama LLM
llm = LLM(
    model="ollama/qwen3:8b",
    base_url="http://localhost:11434"
)

# Create agent and task instances with the LLM
data_agent = create_data_agent(llm=llm)
describe_task = create_describe_dataset_task(data_agent)

# Create the crew with the LLM and instantiated components
crew = Crew(
    agents=[data_agent],
    tasks=[describe_task],
    llm=llm,
    verbose=True
)
