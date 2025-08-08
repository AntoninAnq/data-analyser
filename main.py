from crew import crew
from chat_main import analyze_dataset_chat
import sys


def run_chat_analysis(file_path, query):
    """Run chat-based analysis"""
    print(f"Running chat-based analysis for query: '{query}'")
    result = analyze_dataset_chat(query, file_path)
    print("\n--- Analysis Result ---\n")
    print(result)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py <path_to_dataset>                    # Hardcoded analysis")
        print("  python main.py <path_to_dataset> <query>           # Chat-based analysis")
        print("\nExamples:")
        print("  python main.py dataset/DD_EEC_ANNUEL_2024_data.csv")
        print("  python main.py dataset/DD_EEC_ANNUEL_2024_data.csv 'What are the data types?'")
        exit(1)

    file_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        # Chat-based approach
        query = " ".join(sys.argv[2:])
        run_chat_analysis(file_path, query)
    else:
        # Ask for query interactively
        print(f"\nDataset file: {file_path}")
        print("Enter your analysis query (or press Enter for default analysis):")
        query = input("> ").strip()
        
        if query:
            run_chat_analysis(file_path, query)
        else:
            print("No query provided. Exiting.")
            exit(0)
