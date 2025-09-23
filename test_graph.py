import os
from dotenv import load_dotenv
from src.graph import graph
from src.configuration import Configuration, LLMProvider, SearchAPI

# Load environment variables from .env file
load_dotenv()

# Debug: print SEARCH_API value from environment
print(f"Debug - SEARCH_API from env: '{os.getenv('SEARCH_API')}'")

def test_graph():
    # Configure the graph
    # Get MAX_WEB_RESEARCH_LOOPS with better error handling
    max_loops_str = os.getenv("MAX_WEB_RESEARCH_LOOPS", "10")
    max_loops = int(max_loops_str) if max_loops_str.strip() else 10
    
    config = {
        "configurable": {
            "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
            "llm_model": os.getenv("LLM_MODEL", "o3-mini"),
            "search_api": os.getenv("SEARCH_API", "tavily"),
            "max_web_research_loops": max_loops
        }
    }
    
    # Define a research topic
    research_topic = "Salesforce acquisitions last 10 years. What are companies, who are the founders of those companies, whether if those founders are still working for Salesforce. Please list them as a table"
    
    print(f"\n{'='*80}")
    print(f"Starting research on: {research_topic}")
    print(f"Using LLM provider: {config['configurable']['llm_provider']}")
    print(f"Using LLM model: {config['configurable']['llm_model']}")
    print(f"Using search API: {config['configurable']['search_api']}")
    print(f"Max research loops: {config['configurable']['max_web_research_loops']}")
    print(f"{'='*80}\n")
    
    # Run the graph
    # Add recursion_limit to the config (outside of 'configurable')
    config["recursion_limit"] = 50
    result = graph.invoke({"research_topic": research_topic}, config=config)
    
    print(f"\n{'='*80}")
    print("--- Research Complete ---")
    print(f"{'='*80}\n")
    print(result.get("running_summary", "No summary generated"))

if __name__ == "__main__":
    test_graph()