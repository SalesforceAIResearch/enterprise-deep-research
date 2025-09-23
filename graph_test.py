import os
from dotenv import load_dotenv
import importlib
import sys

# Clear out any cached module
if 'src.graph' in sys.modules:
    del sys.modules['src.graph']
if 'src.configuration' in sys.modules:
    del sys.modules['src.configuration']

# Clear any existing env vars
if "LLM_MODEL" in os.environ:
    del os.environ["LLM_MODEL"]

# Load environment variables from .env file with override
load_dotenv(override=True)

# Print environment variables
print(f"Environment variable LLM_MODEL: {os.environ.get('LLM_MODEL')}")

# Import the modules
from src.configuration import Configuration

# Create a configuration and print the model
config = Configuration()
print(f"Configuration LLM model: {config.llm_model}")

# Test with no environment variable
del os.environ["LLM_MODEL"]
config = Configuration()
print(f"Configuration LLM model with no env var: {config.llm_model}")

# Import the graph module
from src import graph

# Print the model used in a graph function
print(f"\nTesting graph module behavior:")
config = {"configurable": {"llm_model": ""}}
from src.graph import initial_query
try:
    # This will fail but we just want to see what model it tries to use
    initial_query({"research_topic": "test"}, config)
except Exception as e:
    print(f"Expected error: {str(e)[:100]}...") 