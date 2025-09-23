import os
from dotenv import load_dotenv
from src.configuration import Configuration

# Clear any existing env vars
if "LLM_MODEL" in os.environ:
    del os.environ["LLM_MODEL"]
if "LLM_PROVIDER" in os.environ:
    del os.environ["LLM_PROVIDER"]

# Load environment variables from .env file
load_dotenv(override=True)

# Print the raw environment variables
print(f"Env var LLM_MODEL: {os.environ.get('LLM_MODEL')}")
print(f"Env var LLM_PROVIDER: {os.environ.get('LLM_PROVIDER')}")

# Get the configuration
config = Configuration()
print(f"\nDefault Configuration:")
print(f"LLM provider: {config.llm_provider}")
print(f"LLM model: {config.llm_model}")

# Test with explicit env vars
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_MODEL"] = ""  # Empty to test the fallback

# Get a new configuration instance
config = Configuration()
print(f"\nWith explicit provider and empty model:")
print(f"LLM provider: {config.llm_provider}")
print(f"LLM model: {config.llm_model}") 