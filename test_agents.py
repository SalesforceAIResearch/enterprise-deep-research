#!/usr/bin/env python
"""
Simple test script for the agent architecture.
"""

import os
import sys
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src directory to Python path
sys.path.append(".")

# Create a simple test state class
class TestState:
    def __init__(self, topic):
        self.research_topic = topic
        self.search_query = None
        self.knowledge_gap = ""
        self.research_loop_count = 0
        self.config = None

def main():
    """Test the agent architecture directly."""
    
    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        return {"success": False, "error": "OPENAI_API_KEY environment variable is not set"}
    
    # Import the agent classes
    from src.agent_architecture import MasterResearchAgent
    
    # Create a test state with a simple research topic
    test_topic = "The impact of artificial intelligence on healthcare"
    state = TestState(test_topic)
    
    # Initialize the master agent
    master_agent = MasterResearchAgent()
    
    try:
        # Execute research
        logger.info(f"Starting research on topic: {test_topic}")
        results = master_agent.execute_research(state)
        
        # Print summary of results
        topic_complexity = results.get("research_results", {}).get("topic_complexity", "unknown")
        sources_count = len(results.get("sources_gathered", []))
        tools_used = ", ".join(results.get("tools", []))
        
        logger.info(f"Research complete!")
        logger.info(f"Topic complexity: {topic_complexity}")
        logger.info(f"Sources found: {sources_count}")
        logger.info(f"Tools used: {tools_used}")
        
        return {
            "success": True,
            "topic_complexity": topic_complexity,
            "sources_count": sources_count,
            "tools_used": results.get("tools", [])
        }
        
    except Exception as e:
        logger.error(f"Error testing agent architecture: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = main()
    
    # Print results in a formatted way
    print("\n" + "="*50)
    print("TEST RESULT:")
    print(json.dumps(result, indent=2))
    print("="*50) 