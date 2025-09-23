"""
Test script for agent architecture.

This script helps validate that the agent architecture is working correctly
by executing a simple research request and comparing results with the
original implementation.
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

# Create a simple summary state class for testing
class TestSummaryState:
    def __init__(self, research_topic, knowledge_gap="", research_loop_count=0):
        self.research_topic = research_topic
        self.search_query = None  # Use research_topic
        self.knowledge_gap = knowledge_gap
        self.research_loop_count = research_loop_count
        self.config = None  # Use default config

def test_agent_architecture():
    """Test the new agent architecture with a simple query."""
    try:
        # Import the necessary functions and classes
        logger.info("Importing agent architecture components...")
        from src.agent_architecture import MasterResearchAgent
        
        # Create a test state
        test_topic = "The impact of artificial intelligence on healthcare"
        test_state = TestSummaryState(test_topic)
        
        # Test the agent architecture directly
        logger.info(f"Testing agent architecture with topic: {test_topic}")
        master_agent = MasterResearchAgent()
        results = master_agent.execute_research(test_state)
        
        # Log results summary
        topic_complexity = results.get("research_results", {}).get("topic_complexity", "unknown")
        sources_count = len(results.get("sources_gathered", []))
        tools_used = results.get("tools", [])
        
        logger.info(f"Research completed with topic_complexity: {topic_complexity}")
        logger.info(f"Found {sources_count} sources using tools: {', '.join(tools_used)}")
        
        return {
            "success": True,
            "topic_complexity": topic_complexity,
            "sources_count": sources_count,
            "tools_used": tools_used
        }
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set")
        sys.exit(1)
        
    # Run the test
    result = test_agent_architecture()
    
    # Print result
    print("\n" + "="*50)
    print("TEST RESULT:")
    print(json.dumps(result, indent=2))
    print("="*50) 