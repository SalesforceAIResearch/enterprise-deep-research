"""
Test script for the visualization agent functionality.

This script tests the visualization agent in isolation to verify that
it correctly analyzes research content, generates visualization code,
and executes it using the E2B sandbox.
"""

import asyncio
import os
import json
import logging
from src.visualization_agent import VisualizationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Sample search result (mimics the output from the SearchAgent)
SAMPLE_SEARCH_RESULT = {
    "formatted_sources": [
        "1. Title: Global Smartphone Market Share 2023 (Source: https://example.com/smartphone-stats)",
        "Apple's iPhone captured 25% of the global smartphone market in 2023, while Samsung maintained its lead with 30%. Xiaomi came in third with 15%, followed by Oppo at 10% and Vivo at 8%. Other manufacturers combined for the remaining 12% market share.",
        "2. Title: Smartphone Unit Sales 2022-2023 (Source: https://example.com/smartphone-sales)",
        "Global smartphone shipments decreased by 3% in 2023 compared to 2022, with a total of 1.2 billion units shipped. Apple shipped 300 million units, Samsung 360 million, Xiaomi 180 million, Oppo 120 million, and Vivo 96 million."
    ],
    "search_string": "smartphone market share 2023",
    "subtask": {
        "type": "search",
        "name": "Smartphone Market Analysis",
        "query": "smartphone market share 2023",
        "tool": "general_search",
        "aspect": "market share statistics"
    }
}

async def test_visualization_agent():
    """Test the VisualizationAgent class."""
    print("\n=== Testing VisualizationAgent ===\n")
    
    # Initialize the visualization agent
    agent = VisualizationAgent()
    
    # Test determine_visualization_needs
    print("\n--- Testing determine_visualization_needs ---\n")
    viz_needs = await agent.determine_visualization_needs(SAMPLE_SEARCH_RESULT)
    
    if viz_needs:
        print(f"Visualization needed: {viz_needs.get('visualization_needed')}")
        print(f"Rationale: {viz_needs.get('rationale')}")
        print("\nVisualization types:")
        for viz_type in viz_needs.get("visualization_types", []):
            print(f"- {viz_type.get('type')}: {viz_type.get('description')}")
            print(f"  Data requirements: {viz_type.get('data_requirements')}")
    else:
        print("Failed to determine visualization needs")
        return
    
    # Skip the rest if visualization is not needed
    if not viz_needs.get("visualization_needed", False):
        print("Visualization not needed for this content")
        return
    
    # Test generate_visualization_code
    print("\n--- Testing generate_visualization_code ---\n")
    code_data = await agent.generate_visualization_code(SAMPLE_SEARCH_RESULT, viz_needs)
    
    if code_data:
        print(f"Generated code for {len(code_data.get('visualization_types', []))} visualization types")
        print("\nCode preview (first 300 characters):")
        code_preview = code_data.get("code", "")[:300] + "..." if code_data.get("code") else "No code generated"
        print(code_preview)
    else:
        print("Failed to generate visualization code")
        return
    
    # Test execute_visualization_code
    print("\n--- Testing execute_visualization_code ---\n")
    viz_results = await agent.execute_visualization_code(code_data)
    
    if viz_results:
        if "error" in viz_results:
            print(f"Error executing code: {viz_results.get('error')}")
        else:
            print(f"Generated {len(viz_results.get('results', []))} visualization files")
            
            # Print visualization info
            for viz in viz_results.get("results", []):
                print(f"- {viz.get('type')}: {viz.get('filename')}")
                print(f"  Path: {viz.get('filepath')}")
                
                # Check if file exists
                if os.path.exists(viz.get("filepath", "")):
                    print(f"  File exists: {os.path.getsize(viz.get('filepath', ''))} bytes")
                else:
                    print(f"  File does not exist")
    else:
        print("Failed to execute visualization code")
        return
    
    # # Test end-to-end execution
    # print("\n--- Testing end-to-end execution ---\n")
    # result = await agent.execute(SAMPLE_SEARCH_RESULT)
    
    # if result:
    #     if "error" in result:
    #         print(f"Error in end-to-end execution: {result.get('error')}")
    #     else:
    #         print("End-to-end execution successful")
    #         print(f"Visualization needs: {result.get('visualization_needs', {}).get('visualization_needed')}")
    #         print(f"Generated {len(result.get('results', []))} visualization files")
    # else:
    #     print("End-to-end execution returned None")

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_visualization_agent()) 