import asyncio
import json
from src.graph import create_fresh_graph
from src.configuration import Configuration, LLMProvider, SearchAPI
from src.state import SummaryStateInput
from typing import Dict, Any

def test_unified_query(max_loops: int = 3, recursion_limit: int = 100) -> Dict[str, Any]:
    """
    Test the unified query planning and parallel search capabilities on a complex topic
    
    Args:
        max_loops: Maximum number of research loops to perform
        recursion_limit: Maximum recursion limit for the graph
        
    Returns:
        Dictionary containing the research results
    """
    # Create a configuration object - using a dict to match expected format
    config = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",  # Alternatively, use an available model
        "search_api": "tavily",  # Use available search API
        "include_raw_content": True,
        "max_web_research_loops": max_loops,
        "recursion_limit": recursion_limit
    }
    
    # Create a fresh graph
    graph = create_fresh_graph()
    
    # Set the complex research topic (on Agentic RAG systems)
    research_topic = """
    Agentic RAG Systems - Architecture, Benefits, and Implementation.
    
    I'm interested in understanding the architectural components, advanced techniques,
    benefits, and implementation approaches for Agentic RAG systems that enhance
    traditional Retrieval-Augmented Generation with agent-based capabilities.
    
    Please cover core components, architectural patterns, strategic retrieval planning,
    multi-hop reasoning, adaptive retrieval, self-improvement mechanisms, and notable
    implementations.
    """
    
    # Create input object
    input_obj = SummaryStateInput(research_topic=research_topic)
    
    # Add tracing callback to debug state transitions
    def trace_state_changes(state, node_name=None):
        if node_name == "research_agent" and hasattr(state, "subtopic_queries"):
            print(f"\n[DEBUG] After research_agent node, state.subtopic_queries: {state.subtopic_queries}")
    
    try:
        # Run the graph with the input
        print(f"\n[TEST] Starting unified query planning test with max_loops={max_loops}, recursion_limit={recursion_limit}...")
        
        # Configure tracing in the config
        config["callbacks"] = {"on_node_run": trace_state_changes}
        
        # Run the graph with the input
        results = graph.invoke(
            input_obj,
            {"configurable": config}
        )
        
        # Print the final summary
        print("\n[TEST] ========== RESEARCH RESULTS ==========\n")
        print(f"[TEST] Final summary length: {len(results['running_summary']) if 'running_summary' in results else 0} characters")
        print(f"[TEST] Research loops completed: {results.get('research_loop_count', 0)}")
        
        # Print additional debug info
        print(f"[TEST] Has subtopic_queries: {'subtopic_queries' in results}")
        if 'subtopic_queries' in results:
            print(f"[TEST] Number of subtopic queries: {len(results['subtopic_queries'])}")
            print(f"[TEST] Subtopic queries: {results['subtopic_queries']}")
        
        # Print the first 500 characters of the summary as a preview
        if 'running_summary' in results and results['running_summary']:
            print("\n[TEST] ----- Summary Preview (first 500 chars) -----")
            print(results['running_summary'][:500])
            print("...\n")
        
        print("[TEST] ----- Source Citations -----")
        if results.get('source_citations'):
            for num, src in sorted(results['source_citations'].items()):
                print(f"[{num}] {src.get('title', 'No title')} : {src.get('url', 'No URL')}")
        else:
            print("[TEST] No source citations found")
        
        return results
        
    except Exception as e:
        print(f"\n[TEST] ERROR: {str(e)}")
        print("[TEST] Test failed but we can still analyze what happened up to this point.")
        return {"error": str(e), "research_topic": research_topic}

if __name__ == "__main__":
    # Run the test with minimal loops and a high recursion limit to avoid errors
    # We'll focus on the initial query planning and parallel search
    results = test_unified_query(max_loops=0, recursion_limit=200)
    
    # Save output to file even if we had an error
    output_file = "unified_query_results.md"
    
    try:
        with open(output_file, "w") as f:
            f.write("# Unified Query Planning Research Results\n\n")
            f.write("## Research Topic\n")
            f.write(f"{results.get('research_topic', 'Topic not available')}\n\n")
            
            if "error" in results:
                f.write(f"## Error Encountered\n\n")
                f.write(f"{results['error']}\n\n")
            
            if "running_summary" in results:
                f.write("## Full Research Summary\n\n")
                f.write(results['running_summary'])
                f.write("\n\n")
            
            if results.get('source_citations'):
                f.write("## Source Citations\n\n")
                for num, src in sorted(results['source_citations'].items()):
                    f.write(f"[{num}] {src.get('title', 'No title')} : {src.get('url', 'No URL')}\n")
            else:
                f.write("## Source Citations\n\nNo source citations found\n")
    
        print(f"\n[TEST] Results saved to '{output_file}'")
    except Exception as e:
        print(f"\n[TEST] Error saving results: {str(e)}")
        
    # Save a debug file with full results as JSON for inspection
    try:
        debug_results = {k: v for k, v in results.items() if k != "running_summary"}
        if "running_summary" in results:
            debug_results["summary_length"] = len(results["running_summary"])
            
        with open("unified_query_debug.json", "w") as f:
            json.dump(debug_results, f, indent=2, default=str)
            
        print("[TEST] Debug information saved to 'unified_query_debug.json'")
    except Exception as e:
        print(f"[TEST] Error saving debug info: {str(e)}") 