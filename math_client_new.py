"""
Example client that uses langchain-mcp-adapters to connect to the math_server
and uses the tools with a LangGraph agent.
"""
import asyncio
import os
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents import AgentType
from langchain.agents.initialize import initialize_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools

# Check if OpenAI API key is set
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    print("You can set it with: export OPENAI_API_KEY=your_api_key_here")

async def main():
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Path to the math_server.py file
    server_path = current_dir / "math_server.py"
    
    # Create server parameters for stdio connection
    server_params = StdioServerParameters(
        command=sys.executable,  # Use the current Python interpreter
        args=[str(server_path)],
    )
    
    # Initialize the chat model
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=OPENAI_API_KEY,  # Use the API key we got from the environment
    )
    
    print("Starting MCP client session...")
    
    # Connect to the MCP server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # Get the list of available tools
            tools = await load_mcp_tools(session)
            
            # Print available tools
            print(f"Loaded {len(tools)} tools from MCP server:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Initialize a structured agent that properly handles JSON schema tools
            agent_executor = initialize_agent(
                tools,
                model,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            
            # Define some example queries to run
            queries = [
                "What is 25 plus 17?",
                "If I have 100 and subtract 28, what do I get?",
                "What is 13 multiplied by 5?",
                "What is 120 divided by 4?",
                "If I add 10 to 20, then multiply by 3, what's the result?"
            ]
            
            # Run the agent for each query
            for query in queries:
                print("\n" + "="*50)
                print(f"Query: {query}")
                print("="*50)
                
                try:
                    # Invoke the agent
                    agent_response = await agent_executor.ainvoke({"input": query})
                    
                    # Print the agent's response
                    print("\nAgent Response:")
                    print(agent_response["output"])
                except Exception as e:
                    print(f"Error processing query: {e}")

if __name__ == "__main__":
    asyncio.run(main())
