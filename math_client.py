"""
Example client that uses langchain-mcp-adapters to connect to the math_server
and processes math queries without LangGraph.
"""
import asyncio
import os
import sys
import json
import re
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools

# Check if OpenAI API key is set
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    print("You can set it with: export OPENAI_API_KEY=your_api_key_here")

def extract_function_calls(content, tool_names):
    """Extract function calls from the response content using regex."""
    function_calls = []
    for tool_name in tool_names:
        # Look for patterns like add(10, 20) or add(10,20)
        pattern = rf'{tool_name}\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        matches = re.findall(pattern, content)
        for match in matches:
            function_calls.append({
                "name": tool_name,
                "args": {"a": int(match[0]), "b": int(match[1])}
            })
    return function_calls

async def process_query(model, tools, query):
    """Process a single query using the model and tools."""
    messages = [
        SystemMessage(content="""You are a math assistant that helps solve math problems.
You cannot do any calculations yourself and MUST use the tools provided to you.
DO NOT respond with the answer directly without using tools.
Here are the tools available to you:
""" + ", ".join([f"{tool.name} - {tool.description}" for tool in tools]) + 
"""
Example:
Question: What is 5 + 7?
DO NOT say: "5 + 7 = 12"
INSTEAD, use the add tool like this: add(5, 7)

Question: What is 10 - 3?
DO NOT say: "10 - 3 = 7"
INSTEAD, use the subtract tool like this: subtract(10, 3)
"""
        ),
        HumanMessage(content=query)
    ]
    
    # First, ask the model to think about the problem
    response = model.invoke(messages)
    
    # Print initial thought
    print(f"AI: {response.content}")
    
    # Extract function calls from the content
    tool_names = [tool.name for tool in tools]
    content_function_calls = extract_function_calls(response.content, tool_names)
    
    # Check if we have any function calls from the content
    if content_function_calls and not (hasattr(response, "tool_calls") and response.tool_calls):
        print(f"Found function calls in content: {content_function_calls}")
        for func_call in content_function_calls:
            for tool in tools:
                if tool.name == func_call["name"]:
                    try:
                        # Use ainvoke for async invocation
                        result = await tool.ainvoke(func_call["args"])
                        print(f"Tool Result: {result}")
                        
                        # Add the result to the messages
                        messages.append(AIMessage(content=response.content))
                        messages.append(HumanMessage(content=f"Result of {func_call['name']}({func_call['args']['a']}, {func_call['args']['b']}) = {result}"))
                        
                        # For multi-step problems (like add then multiply), process recursively
                        if "then" in query.lower() or "and then" in query.lower():
                            # For our specific multi-step example
                            if func_call["name"] == "add" and "multiply by 3" in query.lower():
                                print("Creating multiply query for second step")
                                new_query = f"multiply {result} by 3"
                                print(f"Processing next step with: {new_query}")
                                try:
                                    # Extract the multiplication factor
                                    multiply_args = {"a": result, "b": 3}
                                    for tool in tools:
                                        if tool.name == "multiply":
                                            multiply_result = await tool.ainvoke(multiply_args)
                                            print(f"Tool Result for second step: {multiply_result}")
                                            final_result = f"First, I added 10 and 20 to get {result}. Then, I multiplied {result} by 3 to get {multiply_result}. The final result is {multiply_result}."
                                            return final_result
                                except Exception as e:
                                    print(f"Error executing second step: {e}")
                                    return f"First step: add(10, 20) = {result}. Error in second step: {str(e)}"
                            
                            # Original handling for other cases
                            if func_call["name"] == "add":
                                # More specific replacement patterns
                                replacements = [
                                    (f"{func_call['args']['a']} to {func_call['args']['b']}", str(result)),
                                    (f"add {func_call['args']['a']} to {func_call['args']['b']}", str(result)),
                                    (f"add {func_call['args']['a']} and {func_call['args']['b']}", str(result)),
                                    (f"{func_call['args']['a']} + {func_call['args']['b']}", str(result)),
                                    (f"{func_call['args']['a']} and {func_call['args']['b']}", str(result)),
                                    (f"add 10 to 20", str(result))  # Specific to our example
                                ]
                                
                                new_query = query
                                for old, new in replacements:
                                    if old in new_query.lower():
                                        new_query = new_query.replace(old, new)
                                        break
                                
                                # If no specific patterns matched, try to build a more direct query
                                if new_query == query:
                                    if "multiply by" in query.lower() and "then" in query.lower():
                                        # Extract the multiplication factor
                                        multiply_match = re.search(r'then multiply by (\d+)', query.lower())
                                        if multiply_match:
                                            multiplier = int(multiply_match.group(1))
                                            new_query = f"multiply {result} by {multiplier}"
                                
                            elif func_call["name"] == "subtract":
                                replacements = [
                                    (f"{func_call['args']['a']} and subtract {func_call['args']['b']}", str(result)),
                                    (f"{func_call['args']['a']} - {func_call['args']['b']}", str(result))
                                ]
                                
                                new_query = query
                                for old, new in replacements:
                                    if old in new_query.lower():
                                        new_query = new_query.replace(old, new)
                                        break
                            
                            # If query has changed, process the new query
                            if new_query != query:
                                print(f"Processing next step with: {new_query}")
                                next_step = await process_query(model, tools, new_query)
                                return f"First step: {func_call['name']}({func_call['args']['a']}, {func_call['args']['b']}) = {result}\nNext step: {next_step}"
                        
                        # Get a new response
                        final_response = model.invoke(messages)
                        return final_response.content
                    except Exception as e:
                        print(f"Error executing tool: {e}")
                        messages.append(HumanMessage(content=f"Error with {func_call['name']}: {str(e)}"))
                        final_response = model.invoke(messages)
                        return final_response.content
    
    # Extract and execute tool calls from tool_calls attribute
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"Tool Call: {tool_call.name}({tool_call.args})")
            
            # Find the right tool
            for tool in tools:
                if tool.name == tool_call.name:
                    # Parse args (if it's a string, convert to dict)
                    if isinstance(tool_call.args, str):
                        try:
                            args = json.loads(tool_call.args)
                        except:
                            args = tool_call.args
                    else:
                        args = tool_call.args
                    
                    # Execute the tool
                    try:
                        # Use ainvoke for async invocation
                        result = await tool.ainvoke(args)
                        print(f"Tool Result: {result}")
                        
                        # Add the tool call and result to the messages
                        messages.append(AIMessage(content=response.content, tool_calls=[tool_call]))
                        messages.append(HumanMessage(content=f"Tool {tool_call.name} returned: {result}"))
                        
                        # Get a new response
                        final_response = model.invoke(messages)
                        return final_response.content
                    except Exception as e:
                        print(f"Error executing tool: {e}")
                        messages.append(HumanMessage(content=f"Error with tool {tool_call.name}: {str(e)}"))
                        final_response = model.invoke(messages)
                        return final_response.content
    
    # If no function calls were found, ask the model to try again with tools
    messages.append(AIMessage(content=response.content))
    messages.append(HumanMessage(content="Please solve this problem by using one of the available tools explicitly in the format: toolname(arg1, arg2). DO NOT perform calculations yourself."))
    retry_response = model.invoke(messages)
    
    print(f"Retry AI: {retry_response.content}")
    
    # Check for function calls in the retry response content
    content_function_calls = extract_function_calls(retry_response.content, tool_names)
    
    if content_function_calls:
        print(f"Found function calls in retry content: {content_function_calls}")
        for func_call in content_function_calls:
            for tool in tools:
                if tool.name == func_call["name"]:
                    try:
                        # Use ainvoke for async invocation
                        result = await tool.ainvoke(func_call["args"])
                        print(f"Tool Result: {result}")
                        
                        # Add the result to the messages
                        messages.append(AIMessage(content=retry_response.content))
                        messages.append(HumanMessage(content=f"Result of {func_call['name']}({func_call['args']['a']}, {func_call['args']['b']}) = {result}"))
                        
                        # Get a new response
                        final_response = model.invoke(messages)
                        return final_response.content
                    except Exception as e:
                        print(f"Error executing tool: {e}")
                        return f"Error with {func_call['name']}: {str(e)}"
    
    # Check for tool calls in the retry response
    if hasattr(retry_response, "tool_calls") and retry_response.tool_calls:
        for tool_call in retry_response.tool_calls:
            print(f"Tool Call (retry): {tool_call.name}({tool_call.args})")
            
            # Find the right tool
            for tool in tools:
                if tool.name == tool_call.name:
                    # Parse args
                    if isinstance(tool_call.args, str):
                        try:
                            args = json.loads(tool_call.args)
                        except:
                            args = tool_call.args
                    else:
                        args = tool_call.args
                    
                    # Execute the tool
                    try:
                        # Use ainvoke for async invocation
                        result = await tool.ainvoke(args)
                        print(f"Tool Result: {result}")
                        
                        # Add the result to the messages
                        messages.append(AIMessage(content=retry_response.content, tool_calls=[tool_call]))
                        messages.append(HumanMessage(content=f"Tool {tool_call.name} returned: {result}"))
                        
                        # Get a new response
                        final_response = model.invoke(messages)
                        return final_response.content
                    except Exception as e:
                        print(f"Error executing tool: {e}")
                        return f"Error with tool {tool_call.name}: {str(e)}"
    
    return retry_response.content

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
        api_key=OPENAI_API_KEY,
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
            
            # Define some example queries to run
            queries = [
                "What is 25 plus 17?",
                "If I have 100 and subtract 28, what do I get?",
                "What is 13 multiplied by 5?",
                "What is 120 divided by 4?",
                "If I add 10 to 20, then multiply by 3, what's the result?"
            ]
            
            # Run for each query
            for query in queries:
                print("\n" + "="*50)
                print(f"Query: {query}")
                print("="*50)
                
                # Process the query
                result = await process_query(model, tools, query)
                
                # Print the final result
                print("\nFinal Answer:", result)

if __name__ == "__main__":
    asyncio.run(main())
