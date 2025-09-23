import os
from dotenv import load_dotenv, find_dotenv
import sys
import json
import argparse
import subprocess
import platform
import base64
import re
from datetime import datetime
import importlib.util
import time
import anthropic
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables from .env file with override
dotenv_path = find_dotenv()
load_dotenv(dotenv_path, override=True)

# Import LLM client utilities from our new module
from llm_clients import (
    get_llm_client, 
    get_model_response, 
    get_available_providers,
    MODEL_CONFIGS,
    SimpleOpenAIClient,
    SYSTEM_PROMPT_TEMPLATE,
    CURRENT_DATE,
    CURRENT_YEAR,
    CURRENT_MONTH, 
    CURRENT_DAY,
    ONE_YEAR_AGO,
    YTD_START,
    ERROR_CORRECTION_PROMPT,
    get_formatted_system_prompt
)

from e2b_code_interpreter import Sandbox

# API keys for different providers
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
E2B_API_KEY = os.getenv("E2B_API_KEY")
SAMBNOVA_API_KEY = os.getenv("SAMBNOVA_API_KEY")

# Configuration settings
MAX_RETRY_ATTEMPTS = 2  # Maximum number of retries for code generation when execution fails

# Check E2B API key
if not E2B_API_KEY:
    raise ValueError("E2B_API_KEY environment variable not set")

def code_interpret(sandbox, code):
    print("Running code interpreter...")
    try:
        execution = sandbox.run_code(
            code,
            # You can also stream code execution results with callbacks
            # on_stderr=lambda stderr: print("[Code Interpreter]", stderr),
            # on_stdout=lambda stdout: print("[Code Interpreter]", stdout),
        )

        if execution.error:
            print("[Code Interpreter ERROR]", execution.error)
            return None, execution
        print(execution)
        print("--------------------------------")
        print(execution.results)
        return execution.results, execution
    except Exception as e:
        print(f"[Code Interpreter EXCEPTION] {str(e)}")
        return None, None

def extract_execution_logs(execution):
    """Extract stdout and stderr logs from execution object"""
    if not execution:
        return "No execution logs available."
    
    logs = []
    
    # Extract stdout logs
    if hasattr(execution, 'logs') and hasattr(execution.logs, 'stdout'):
        stdout_logs = ''.join(execution.logs.stdout) if execution.logs.stdout else ""
        if stdout_logs:
            logs.append(f"STDOUT:\n{stdout_logs}")
    
    # Extract stderr logs
    if hasattr(execution, 'logs') and hasattr(execution.logs, 'stderr'):
        stderr_logs = ''.join(execution.logs.stderr) if execution.logs.stderr else ""
        if stderr_logs:
            logs.append(f"STDERR:\n{stderr_logs}")
    
    # Extract execution error if present
    if hasattr(execution, 'error') and execution.error:
        logs.append(f"ERROR:\n{execution.error}")
    
    if not logs:
        return "Execution completed but returned no results or logs."
    
    return "\n\n".join(logs)

def match_code_blocks(llm_response):
    print("\nDEBUG - Full LLM Response:")
    print(f"===\n{llm_response}\n===")
    
    # Extract all Python code blocks
    python_pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    matches = python_pattern.findall(llm_response)
    
    if matches:
        print(f"Found {len(matches)} code blocks with ```python tag")
        # Combine all code blocks into a single script with comments
        combined_code = ""
        for i, code_block in enumerate(matches):
            combined_code += f"\n# Code Block {i+1}\n{code_block.strip()}\n"
        print(combined_code)
        return combined_code, True  # Return the code and a flag indicating code was found
    
    # Try to match code blocks with just ```
    alt_pattern = re.compile(r"```(?!python)(.*?)```", re.DOTALL)
    matches = alt_pattern.findall(llm_response)
    if matches:
        print(f"Found {len(matches)} code blocks with ``` tag")
        combined_code = ""
        for i, code_block in enumerate(matches):
            combined_code += f"\n# Code Block {i+1}\n{code_block.strip()}\n"
        print(combined_code)
        return combined_code, True
    
    # Try to match code blocks with dashed lines (o3-mini style)
    dash_pattern = re.compile(r"-{4,}\n(.*?)\n-{4,}", re.DOTALL)
    matches = dash_pattern.findall(llm_response)
    if matches:
        print(f"Found {len(matches)} code blocks with dashed lines separator")
        combined_code = ""
        for i, code_block in enumerate(matches):
            combined_code += f"\n# Code Block {i+1}\n{code_block.strip()}\n"
        print(combined_code)
        return combined_code, True
    
    # Last resort: If there are any lines that look like Python code, extract them
    if "def " in llm_response or "print(" in llm_response or "import " in llm_response:
        print("Extracting Python-looking code as a last resort")
        lines = llm_response.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('print(') or line.strip().startswith('import ') or in_code_block:
                in_code_block = True
                code_lines.append(line)
                
        if code_lines:
            code = '\n'.join(code_lines)
            print(code)
            return code, True
    
    print("No code blocks found in the response")
    return llm_response, False  # Return the original response and a flag indicating no code was found

def extract_explanation_text(llm_response):
    """Extract the explanatory text from the LLM response by removing code blocks."""
    # Remove Python code blocks
    text = re.sub(r"```python\n.*?\n```", "[Code block removed for clarity]", llm_response, flags=re.DOTALL)
    
    # Remove other code blocks
    text = re.sub(r"```.*?```", "[Code block removed for clarity]", text, flags=re.DOTALL)
    
    # Remove dashed line code blocks
    text = re.sub(r"-{4,}\n.*?\n-{4,}", "[Code block removed for clarity]", text, flags=re.DOTALL)
    
    return text

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run code generation using different LLM providers")
    
    # Get available providers
    available_providers = get_available_providers()
    
    # Use environment variables if available, otherwise use the first available provider
    default_provider = os.getenv("LLM_PROVIDER") or (available_providers[0] if available_providers else None)
    default_model = os.getenv("LLM_MODEL")
    
    # Add provider argument
    parser.add_argument(
        "--provider", 
        type=str, 
        choices=["groq", "openai", "anthropic", "sambnova"],
        default=default_provider,
        help="LLM provider to use (default: from LLM_PROVIDER env var or first available provider)"
    )
    
    # Add model argument
    parser.add_argument(
        "--model", 
        type=str,
        default=default_model, 
        help="Specific model to use (default: from LLM_MODEL env var or provider's default model). "
             "For OpenAI, use 'o4-mini' (latest cost-efficient reasoning model) or 'o4-mini-high' "
             "for enhanced performance. For Claude 4, use 'claude-sonnet-4' (flagship model) or "
             "'claude-sonnet-4-thinking' for extended thinking mode. For Claude 3.7, use "
             "'claude-3-7-sonnet' for standard mode or 'claude-3-7-sonnet-thinking' for deeper reasoning."
    )
    
    # Add prompt arguments with multiple options
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt", 
        type=str,
        help="Custom user prompt as a string"
    )
    prompt_group.add_argument(
        "--prompt-file", 
        type=str,
        help="Path to a file containing the user prompt"
    )
    
    return parser.parse_args()

def upload_dataset(sandbox):
    """Upload the dataset to the sandbox"""
    print("Uploading dataset to Sandbox...")
    dataset_path = "./data.csv"
    
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "rb") as f:
                file_data = f.read()
                # Upload to the sandbox at /home/user/data.csv
                sandbox.files.write("/home/user/data.csv", file_data)
            print("Dataset uploaded successfully")
            return True
        except Exception as e:
            print(f"Error uploading dataset: {str(e)}")
            return False
    else:
        print(f"Dataset file not found at {dataset_path}, but continuing without it")
        return True  # Return True to continue even without the dataset

def ensure_output_dir(directory="code_output"):
    """Ensure the output directory exists."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"📁 Created output directory: {directory}")
        except Exception as e:
            print(f"❌ Error creating directory {directory}: {str(e)}")
            return False
    return True

def save_explanation_text(explanation, filename="explanation.md", output_dir="code_output"):
    """Save the explanation text from the LLM to a file in the output directory."""
    try:
        # Ensure output directory exists
        if not ensure_output_dir(output_dir):
            return None
            
        # Create path within output directory
        file_path = os.path.join(output_dir, filename)
        abs_path = os.path.abspath(file_path)
        
        with open(file_path, 'w') as f:
            f.write(explanation)
        
        print(f"✅ Explanation saved at: {abs_path}")
        
        # Display file existence confirmation
        if os.path.exists(abs_path):
            file_size = os.path.getsize(abs_path)
            print(f"   File exists: {os.path.exists(abs_path)}, Size: {file_size} bytes")
        else:
            print(f"   WARNING: File was written but cannot be found at {abs_path}")
            
        return abs_path
    except Exception as e:
        print(f"❌ Error saving explanation: {str(e)}")
        return None

def save_plot_image(results, filename="chart.png", output_dir="code_output"):
    """Save the plot image from the execution results to a file in the output directory."""
    if not results or len(results) == 0:
        print("No results found to save")
        return None
    
    # Ensure output directory exists
    if not ensure_output_dir(output_dir):
        return None
        
    # Get the first result (typically the plot)
    first_result = results[0]
    
    # Debug: Print all available attributes on the result object
    print(f"DEBUG - Result object attributes: {dir(first_result)}")
    
    # Create path within output directory
    file_path = os.path.join(output_dir, filename)
    abs_path = os.path.abspath(file_path)
    
    # Check for various image formats (png is most common for matplotlib)
    for img_format in ['png', 'jpg', 'jpeg', 'svg', 'pdf']:
        if hasattr(first_result, img_format) and getattr(first_result, img_format):
            image_data = getattr(first_result, img_format)
            try:
                # Write the base64 encoded image to a file
                with open(file_path, 'wb') as f:
                    f.write(base64.b64decode(image_data))
                print(f"✅ Image saved as {abs_path}")
                return abs_path
            except Exception as e:
                print(f"Error saving image: {str(e)}")
                return None
    
    print("No image found in the results")
    return None

def open_image_file(filepath):
    """Open an image file with the default system viewer."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    try:
        system = platform.system()
        if system == 'Darwin':  # macOS
            subprocess.call(['open', filepath])
        elif system == 'Windows':
            os.startfile(filepath)
        elif system == 'Linux':
            subprocess.call(['xdg-open', filepath])
        else:
            print(f"Unsupported platform: {system}")
    except Exception as e:
        print(f"Error opening image file: {str(e)}")

def get_prompt_from_file(file_path):
    """Read a prompt from a file."""
    try:
        with open(file_path, "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading prompt file: {str(e)}")
        return None

def get_default_prompt():
    """Return the default prompt for financial data visualization."""
    # return """Write Python code to fetch financial market data from yfinance and create a visualization that shows a comparison between two major tech stocks over the past year."""
    
    return """
Analyze my financial situation and create a detailed plan to reach a $1.3 million investment goal as quickly as possible, including a breakdown of savings, investment growth, and timelines. I earn a net income of $7,460 per month, and my girlfriend earns a net income of $5,640 per month (based on her $100,000 gross income after taxes). We live in Rock Springs, Wyoming, with a baby. Our monthly expenses are $5,750 when we both work (including $1,000 for my truck, $1,000 for my house, $650 for childcare, and $3,100 for other costs like groceries, utilities, and insurance). If my girlfriend stays home, childcare costs drop to $0, reducing expenses to $5,100, and we lose her income. She wants to stay home with our child but is willing to work for a few years to help reach the goal faster.

Breakdown and Analysis:
Calculate our combined monthly and annual savings when both working and when only I work.
Estimate how long it will take to reach $1.3 million under different scenarios: (a) both working until we reach the goal, (b) my girlfriend working for 2 years then staying home, (c) my girlfriend working for 5 years then staying home, and (d) my girlfriend staying home immediately.
Assume a 10% annual investment return for all scenarios, and assume we start with $0 in investments.
Include the impact of taxes (we file jointly, and Wyoming has no state income tax) and any other relevant factors.
Visualizations for Easier Reading:
Create a line graph showing the growth of our investment portfolio over time for each scenario, with the x-axis as years and the y-axis as portfolio value, highlighting when we reach $1.3 million.
Generate a pie chart showing the breakdown of our monthly expenses when both working ($5,750 total) and when only I work ($5,100 total).
Produce a bar chart comparing the total time to reach $1.3 million across all scenarios.
Include a table summarizing key metrics for each scenario: annual savings, years to goal, and total contributions.
Output Format:
Deliver the analysis as a PDF report with embedded graphs and tables, structured with clear headings for each section (e.g., 'Savings Breakdown,' 'Investment Growth Analysis,' 'Visualizations').
Also provide an interactive dashboard hosted on a public URL where I can explore the data and graphs dynamically.
Additional Information:
Highlight any risks or assumptions (e.g., market volatility, expense changes) and suggest strategies to accelerate reaching the goal (e.g., increasing income, reducing expenses).
Ensure the report is easy to read for someone without a financial background, using simple language and clear explanations."
Step 3: Why This Prompt Works
Specificity: It provides all necessary financial details (incomes, expenses, goal) and clearly defines the scenarios to analyze, reducing ambiguity for Manus.
Task Decomposition: It breaks the task into analysis, visualization, and output formatting, aligning with Manus's ability to handle multi-step workflows.
Visualization Request: Asking for specific graphs (line, pie, bar) and a table ensures the output is visual and easy to digest, leveraging Manus's data visualization capabilities.
Output Format: Requesting a PDF report and an interactive dashboard caters to both static and dynamic reading preferences, making the information accessible.
Context and Assumptions: Clarifying the investment return rate (10%) and starting point ($0) ensures Manus makes consistent calculations.
Step 4: Expected Output from Manus
Based on Manus's capabilities, here's what you can expect:

Detailed Analysis: A report with sections like:
Savings Breakdown: When both working, savings are $14,016 - $5,750 = $8,266/month ($99,192/year). When only you work, savings are $7,460 - $5,100 = $2,360/month ($28,320/year), adjusted to $3,010/month ($36,120/year) without childcare.
Timeline to $1.3 Million: Scenarios showing total years (e.g., 9 years if both work, 12 years if she works 3 years, 16 years if she stays home now).
Visualizations:
A line graph showing portfolio growth over time for each scenario, reaching $1.3 million.
A pie chart breaking down expenses (e.g., 17% truck, 17% house, 11% childcare, 55% other when both work).
A bar chart comparing years to goal across scenarios.
A table summarizing savings and timelines.
Interactive Dashboard: A web-based dashboard where you can hover over graphs to see data points, such as portfolio value at specific years.
Additional Insights: Suggestions like reducing expenses (e.g., paying off the truck loan early) or increasing income (e.g., side hustles) to shorten the timeline.
Step 5: Tips for Optimizing Your Prompt
Start Simple: If the prompt feels too complex, start with a smaller task, like "Analyze my monthly expenses and create a pie chart," then build on the results.
Iterate: If Manus's output isn't exactly what you want, refine your prompt. For example, if the graphs aren't clear, ask for "larger labels and a color-coded legend."
Leverage Multi-Agent Workflows: Encourage Manus to use its sub-agents by framing the task as a multidisciplinary project (e.g., analysis, visualization, and reporting).
Specify Audience: Mention that the report should be easy to read for non-financial experts, ensuring Manus uses simple language.
"""

def open_file(filepath):
    """Open a file with the default system viewer."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    try:
        system = platform.system()
        if system == 'Darwin':  # macOS
            subprocess.call(['open', filepath])
        elif system == 'Windows':
            os.startfile(filepath)
        elif system == 'Linux':
            subprocess.call(['xdg-open', filepath])
        else:
            print(f"Unsupported platform: {system}")
    except Exception as e:
        print(f"Error opening file: {str(e)}")

def save_execution_output(execution, filename="execution_output.txt", output_dir="code_output"):
    """Save execution output (stdout, stderr, error) to a file in the output directory."""
    if not execution:
        print("No execution data to save")
        return None
    
    # Ensure output directory exists
    if not ensure_output_dir(output_dir):
        return None
        
    # Create path within output directory
    file_path = os.path.join(output_dir, filename)
    abs_path = os.path.abspath(file_path)
    
    try:
        with open(file_path, 'w') as f:
            f.write("=== EXECUTION OUTPUT ===\n\n")
            
            # Extract stdout logs
            if hasattr(execution, 'logs') and hasattr(execution.logs, 'stdout'):
                stdout_logs = ''.join(execution.logs.stdout) if execution.logs.stdout else ""
                if stdout_logs:
                    f.write("STDOUT:\n")
                    f.write(stdout_logs)
                    f.write("\n\n")
            
            # Extract stderr logs
            if hasattr(execution, 'logs') and hasattr(execution.logs, 'stderr'):
                stderr_logs = ''.join(execution.logs.stderr) if execution.logs.stderr else ""
                if stderr_logs:
                    f.write("STDERR:\n")
                    f.write(stderr_logs)
                    f.write("\n\n")
            
            # Extract execution error if present
            if hasattr(execution, 'error') and execution.error:
                f.write("ERROR:\n")
                f.write(execution.error)
                f.write("\n\n")
                
        print(f"✅ Execution output saved at: {abs_path}")
        return abs_path
    except Exception as e:
        print(f"❌ Error saving execution output: {str(e)}")
        return None

def fix_output_paths_in_code(code, output_dir="code_output"):
    """Replace hardcoded output file paths in code to use the code_output directory."""
    # Create pattern to find common plot saving calls and file operations
    patterns = [
        (r"plt\.savefig\(['\"](.*?)['\"]", f"plt.savefig('{output_dir}/\\1'"),
        (r"savefig\(['\"](.*?)['\"]", f"savefig('{output_dir}/\\1'"),
        (r"with open\(['\"](.*?)['\"]\s*,\s*['\"](w|wb)['\"]", f"with open('{output_dir}/\\1', '\\2'"),
        (r"pd\.to_csv\(['\"](.*?)['\"]", f"pd.to_csv('{output_dir}/\\1'"),
        (r"fig\.write_image\(['\"](.*?)['\"]", f"fig.write_image('{output_dir}/\\1'"),
    ]
    
    # Apply all patterns
    modified_code = code
    for pattern, replacement in patterns:
        modified_code = re.sub(pattern, replacement, modified_code)
    
    # Add output directory import if it changed
    if modified_code != code:
        # Add code to check and create output directory at the start
        dir_check_code = f"""
# Ensure output directory exists
import os
if not os.path.exists('{output_dir}'):
    os.makedirs('{output_dir}')
"""
        # Find the first import statement
        import_match = re.search(r"^import ", modified_code, re.MULTILINE)
        if import_match:
            # Insert after the end of the import block
            import_block_end = modified_code.find('\n\n', import_match.start())
            if import_block_end == -1:  # If no double newline, find the first non-import line
                lines = modified_code.split('\n')
                line_num = 0
                for i, line in enumerate(lines):
                    if not line.strip().startswith('import ') and not line.strip().startswith('from '):
                        line_num = i
                        break
                if line_num > 0:
                    modified_code = '\n'.join(lines[:line_num]) + '\n' + dir_check_code + '\n'.join(lines[line_num:])
            else:
                modified_code = modified_code[:import_block_end] + '\n' + dir_check_code + modified_code[import_block_end:]
        else:
            # No imports found, add at the beginning
            modified_code = dir_check_code + modified_code
    
    return modified_code

def main():
    try:
        # Parse command line arguments
        args = get_args()
        
        # Print configuration source info
        env_provider = os.getenv("LLM_PROVIDER")
        env_model = os.getenv("LLM_MODEL")
        if env_provider:
            print(f"📋 Using LLM_PROVIDER from .env: {env_provider}")
        if env_model:
            print(f"📋 Using LLM_MODEL from .env: {env_model}")
        
        # Determine the user prompt from arguments or default
        if args.prompt:
            USER_PROMPT = args.prompt
        elif args.prompt_file:
            USER_PROMPT = get_prompt_from_file(args.prompt_file)
            if not USER_PROMPT:
                print("❌ Failed to read prompt from file, using default prompt")
                USER_PROMPT = get_default_prompt()
        else:
            # Check if there's input on stdin
            import sys
            if not sys.stdin.isatty():
                # Read from stdin if available
                USER_PROMPT = sys.stdin.read().strip()
                if not USER_PROMPT:
                    USER_PROMPT = get_default_prompt()
            else:
                USER_PROMPT = get_default_prompt()
                
        print(f"\n{'='*80}\nUSER REQUEST: {USER_PROMPT}\n{'='*80}\n")
        
        # Get available providers
        available_providers = get_available_providers()
        if not available_providers:
            print("❌ No API keys configured for any provider. Please set at least one API key.")
            return
        
        # Use the provider from arguments or first available
        provider = args.provider
        if provider not in available_providers:
            print(f"⚠️ Provider '{provider}' is not available or lacks an API key. Available providers: {', '.join(available_providers)}")
            provider = available_providers[0]
            print(f"🔄 Using {provider.upper()} as the fallback provider.")
        else:
            print(f"🔄 Using {provider.upper()} as the provider. Available providers: {', '.join(available_providers)}")
        
        # Get the LLM client with the specified model (if any)
        llm = get_llm_client(provider, args.model)
        
        # Handle different attribute names for model across providers
        if provider == "anthropic":
            print(f"🔄 Using model: {llm.model}")
        else:
            print(f"🔄 Using model: {llm.model_name}")
        
        # Get formatted system prompt
        SYSTEM_PROMPT = get_formatted_system_prompt()
        
        # Initialize sandbox at the top so we can reuse it for retries
        with Sandbox() as sandbox:
            print("✅ Sandbox initialized successfully")
            
            # Upload the dataset (if it exists)
            if not upload_dataset(sandbox):
                print("❌ Failed to upload dataset")
                return
            
            # Initialize retry counter and keep original prompt
            retry_count = 0
            current_prompt = USER_PROMPT
            
            # Main execution loop with retries
            while True:
                print(f"🔄 Getting model response from {provider.upper()}... (Attempt {retry_count + 1})")
                # Get model response with retries
                response = get_model_response(llm, SYSTEM_PROMPT, current_prompt)
                if not response:
                    print("❌ Failed to get model response after retries")
                    return
                
                print("✅ Got model response")
                
                # Extract explanatory text from the response
                explanation = extract_explanation_text(response)
                
                # Extract code from the response    
                code, code_found = match_code_blocks(response)
                if not code_found:
                    print("❓ No explicit code blocks found in the response")
                    print("📝 The LLM provided an explanation only:")
                    print(f"\n{'='*80}\nLLM EXPLANATION:\n{'='*80}\n{explanation}\n{'='*80}\n")
                    
                    # Save the explanation to a file
                    explanation_file = save_explanation_text(explanation)
                    if explanation_file:
                        # Ask if user wants to open the saved explanation file
                        open_file_input = input("Would you like to open the explanation file? (y/n): ").strip().lower()
                        if open_file_input == 'y':
                            open_file(explanation_file)
                    
                    # Ask user if they want to generate and execute code based on the explanation
                    user_input = input("Would you like to generate code based on this explanation? (y/n): ").strip().lower()
                    if user_input != 'y':
                        print("✅ Exiting without code execution")
                        return
                    
                    # If user wants to generate code, ask the LLM to convert the explanation to code
                    print("🔄 Asking LLM to generate code based on the explanation...")
                    code_gen_prompt = f"Convert the following explanation into executable Python code. Only return the Python code, no additional text. Make sure to include all necessary imports and create clear visualizations:\n\n{explanation}"
                    code_response = get_model_response(llm, SYSTEM_PROMPT, code_gen_prompt)
                    
                    # Extract code from the new response
                    code, code_found = match_code_blocks(code_response)
                    if not code_found:
                        print("❌ Failed to generate executable code from the explanation")
                        return
                
                print(f"\n{'='*80}\nEXTRACTED CODE (Attempt {retry_count + 1}):\n{'='*80}\n{code}\n{'='*80}\n")
                
                # Fix date issues in code - ensure no datetime.now() is used
                if 'datetime.now()' in code:
                    print("⚠️ Found datetime.now() in code, replacing with fixed dates")
                    # Replace datetime.now() with current date string
                    code = code.replace("datetime.now()", f"# Using fixed date\ndatetime.strptime('{CURRENT_DATE}', '%Y-%m-%d')")
                    code = code.replace("datetime.now().year", str(CURRENT_YEAR))
                    code = code.replace("datetime.now().strftime", f"'{CURRENT_DATE}'.split('-')[0] + ")
                    
                    # Replace year-to-date calculations
                    code = code.replace(
                        "start_date = f'{datetime.now().year}-01-01'", 
                        f"start_date = '{YTD_START}'"
                    )
                    code = code.replace(
                        "end_date = datetime.now().strftime('%Y-%m-%d')", 
                        f"end_date = '{CURRENT_DATE}'"
                    )
                    print("📝 Applied date patches to fix future date issues")
                
                # Fix output paths to save files to code_output directory
                code = fix_output_paths_in_code(code)
                print("📝 Ensured all output files will be saved to the code_output directory")
                
                print("🔄 Executing code...")
                # Execute the code with error handling
                results, execution = code_interpret(sandbox, code)
                
                # Check if execution was successful - consider it a success if there's output even without results
                execution_output = ""
                if hasattr(execution, 'logs') and hasattr(execution.logs, 'stdout'):
                    execution_output = ''.join(execution.logs.stdout) if execution.logs.stdout else ""
                
                if results or (execution_output and not execution.error):
                    print("✅ Code executed successfully")
                    
                    # Save execution output
                    execution_output_file = save_execution_output(execution)
                    
                    # If we have logs but no results, we should treat it as success
                    if not results and execution_output:
                        print("⚠️ No results object returned, but code executed with output:")
                        print(execution_output)
                        
                    break  # Exit the retry loop on success
                
                # Handle failed execution
                print(f"❌ Code execution failed (Attempt {retry_count + 1})")
                
                # Check if we've reached the maximum retry count
                if retry_count >= MAX_RETRY_ATTEMPTS:
                    print(f"❌ Maximum retry attempts ({MAX_RETRY_ATTEMPTS}) reached, giving up")
                    return
                
                # Extract logs for the retry prompt
                execution_logs = extract_execution_logs(execution)
                
                # Prepare the enhanced prompt with error logs for retry
                retry_prompt = current_prompt + ERROR_CORRECTION_PROMPT.format(
                    error_logs=execution_logs,
                    ytd_start=YTD_START
                )
                current_prompt = retry_prompt
                
                # Increment retry counter
                retry_count += 1
                print(f"🔄 Retrying code generation with error logs (Attempt {retry_count + 1})")
            
            # Code executed successfully, process the results
            if results:
                print(f"\n{'='*80}\nEXECUTION RESULTS:\n{'='*80}")
                for result in results:
                    if hasattr(result, 'type') and result.type:
                        print(f"Result type: {result.type}")
                    if hasattr(result, 'text') and result.text:
                        print(f"Text output: {result.text}")
                    if hasattr(result, 'path') and result.path:
                        print(f"File path: {result.path}")
                print(f"{'='*80}\n")
                
                # Save the plot as an image file
                image_path = save_plot_image(results)
                if image_path:
                    abs_path = os.path.abspath(image_path)
                    print(f"You can open the saved image at: {abs_path}")
                    # Automatically open the image file
                    open_image_file(abs_path)
            
    except Exception as e:
        print(f"❌ [Main ERROR] {str(e)}")

if __name__ == "__main__":
    main()