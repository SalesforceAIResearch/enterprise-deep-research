import os
import time
import anthropic
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime
from langsmith import traceable

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Import LangChain and provider-specific packages
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.messages import ChatMessage
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_google_vertexai import ChatVertexAI



# API keys for different providers
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SFR_GATEWAY_API_KEY = os.getenv("SFR_GATEWAY_API_KEY")
SAMBNOVA_API_KEY = os.getenv("SAMBNOVA_API_KEY")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# Token limit configurations for different providers
# OpenAI token limits
OPENAI_MAX_TOKENS = 30000

# Anthropic token limits
ANTHROPIC_CLAUDE_4_MAX_TOKENS = 50000    # Claude 4 Sonnet (new flagship model)
ANTHROPIC_CLAUDE_37_MAX_TOKENS = 30000
ANTHROPIC_CLAUDE_35_MAX_TOKENS = 30000
ANTHROPIC_CLAUDE_3_MAX_TOKENS = 4096
ANTHROPIC_THINKING_BUDGET_TOKENS = 4000
ANTHROPIC_ASYNC_MAX_TOKENS = 8192

# Google Gemini token limits
GOOGLE_MAX_OUTPUT_TOKENS = 30000

# Get the current date in various formats for the system prompt
CURRENT_DATE = datetime.now().strftime('%Y-%m-%d')
CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month
CURRENT_DAY = datetime.now().day
ONE_YEAR_AGO = datetime.now().replace(year=datetime.now().year - 1).strftime('%Y-%m-%d')
YTD_START = f"{CURRENT_YEAR}-01-01"

# Define model configurations for each provider
MODEL_CONFIGS = {
    # Groq models
    "groq": {
        "available_models": [
            'deepseek-r1-distill-llama-70b',
            'llama-3.3-70b-versatile',
            'llama3-70b-8192'
        ],
        "default_model": "deepseek-r1-distill-llama-70b",
        "requires_api_key": GROQ_API_KEY,
    },
    # OpenAI models
    "openai": {
        "available_models": [
            "gpt-4o",
            "o4-mini",          # Latest compact reasoning model (April 2025)
            "o4-mini-high",     # High-performance variant for paid tier users
            "o3-mini",
            "o3-mini-reasoning"  # This will use o3-mini with high reasoning effort
        ],
        "default_model": "o4-mini",  # Latest and most cost-efficient reasoning model
        "requires_api_key": OPENAI_API_KEY,
    },
    # Anthropic models
    "anthropic": {
        "available_models": [
            "claude-sonnet-4",                  # New Claude 4 Sonnet (flagship model)
            "claude-sonnet-4-thinking",         # Claude 4 Sonnet with extended thinking mode
            "claude-3-7-sonnet",                # Standard mode Claude 3.7 Sonnet
            "claude-3-7-sonnet-thinking",       # Claude 3.7 Sonnet with extended thinking mode enabled
            "claude-3-5-sonnet"                 # Legacy Claude 3 Sonnet
        ],
        "default_model": "claude-sonnet-4",
        "requires_api_key": ANTHROPIC_API_KEY,
    },
    # Salesforce Research Gateway models
    "sfrgateway": {
        "available_models": ["deepseek-v3-0324"],
        "default_model": "deepseek-v3-0324",
        "requires_api_key": SFR_GATEWAY_API_KEY,
    },
    # SambaNova models
    "sambnova": {
        "available_models": ["DeepSeek-V3-0324"],
        "default_model": "DeepSeek-V3-0324",
        "requires_api_key": SAMBNOVA_API_KEY,
    },
    # Google Vertex AI models
    "google": {
        "available_models": [
            "gemini-2.5-pro",
            "gemini-1.5-pro-latest",  # Recommended Pro model
            "gemini-1.5-flash-latest",  # Recommended Flash model
            "gemini-pro",  # Older Pro model
        ],
        "default_model": "gemini-2.5-pro",
        "requires_api_key": GOOGLE_CLOUD_PROJECT,
    },
}

# Base system prompt template - will be formatted with current date information
SYSTEM_PROMPT_TEMPLATE = """
<intro>
You excel at the following tasks:
1. Information gathering, fact-checking, and documentation
2. Data processing, analysis, and visualization
3. Writing multi-chapter articles and in-depth research reports
4. Creating websites, applications, and tools
5. Using programming to solve various problems beyond development
6. Various tasks that can be accomplished using computers and the internet
7. IMPORTANT: The current date is {current_date}. Always use this as your reference instead of datetime.now().
</intro>

<date_information>
Current date: {current_date}
Current year: {current_year}
Current month: {current_month}
Current day: {current_day}
One year ago: {one_year_ago}
Year-to-date start: {ytd_start}
</date_information>

<requirements>
When writing code, your code MUST:
1. Start with installing necessary packages (e.g., '!pip install matplotlib pandas')
2. Include robust error handling for data retrieval and processing
3. Print sample data to validate successful retrieval
4. Properly handle data structures based on the returned format:
   - For multi-level dataframes: Use appropriate indexing like df['Close'] or df.loc[:, ('Price', 'Close')]
   - For single-level dataframes: Access columns directly
5. If asked to create visualizations with professional styling including:
   - Clear title and axis labels
   - Grid lines for readability
   - Appropriate date formatting on x-axis using matplotlib.dates
   - Legend when plotting multiple series
6. Include data validation to check for:
   - Dataset sizes and date ranges
   - Missing values (NaN) and their handling
   - Data types and any necessary conversions
7. Implement appropriate data transformations like:
   - Normalizing prices to a common baseline
   - Calculating moving averages or other indicators
   - Computing ratios or correlations between assets
8. IMPORTANT: When fetching date-sensitive data:
   - DO NOT use datetime.now() in your code
   - Instead, use these fixed dates: current="{current_date}", year_start="{ytd_start}", year_ago="{one_year_ago}"
</requirements>
"""

# Error correction prompt addition when code execution fails
ERROR_CORRECTION_PROMPT = """
<error_correction>
The previous code failed to execute properly. I'm providing the error logs below.
Please fix the code to address these issues and ensure it runs correctly:

ERROR LOGS:
{error_logs}

Common issues to check:
1. Date handling issues - Use explicit date ranges (e.g., '{ytd_start}' instead of datetime.now())
2. Data structure validation - Verify the expected structure of returned data
3. Library compatibility - Ensure all functions used are available in the imported libraries
4. Error handling - Add more robust try/except blocks
</error_correction>
"""


# Custom OpenAI client for models that don't support temperature
class SimpleOpenAIClient:
    """Simple OpenAI client that doesn't use LangChain for models that don't support temperature."""

    def __init__(
        self, model_name: str, api_key: str, max_tokens: int = OPENAI_MAX_TOKENS
    ):
        """Initialize the simple OpenAI client.

        Args:
            model_name: The name of the model to use
            api_key: The OpenAI API key
            max_tokens: The maximum number of tokens to generate
        """
        self.model_name = model_name  # Public attribute for compatibility
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client = openai.OpenAI(api_key=api_key)

    @traceable
    def invoke(self, messages, config=None):
        """Invoke the model with the given messages without using temperature.

        Args:
            messages: List of LangChain message objects
            config: Optional RunConfig that may contain trace information

        Returns:
            A ChatMessage with the model's response
        """
        try:
            # Convert LangChain messages to OpenAI format
            openai_messages = []
            for msg in messages:
                role = msg.type
                if role == "human":
                    role = "user"
                elif role == "system":
                    role = "system"
                elif role == "ai":
                    role = "assistant"
                openai_messages.append({"role": role, "content": msg.content})

            # Extract trace info from config if available
            metadata = {}
            if config:
                # Extract langsmith tracing info if available
                if "callbacks" in config and hasattr(
                    config["callbacks"], "get_trace_id"
                ):
                    trace_id = config["callbacks"].get_trace_id()
                    if trace_id:
                        metadata["ls_trace_id"] = trace_id

                # Extract any other metadata from config
                if "metadata" in config:
                    metadata.update(config["metadata"])

            # For o3-mini, o4-mini and o1 models, use max_completion_tokens instead of max_tokens
            token_param = {}
            if (
                self.model_name.startswith("o3")
                or self.model_name.startswith("o4")
                or self.model_name.startswith("o1")
            ):
                token_param = {"max_completion_tokens": self._max_tokens}
                print(f"Using max_completion_tokens for {self.model_name}")
            else:
                token_param = {"max_tokens": self._max_tokens}

            # Call the OpenAI API directly without temperature
            print(f"Calling OpenAI API with model {self.model_name} (no temperature)")
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                **token_param,  # Use the appropriate token parameter
                # Include metadata for LangSmith tracing if available
                extra_headers=(
                    {"X-LangSmith-Trace-Id": metadata.get("ls_trace_id", "")}
                    if "ls_trace_id" in metadata
                    else None
                ),
            )

            # Return the response content
            return response.choices[0].message.content
        except Exception as e:
            print(f"[SimpleOpenAIClient ERROR] {str(e)}")
            raise


# Custom OpenAI client for models that support reasoning_effort parameter
class ReasoningEffortOpenAIClient(SimpleOpenAIClient):
    """OpenAI client with reasoning_effort parameter for enhanced models like o3-mini-high."""

    @traceable
    def invoke(self, messages, config=None):
        """Invoke the model with the given messages using high reasoning effort.

        Args:
            messages: List of LangChain message objects
            config: Optional RunConfig that may contain trace information

        Returns:
            A string with the model's response
        """
        try:
            # Convert LangChain messages to OpenAI format
            openai_messages = []
            for msg in messages:
                role = msg.type
                if role == "human":
                    role = "user"
                elif role == "system":
                    role = "system"
                elif role == "ai":
                    role = "assistant"
                openai_messages.append({"role": role, "content": msg.content})

            # Extract trace info from config if available
            metadata = {}
            if config:
                # Extract langsmith tracing info if available
                if "callbacks" in config and hasattr(
                    config["callbacks"], "get_trace_id"
                ):
                    trace_id = config["callbacks"].get_trace_id()
                    if trace_id:
                        metadata["ls_trace_id"] = trace_id

                # Extract any other metadata from config
                if "metadata" in config:
                    metadata.update(config["metadata"])

            # For o3-mini, o4-mini and o1 models, use max_completion_tokens instead of max_tokens
            token_param = {}
            if (
                self.model_name.startswith("o3")
                or self.model_name.startswith("o4")
                or self.model_name.startswith("o1")
            ):
                token_param = {"max_completion_tokens": self._max_tokens}
                print(f"Using max_completion_tokens for {self.model_name}")
            else:
                token_param = {"max_tokens": self._max_tokens}

            # Call the OpenAI API directly with high reasoning effort
            print(
                f"Calling OpenAI API with model {self.model_name} (high reasoning effort)"
            )
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                reasoning_effort="high",  # Add high reasoning effort parameter
                **token_param,  # Use the appropriate token parameter
                # Include metadata for LangSmith tracing if available
                extra_headers=(
                    {"X-LangSmith-Trace-Id": metadata.get("ls_trace_id", "")}
                    if "ls_trace_id" in metadata
                    else None
                ),
            )

            # Return the response content
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ReasoningEffortOpenAIClient ERROR] {str(e)}")
            raise


# Custom Anthropic client for Claude 3.x models with proper model ID mapping and thinking mode
class Claude3ExtendedClient:
    """Anthropic client that supports all Claude 3.x and 4.x models with proper model ID mapping.

    This client is optimized for Claude models including:
    - claude-sonnet-4: Latest Claude 4 Sonnet model (flagship)
    - claude-sonnet-4-thinking: Claude 4 Sonnet with extended thinking capabilities
    - claude-3-7-sonnet: Standard Claude 3.7 Sonnet model
    - claude-3-7-sonnet-thinking: Claude 3.7 Sonnet with extended thinking capabilities
    - claude-3-5-sonnet: Claude 3.5 Sonnet model

    The client handles:
    1. Mapping simplified model names to their full model IDs
    2. Extended thinking capabilities via API parameters (for -thinking variants)
    3. Robust response content extraction
    4. Appropriate token limits based on model capabilities
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        max_tokens: int = ANTHROPIC_THINKING_BUDGET_TOKENS,
    ):
        """Initialize the custom Claude 3.x client.

        Args:
            model_name: The name of the Claude model to use
            api_key: The Anthropic API key
            max_tokens: The maximum number of tokens for the response
        """
        self.model = model_name  # This is needed for compatibility with the main code
        self.api_key = api_key
        self.thinking_tokens = max_tokens
        self._client = anthropic.Anthropic(
            api_key=api_key,
        )

        # Map simplified model names to actual Anthropic model IDs
        # These are the versioned IDs that must be used with the API
        self._model_map = {
            # Claude 4 models
            "claude-sonnet-4": "claude-sonnet-4-20250514",
            "claude-sonnet-4-thinking": "claude-sonnet-4-20250514",  # Same model ID, but with thinking enabled
            # Claude 3.7 models
            "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
            "claude-3-7-sonnet-thinking": "claude-3-7-sonnet-20250219",  # Same model ID, but with thinking enabled
            # Claude 3.5 models
            "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",  # New version
            "claude-3-5-sonnet-old": "claude-3-5-sonnet-20240620",  # Old version
            "claude-3-5-haiku": "claude-3-5-haiku-20241022",
            # Claude 3.0 models
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
        }

    def _extract_content_from_blocks(self, response):
        """Enhanced method to extract content from response blocks, handling various formats."""
        try:
            # Initialize variables for extraction attempts
            extracted_content = ""
            thinking_content = ""
            extraction_attempts = 0
            extraction_methods = []

            # Get the content blocks from the response
            if hasattr(response, "content") and response.content:
                content_blocks = response.content

                # For debug
                if hasattr(response, "completion"):
                    print("Response has completion attribute")
                if hasattr(response, "message"):
                    print("Response has message attribute")

                # First pass: Identify all content blocks by type
                for i, block in enumerate(content_blocks):
                    extraction_attempts += 1

                    # Log block type if available
                    if hasattr(block, "type"):
                        print(f"Block {i} type: {block.type}")

                    # Extract content based on block type or attributes
                    if (
                        hasattr(block, "type")
                        and block.type == "thinking"
                        and hasattr(block, "thinking")
                    ):
                        thinking_content = block.thinking
                        extraction_methods.append(
                            f"Found thinking content in block.thinking (attempt {extraction_attempts})"
                        )
                    elif (
                        hasattr(block, "type")
                        and block.type == "text"
                        and hasattr(block, "text")
                    ):
                        extracted_content = block.text
                        extraction_methods.append(
                            f"Found content in block.text (attempt {extraction_attempts})"
                        )
                    # Fallbacks for different attribute patterns
                    elif hasattr(block, "text") and block.text:
                        extracted_content = block.text
                        extraction_methods.append(
                            f"Found content in block.text (attempt {extraction_attempts})"
                        )
                    elif hasattr(block, "thinking") and block.thinking:
                        thinking_content = block.thinking
                        extraction_methods.append(
                            f"Found thinking content in block.thinking (attempt {extraction_attempts})"
                        )
                    elif hasattr(block, "value") and block.value:
                        extracted_content = block.value
                        extraction_methods.append(
                            f"Found content in block.value (attempt {extraction_attempts})"
                        )

                # If text extraction failed, try to convert the first block to string
                if not extracted_content and not thinking_content and content_blocks:
                    extraction_attempts += 1
                    try:
                        # Try accessing dictionary-like structure
                        if hasattr(content_blocks[0], "items"):
                            for key, value in content_blocks[0].items():
                                if isinstance(value, str) and value:
                                    extracted_content = value
                                    extraction_methods.append(
                                        f"Found content in dict-like access: {key} (attempt {extraction_attempts})"
                                    )
                                    break

                        # If still no content, try string representation
                        if not extracted_content and not thinking_content:
                            extracted_content = str(content_blocks[0])
                            extraction_methods.append(
                                f"Converted first block to string (attempt {extraction_attempts})"
                            )
                    except Exception as e:
                        extraction_methods.append(
                            f"Error in block conversion: {str(e)}"
                        )

                # Debug info
                print("\nContent extraction details:")
                print(f"- Attempts: {extraction_attempts}")
                for method in extraction_methods:
                    print(f"- {method}")

                # For thinking mode, we might want to combine or choose between thinking and final content
                if thinking_content and extracted_content:
                    combined = f"[THINKING]\n{thinking_content}\n\n[RESPONSE]\n{extracted_content}"
                    print("- Combining thinking and response content")
                    return combined
                elif thinking_content:
                    print("- Using thinking content only (no regular content found)")
                    return thinking_content
                elif extracted_content:
                    print("- Using regular content (no thinking content found)")
                    return extracted_content

            # Fallback: try to get completion from response
            if hasattr(response, "completion"):
                return response.completion

            # Last resort: convert entire response to string
            return str(response)

        except Exception as e:
            print(f"Error extracting content: {str(e)}")
            # Return empty string or some error message as fallback
            return f"Error extracting content: {str(e)}"

    @traceable
    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def invoke(self, messages, config=None, prompts=None, system=None):
        """Send the messages to the Claude model, including thinking parameter.

        Args:
            messages: List of message objects
            config: Optional RunConfig that may contain trace information
            prompts: Optional prompts (for backward compatibility)
            system: Optional system message (for backward compatibility)

        Returns:
            String response from Claude
        """
        try:
            # Extract the system message if it is present in the messages
            system_content = None
            filtered_messages = []

            # Handle LangChain message objects
            if hasattr(messages[0], "type"):  # LangChain message objects
                for msg in messages:
                    if msg.type == "system":
                        system_content = msg.content
                    else:
                        # Convert non-system messages to dictionary format
                        role = msg.type
                        if role == "human":
                            role = "user"
                        elif role == "ai":
                            role = "assistant"
                        filtered_messages.append({"role": role, "content": msg.content})
            else:  # Dictionary format messages
                for message in messages:
                    if message["role"] == "system":
                        system_content = message["content"]
                    else:
                        filtered_messages.append(message)

            # If no system message found but system parameter provided, use that
            if not system_content and system:
                system_content = system

            # Get the actual model ID to use
            actual_model_id = self._model_map.get(self.model, self.model)

            # Setup tokens for thinking mode
            thinking_tokens = self.thinking_tokens
            max_tokens = ANTHROPIC_CLAUDE_37_MAX_TOKENS  # Default fallback
            temperature = 0.4  # Default

            # Set appropriate max_tokens based on model
            if "claude-sonnet-4" in self.model.lower():
                max_tokens = (
                    ANTHROPIC_CLAUDE_4_MAX_TOKENS  # Claude 4 Sonnet increased limit
                )
                temperature = 0.4  # Default for Claude 4
            elif "claude-3-7-sonnet" in self.model.lower():
                max_tokens = (
                    ANTHROPIC_CLAUDE_37_MAX_TOKENS  # Claude 3.7 Sonnet increased limit
                )
                temperature = 0.4  # Default
            elif "claude-3-5-sonnet" in self.model.lower():
                max_tokens = ANTHROPIC_CLAUDE_35_MAX_TOKENS  # Claude 3.5 Sonnet limit
                temperature = 0.4  # Default
            else:
                max_tokens = ANTHROPIC_CLAUDE_37_MAX_TOKENS  # Safe fallback
                temperature = 0.4  # Default

            # For thinking mode, ensure max_tokens > thinking.budget_tokens and temperature =0.3
            if "thinking" in self.model.lower():
                temperature = 0.3

            # Print debug info
            print(f"\nSending request to Claude")
            print(f"Model ID: {actual_model_id} (from {self.model})")
            if "thinking" in self.model.lower():
                print(f"Thinking tokens: {thinking_tokens}")
            print(f"Max tokens: {max_tokens}")
            print(f"Temperature: {temperature}")
            if "claude-sonnet-4" in self.model.lower():
                print(
                    f"Using Claude 4 Sonnet - Latest flagship model with enhanced capabilities"
                )

            # Create the API call parameters
            api_params = {
                "model": actual_model_id,
                "messages": filtered_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            # Add system content if available
            if system_content:
                api_params["system"] = system_content

            # Add thinking parameter if in thinking mode
            if "thinking" in self.model.lower():
                print(f"Enabling thinking mode")
                api_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_tokens,
                }

            # Extract trace info from config if available for LangSmith integration
            if config:
                metadata = {}
                # Extract langsmith tracing info if available
                if "callbacks" in config and hasattr(
                    config["callbacks"], "get_trace_id"
                ):
                    trace_id = config["callbacks"].get_trace_id()
                    if trace_id:
                        metadata["ls_trace_id"] = trace_id
                        # Add trace ID to Anthropic request headers if available
                        api_params["extra_headers"] = {"X-LangSmith-Trace-Id": trace_id}

            # Enable streaming for all requests to prevent timeouts on long operations
            api_params["stream"] = True

            # Make the API call with streaming
            print("\nSending streaming request to Claude")
            stream = self._client.messages.create(**api_params)

            # Collect content from the stream
            content_chunks = []
            for chunk in stream:
                if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                    content_chunks.append(chunk.delta.text)
                elif hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                    for content_block in chunk.message.content:
                        if content_block.type == "text":
                            content_chunks.append(content_block.text)

            # Combine all chunks into a single response
            response_text = "".join(content_chunks)
            print("\nStreaming response received from Claude")

            # For streaming, we've already extracted the content
            # Debug info
            if response_text:
                content_preview = (
                    response_text[:500] + "..."
                    if len(response_text) > 500
                    else response_text
                )
                print(f"\nExtracted content (preview): {content_preview}")
            else:
                print("\nNo content extracted from streaming response")

            return response_text

        except Exception as e:
            error_msg = f"Error invoking Claude: {str(e)}"
            print(error_msg)

            # If we have additional error details, print them
            if hasattr(e, "response") and hasattr(e.response, "json"):
                try:
                    error_details = e.response.json()
                    print(f"Error details: {error_details}")
                except:
                    pass

            return error_msg


# Custom client for Salesforce Research Gateway API
class SalesforceResearchClient:
    """Client for Salesforce Research Gateway API.

    This client supports both streaming and non-streaming modes for models
    available through the Salesforce Research Gateway.
    """

    def __init__(
        self, model_name: str, api_key: str, max_tokens: int = OPENAI_MAX_TOKENS
    ):
        """Initialize the Salesforce Research Gateway client.

        Args:
            model_name: The name of the model to use (e.g., 'deepseek-v3-0324')
            api_key: The Salesforce Research Gateway API key
            max_tokens: The maximum number of tokens to generate (not used by this API)
        """
        self.model = model_name  # This is needed for compatibility with the main code
        self._api_key = api_key
        self._base_url = "https://gateway.salesforceresearch.ai"

        # Required for compatibility with LangChain
        self.model_name = model_name

    @traceable
    def invoke(self, messages, config=None, stream=True):
        """Invoke the model with the given messages.

        Args:
            messages: List of LangChain message objects
            config: Optional RunConfig that may contain trace information
            stream: Whether to stream the response

        Returns:
            A ChatMessage with the model's response or an iterator if streaming
        """
        import requests
        import json

        # Convert LangChain messages to the format expected by Salesforce Research Gateway
        sf_messages = []
        for msg in messages:
            role = msg.type
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            sf_messages.append({"role": role, "content": msg.content})

        # Extract trace info from config if available
        metadata = {}
        if config:
            # Extract langsmith tracing info if available
            if "callbacks" in config and hasattr(config["callbacks"], "get_trace_id"):
                trace_id = config["callbacks"].get_trace_id()
                if trace_id:
                    metadata["ls_trace_id"] = trace_id

            # Extract any other metadata from config
            if "metadata" in config:
                metadata.update(config["metadata"])

        # Prepare the API URL
        url = f"{self._base_url}/{self.model_name}/process"

        # Prepare headers
        headers = {
            "accept": "application/json",
            "X-Api-Key": self._api_key,
            "Content-Type": "application/json",
        }

        # Prepare the payload
        payload = {"messages": sf_messages, "stream": stream}

        try:
            # Make the API request
            print(
                f"Calling Salesforce Research Gateway API with model {self.model_name} (stream={stream})"
            )
            response = requests.post(url, headers=headers, json=payload, stream=stream)
            response.raise_for_status()

            if not stream:
                # Process the non-streaming response
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    if "delta" in result["choices"][0]:
                        content = result["choices"][0]["delta"]["content"]
                    else:
                        content = result["choices"][0]["message"]["content"]

                    return ChatMessage(content=content, role="ai")
                else:
                    print(
                        "Unexpected response format from Salesforce Research Gateway API"
                    )
                    print(result)
                    return ChatMessage(
                        content="Error: Unexpected response format", role="ai"
                    )
            else:
                # For streaming mode, collect all chunks and build the full response
                full_content = ""
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode("utf-8")
                        # Check if line starts with 'data: '
                        if line_text.startswith("data:"):
                            try:
                                # Extract the JSON data after 'data: '
                                json_str = line_text[5:].strip()
                                if json_str:
                                    chunk_data = json.loads(json_str)
                                    if "result" in chunk_data:
                                        chunk_content = chunk_data["result"]
                                        full_content += chunk_content
                                        # Print progress (optional)
                                        print(".", end="", flush=True)
                            except Exception as e:
                                print(
                                    f"\nError parsing streaming response chunk: {str(e)}"
                                )

                print("\nStreaming complete")  # New line after progress dots
                return ChatMessage(content=full_content, role="ai")

        except Exception as e:
            print(f"Error calling Salesforce Research Gateway API: {str(e)}")
            return ChatMessage(content=f"Error: {str(e)}", role="ai")

    def stream(self, messages, config=None):
        """Stream the response from the model.

        Args:
            messages: List of LangChain message objects
            config: Optional RunConfig that may contain trace information

        Yields:
            A single ChatMessage with the full content when streaming is complete
        """
        import requests
        import json

        # Convert LangChain messages to the format expected by Salesforce Research Gateway
        sf_messages = []
        for msg in messages:
            role = msg.type
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            sf_messages.append({"role": role, "content": msg.content})

        # Prepare the API URL
        url = f"{self._base_url}/{self.model_name}/process"

        # Prepare headers
        headers = {
            "accept": "application/json",
            "X-Api-Key": self._api_key,
            "Content-Type": "application/json",
        }

        # Prepare the payload - always stream=True for this method
        payload = {"messages": sf_messages, "stream": True}

        try:
            # Make the API request
            print(f"Streaming response from {self.model_name}...")
            response = requests.post(url, headers=headers, json=payload, stream=True)
            response.raise_for_status()

            # Stream the response in chunks
            current_content = ""
            for line in response.iter_lines():
                if line:
                    line_text = line.decode("utf-8")
                    # Skip the [DONE] line at the end
                    if line_text == "data: [DONE]":
                        continue

                    # Check if line starts with 'data: '
                    if line_text.startswith("data:"):
                        try:
                            # Extract the JSON data after 'data: '
                            json_str = line_text[5:].strip()
                            if json_str:
                                chunk_data = json.loads(json_str)
                                if (
                                    "choices" in chunk_data
                                    and len(chunk_data["choices"]) > 0
                                ):
                                    delta = chunk_data["choices"][0]["delta"]
                                    if "content" in delta:
                                        chunk_content = delta["content"]
                                        current_content += chunk_content
                                        # Yield each updated chunk
                                        yield ChatMessage(
                                            content=current_content, role="ai"
                                        )
                        except Exception as e:
                            print(f"Error parsing streaming response chunk: {str(e)}")

        except Exception as e:
            print(f"Error streaming from Salesforce Research Gateway API: {str(e)}")
            yield ChatMessage(content=f"Error: {str(e)}", role="ai")


# Custom client for SambaNova API
class SambNovaClient:
    """Client for SambaNova AI API.

    This client supports both streaming and non-streaming modes for models
    available through the SambaNova API.
    """

    def __init__(
        self, model_name: str, api_key: str, max_tokens: int = OPENAI_MAX_TOKENS
    ):
        """Initialize the SambaNova client.

        Args:
            model_name: The name of the model to use (e.g., 'DeepSeek-V3-0324')
            api_key: The SambaNova API key (Bearer token)
            max_tokens: The maximum number of tokens to generate (not used by this API)
        """
        self.model = model_name  # This is needed for compatibility with the main code
        self._api_key = api_key
        self._base_url = "https://api.sambanova.ai/v1/chat/completions"

        # Required for compatibility with LangChain
        self.model_name = model_name

    @traceable
    def invoke(self, messages, config=None, stream=True):
        """Invoke the model with the given messages.

        Args:
            messages: List of LangChain message objects
            config: Optional RunConfig that may contain trace information
            stream: Whether to stream the response

        Returns:
            A ChatMessage with the model's response or an iterator if streaming
        """
        import requests
        import json

        # Convert LangChain messages to the format expected by SambaNova API
        sn_messages = []
        system_message = None

        for msg in messages:
            role = msg.type
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            elif role == "system":
                system_message = {"role": "system", "content": msg.content}
                continue  # Skip adding to regular messages, we'll handle separately

            sn_messages.append({"role": role, "content": msg.content})

        # If we have a system message, add it to the beginning
        if system_message:
            sn_messages.insert(0, system_message)
        # If we don't have any system message, add a default one
        elif not any(msg.get("role") == "system" for msg in sn_messages):
            sn_messages.insert(
                0, {"role": "system", "content": "You are a helpful assistant"}
            )

        # Extract trace info from config if available
        metadata = {}
        if config:
            # Extract langsmith tracing info if available
            if "callbacks" in config and hasattr(config["callbacks"], "get_trace_id"):
                trace_id = config["callbacks"].get_trace_id()
                if trace_id:
                    metadata["ls_trace_id"] = trace_id

            # Extract any other metadata from config
            if "metadata" in config:
                metadata.update(config["metadata"])

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Prepare the payload
        payload = {"model": self.model_name, "messages": sn_messages, "stream": stream}

        try:
            # Make the API request
            print(
                f"Calling SambaNova API with model {self.model_name} (stream={stream})"
            )
            response = requests.post(
                self._base_url, headers=headers, json=payload, stream=stream
            )
            response.raise_for_status()

            if not stream:
                # Process the non-streaming response
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    return ChatMessage(content=content, role="ai")
                else:
                    print("Unexpected response format from SambaNova API")
                    print(result)
                    return ChatMessage(
                        content="Error: Unexpected response format", role="ai"
                    )
            else:
                # For streaming mode, collect all chunks and build the full response
                full_content = ""
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode("utf-8")
                        # Skip the [DONE] line at the end
                        if line_text == "data: [DONE]":
                            continue

                        # Check if line starts with 'data: '
                        if line_text.startswith("data:"):
                            try:
                                # Extract the JSON data after 'data: '
                                json_str = line_text[5:].strip()
                                if json_str:
                                    chunk_data = json.loads(json_str)
                                    if (
                                        "choices" in chunk_data
                                        and len(chunk_data["choices"]) > 0
                                    ):
                                        delta = chunk_data["choices"][0]["delta"]
                                        if "content" in delta:
                                            chunk_content = delta["content"]
                                            full_content += chunk_content
                                            # Print progress (optional)
                                            print(".", end="", flush=True)
                            except Exception as e:
                                print(
                                    f"\nError parsing streaming response chunk: {str(e)}"
                                )

                print("\nStreaming complete")  # New line after progress dots
                return ChatMessage(content=full_content, role="ai")

        except Exception as e:
            print(f"Error calling SambaNova API: {str(e)}")
            return ChatMessage(content=f"Error: {str(e)}", role="ai")

    def stream(self, messages, config=None):
        """Stream the response from the model.

        Args:
            messages: List of LangChain message objects
            config: Optional RunConfig that may contain trace information

        Yields:
            Chunks of the model's response
        """
        import requests
        import json

        # Convert LangChain messages to the format expected by SambaNova API
        sn_messages = []
        system_message = None

        for msg in messages:
            role = msg.type
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            elif role == "system":
                system_message = {"role": "system", "content": msg.content}
                continue  # Skip adding to regular messages, we'll handle separately

            sn_messages.append({"role": role, "content": msg.content})

        # If we have a system message, add it to the beginning
        if system_message:
            sn_messages.insert(0, system_message)
        # If we don't have any system message, add a default one
        elif not any(msg.get("role") == "system" for msg in sn_messages):
            sn_messages.insert(
                0, {"role": "system", "content": "You are a helpful assistant"}
            )

        # Prepare the API URL
        url = self._base_url

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Prepare the payload - always stream=True for this method
        payload = {"model": self.model_name, "messages": sn_messages, "stream": True}

        try:
            # Make the API request
            print(f"Streaming response from {self.model_name}...")
            response = requests.post(url, headers=headers, json=payload, stream=True)
            response.raise_for_status()

            # Stream the response in chunks
            current_content = ""
            for line in response.iter_lines():
                if line:
                    line_text = line.decode("utf-8")
                    # Skip the [DONE] line at the end
                    if line_text == "data: [DONE]":
                        continue

                    # Check if line starts with 'data: '
                    if line_text.startswith("data:"):
                        try:
                            # Extract the JSON data after 'data: '
                            json_str = line_text[5:].strip()
                            if json_str:
                                chunk_data = json.loads(json_str)
                                if (
                                    "choices" in chunk_data
                                    and len(chunk_data["choices"]) > 0
                                ):
                                    delta = chunk_data["choices"][0]["delta"]
                                    if "content" in delta:
                                        chunk_content = delta["content"]
                                        current_content += chunk_content
                                        # Yield each updated chunk
                                        yield ChatMessage(
                                            content=current_content, role="ai"
                                        )
                        except Exception as e:
                            print(f"Error parsing streaming response chunk: {str(e)}")

        except Exception as e:
            print(f"Error streaming from SambaNova API: {str(e)}")
            yield ChatMessage(content=f"Error: {str(e)}", role="ai")


def get_available_providers():
    """Returns a list of available providers based on configured API keys."""
    available_providers = []
    for provider, config in MODEL_CONFIGS.items():
        if config.get("requires_api_key"):  # Use .get() for safety
            available_providers.append(provider)
    return available_providers


def get_llm_client(provider, model_name=None):
    """
    Get the appropriate LLM client based on provider and model name.
    Uses standard LangChain clients to ensure compatibility with features like bind_tools.

    Args:
        provider: The provider name ('groq', 'openai', 'anthropic', 'sfrgateway', 'sambnova')
        model_name: The model name (optional, uses default if not provided)

    Returns:
        A synchronous LangChain chat model client for the specified provider
    """
    # First, check if we have the API key for the provider
    if provider == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in environment")
        api_key = GROQ_API_KEY
        if not model_name:
            model_name = MODEL_CONFIGS["groq"]["default_model"]
        print(f"Using ChatGroq for {model_name}")
        return ChatGroq(
            api_key=api_key,
            model_name=model_name,
        )
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment")
        api_key = OPENAI_API_KEY
        if not model_name:
            model_name = MODEL_CONFIGS["openai"]["default_model"]

        # For o4-mini-high, use ReasoningEffortOpenAIClient with base o4-mini model
        if model_name == "o4-mini-high":
            print(
                f"Using ReasoningEffortOpenAIClient for {model_name} (enhanced reasoning)"
            )
            return ReasoningEffortOpenAIClient(
                model_name="o4-mini",  # Use base model name for API
                api_key=api_key,
                max_tokens=OPENAI_MAX_TOKENS,
            )
        # For o3-mini-reasoning, use ReasoningEffortOpenAIClient
        elif model_name == "o3-mini-reasoning":
            print(
                f"Using ReasoningEffortOpenAIClient for o3-mini with high reasoning effort"
            )
            return ReasoningEffortOpenAIClient(
                model_name="o3-mini",  # Use base model name
                api_key=api_key,
                max_tokens=OPENAI_MAX_TOKENS,
            )
        else:
            # Use standard ChatOpenAI for other models
            print(f"Using standard ChatOpenAI for requested model: {model_name}")
            return ChatOpenAI(
                model_name=model_name.replace(
                    "-reasoning", ""
                ),  # Use base model name if reasoning specified
                api_key=api_key,
                max_tokens=OPENAI_MAX_TOKENS,  # Using variable instead of hardcoded value
                streaming=False,
            )

    # For Salesforce Research Gateway models
    elif provider == "sfrgateway":
        if not SFR_GATEWAY_API_KEY:
            raise ValueError("SFR_GATEWAY_API_KEY is not set in environment")
        if not model_name:
            model_name = MODEL_CONFIGS["sfrgateway"]["default_model"]
        print(f"Using SalesforceResearchClient for {model_name}")
        return SalesforceResearchClient(
            model_name=model_name, api_key=SFR_GATEWAY_API_KEY
        )

    # For SambaNova models
    elif provider == "sambnova":
        if not SAMBNOVA_API_KEY:
            raise ValueError("SAMBNOVA_API_KEY is not set in environment")
        if not model_name:
            model_name = MODEL_CONFIGS["sambnova"]["default_model"]
        print(f"Using SambNovaClient for {model_name}")
        return SambNovaClient(model_name=model_name, api_key=SAMBNOVA_API_KEY)

    elif provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment")
        if not model_name:
            model_name = MODEL_CONFIGS["anthropic"]["default_model"]

        # For "thinking" mode variants and Claude Sonnet 4, use custom client as it handles API params
        # bind_tools should still work if ChatAnthropic base class supports it
        if "thinking" in model_name or "claude-sonnet-4" in model_name:
            print(
                f"Using custom Claude3ExtendedClient for {model_name} (thinking mode or Claude 4)"
            )
            return Claude3ExtendedClient(
                model_name=model_name,
                api_key=ANTHROPIC_API_KEY,
                max_tokens=ANTHROPIC_THINKING_BUDGET_TOKENS,  # Using variable instead of hardcoded value
            )
        else:
            # Use standard ChatAnthropic for others
            print(f"Using Langchain ChatAnthropic for {model_name}")
            model_mapping = {
                "claude-sonnet-4": "claude-sonnet-4-20250514",
                "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
                "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
                "claude-3-haiku": "claude-3-haiku-20240307",
                "claude-3-opus": "claude-3-opus-20240229",
                "claude-3-sonnet": "claude-3-sonnet-20240229",
            }

            anthropic_model = model_mapping.get(model_name, model_name)

            # Set appropriate max_tokens based on model
            if "claude-sonnet-4" in anthropic_model:
                max_tokens = (
                    ANTHROPIC_CLAUDE_4_MAX_TOKENS  # Claude 4 Sonnet increased limit
                )
            elif "claude-3-7-sonnet" in anthropic_model:
                max_tokens = (
                    ANTHROPIC_CLAUDE_37_MAX_TOKENS  # Claude 3.7 Sonnet increased limit
                )
            elif "claude-3-5-sonnet" in anthropic_model:
                max_tokens = ANTHROPIC_ASYNC_MAX_TOKENS  # Claude 3.5 Sonnet limit
            elif "claude-3-opus" in anthropic_model:
                max_tokens = ANTHROPIC_CLAUDE_3_MAX_TOKENS  # Claude 3 Opus limit
            elif "claude-3-sonnet" in anthropic_model:
                max_tokens = ANTHROPIC_CLAUDE_3_MAX_TOKENS  # Claude 3 Sonnet limit
            elif "claude-3-haiku" in anthropic_model:
                max_tokens = ANTHROPIC_CLAUDE_3_MAX_TOKENS  # Claude 3 Haiku limit
            else:
                max_tokens = ANTHROPIC_CLAUDE_3_MAX_TOKENS  # Safe default

            return ChatAnthropic(
                model=anthropic_model, api_key=ANTHROPIC_API_KEY, max_tokens=max_tokens
            )
    elif provider == "google":

        if not GOOGLE_CLOUD_PROJECT:
            raise ValueError("GOOGLE_CLOUD_PROJECT is not set in environment")
        if not model_name:
            model_name = MODEL_CONFIGS["google"]["default_model"]
        print(f"Using ChatVertexAI for {model_name}")
        return ChatVertexAI(
            model_name=model_name,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
            convert_system_message_to_human=True,  # Recommended for Gemini
            max_output_tokens=GOOGLE_MAX_OUTPUT_TOKENS,  # Using variable instead of hardcoded value
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# Helper function to get an async client for the given provider (Uncommented and Updated)
async def get_async_llm_client(provider, model_name=None):
    """
    Get an asynchronous LLM client for the given provider.
    Uses standard LangChain async clients where available.

    Args:
        provider: The provider name ('openai', 'anthropic', 'groq')
        model_name: The model name (optional, uses default if not provided)

    Returns:
        An async LangChain chat model client for the specified provider
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"[get_async_llm_client] Requested provider: {provider}, model: {model_name or 'default'}"
    )

    # Get the default model from MODEL_CONFIGS if not specified
    if not model_name and provider in MODEL_CONFIGS:
        model_name = MODEL_CONFIGS[provider]["default_model"]
        logger.info(
            f"[get_async_llm_client] Using default model for {provider}: {model_name}"
        )

    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment")
        api_key = OPENAI_API_KEY

        # Always use standard ChatOpenAI async client
        # Strip -reasoning suffix if present, and handle o4-mini-high
        if model_name == "o4-mini-high":
            effective_model_name = "o4-mini"  # Use base model name for API
        else:
            effective_model_name = model_name.replace("-reasoning", "")
        logger.info(
            f"[get_async_llm_client] Creating async ChatOpenAI client with model {effective_model_name}"
        )

        return ChatOpenAI(
            model_name=effective_model_name,
            api_key=api_key,
            max_tokens=OPENAI_MAX_TOKENS,  # Using variable instead of hardcoded value
            streaming=False,  # Keep false unless streaming needed in async context
        )
    elif provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment")
        api_key = ANTHROPIC_API_KEY

        # Map model names for compatibility
        model_mapping = {
            "claude-sonnet-4": "claude-sonnet-4-20250514",
            "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
        }

        if not model_name:
            model_name = MODEL_CONFIGS["anthropic"]["default_model"]

        # Check if this is Claude Sonnet 4 - warn about potential compatibility issues
        if "claude-sonnet-4" in model_name:
            logger.warning(
                f"[get_async_llm_client] Claude Sonnet 4 async support via standard LangChain ChatAnthropic may have compatibility issues. Consider using sync client."
            )

        anthropic_model = model_mapping.get(model_name, model_name)

        # Set appropriate max_tokens based on model
        if "claude-sonnet-4" in anthropic_model:
            max_tokens = (
                ANTHROPIC_CLAUDE_4_MAX_TOKENS  # Claude 4 Sonnet increased limit
            )
        elif "claude-3-7-sonnet" in anthropic_model:
            max_tokens = (
                ANTHROPIC_CLAUDE_37_MAX_TOKENS  # Claude 3.7 Sonnet increased limit
            )
        elif "claude-3-5-sonnet" in anthropic_model:
            max_tokens = ANTHROPIC_ASYNC_MAX_TOKENS  # Claude 3.5 Sonnet limit
        elif "claude-3-opus" in anthropic_model:
            max_tokens = ANTHROPIC_CLAUDE_3_MAX_TOKENS  # Claude 3 Opus limit
        elif "claude-3-sonnet" in anthropic_model:
            max_tokens = ANTHROPIC_CLAUDE_3_MAX_TOKENS  # Claude 3 Sonnet limit
        elif "claude-3-haiku" in anthropic_model:
            max_tokens = ANTHROPIC_CLAUDE_3_MAX_TOKENS  # Claude 3 Haiku limit
        else:
            max_tokens = ANTHROPIC_CLAUDE_3_MAX_TOKENS  # Safe default

        logger.info(
            f"[get_async_llm_client] Creating async ChatAnthropic client with model {anthropic_model}"
        )
        return ChatAnthropic(
            model=anthropic_model, api_key=api_key, max_tokens=max_tokens
        )
    elif provider == "groq" and GROQ_API_KEY:
        api_key = GROQ_API_KEY
        if not model_name:
            model_name = MODEL_CONFIGS["groq"]["default_model"]

        logger.info(
            f"[get_async_llm_client] Creating async ChatGroq client with model {model_name}"
        )
        return ChatGroq(
            api_key=api_key,
            model_name=model_name,
        )
    elif provider == "google":
        if not GOOGLE_CLOUD_PROJECT:
            raise ValueError("GOOGLE_CLOUD_PROJECT is not set in environment")
        if not model_name:
            model_name = MODEL_CONFIGS["google"]["default_model"]
        logger.info(
            f"[get_async_llm_client] Creating async ChatVertexAI client with model {model_name}"
        )
        return ChatVertexAI(
            model_name=model_name,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
            convert_system_message_to_human=True,  # Recommended for Gemini
            max_output_tokens=GOOGLE_MAX_OUTPUT_TOKENS,  # Using variable instead of hardcoded value
        )
    else:
        # For providers that don't have standard async clients via Langchain yet
        supported_providers = ["openai", "anthropic"]
        if GROQ_API_KEY:
            supported_providers.append("groq")
        if GOOGLE_CLOUD_PROJECT:  # Add google to supported list
            supported_providers.append("google")

        error_msg = f"Async client not supported or API key missing for provider: {provider}. Supported with keys: {', '.join(supported_providers)}."
        logger.error(f"[get_async_llm_client] {error_msg}")
        raise ValueError(error_msg)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_model_response(llm, system_prompt: str, user_prompt: str, config=None):
    """
    Get a response from an LLM using LangChain or SimpleOpenAIClient.

    Args:
        llm: The chat model client (LangChain model or SimpleOpenAIClient)
        system_prompt: The system prompt to use
        user_prompt: The user prompt to use
        config: Optional config object that may contain LangSmith trace information

    Returns:
        The model's response as a string
    """
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Handle different model types
        if isinstance(llm, SimpleOpenAIClient):
            model_name = llm.model_name
        else:
            # Handle different model attribute names across providers
            model_name = getattr(llm, "model_name", None)
            if model_name is None:
                model_name = getattr(llm, "model", "unknown model")

        print(f"🔄 Sending messages to {model_name}...")

        # Invoke the model with config for tracing
        response = llm.invoke(messages, config=config)

        # SimpleOpenAIClient returns a string directly
        if isinstance(response, str):
            return response
        # LangChain models return an object with content attribute
        else:
            return response.content
    except Exception as e:
        print(f"[Model API ERROR] {str(e)}")
        raise  # Re-raise the exception to trigger retry


# Format system prompt with current date information
def get_formatted_system_prompt():
    """
    Format the system prompt template with current date information.

    Returns:
        The formatted system prompt string
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        current_date=CURRENT_DATE,
        current_year=CURRENT_YEAR,
        current_month=CURRENT_MONTH,
        current_day=CURRENT_DAY,
        one_year_ago=ONE_YEAR_AGO,
        ytd_start=YTD_START,
    )