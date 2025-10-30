import logging
from typing import Dict, Any, Optional, List, Union, Set
from enum import Enum
import asyncio
import time
import datetime
import base64
import os
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.schema import HumanMessage
import json
import urllib.parse  # Added for extract_domain

# Simple LLM client interface - adjust based on your actual implementation
from llm_clients import get_llm_client, get_async_llm_client, get_model_response

logger = logging.getLogger(__name__)


# Helper function to extract domain from URL (copied from src/utils.py)
def extract_domain(url: str) -> str:
    """Extract the domain from a URL.

    Args:
        url (str): The URL to extract the domain from

    Returns:
        str: The extracted domain
    """
    if not url:
        return ""
    try:
        # Ensure the URL has a scheme for urlparse to work correctly
        if not url.startswith(("http://", "https://")):
            # Check if it's a common pattern like example.com/path before prepending https
            if "." in url.split("/")[0]:  # It looks like a domain
                url = "https://" + url
            else:  # It might be a relative path or malformed, return as is or a placeholder
                return url

        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc

        # Remove www. prefix if present
        if domain.startswith("www."):
            domain = domain[4:]

        return domain
    except Exception as e:
        logger.warning(f"Error extracting domain from URL '{url}': {str(e)}")
        # Fallback to simple extraction if parsing fails
        try:
            if "//" in url:
                domain_part = url.split("//")[1].split("/")[0]
                if domain_part.startswith("www."):
                    return domain_part[4:]
                return domain_part
            else:
                domain_part = url.split("/")[0]
                if domain_part.startswith("www."):
                    return domain_part[4:]
                return domain_part
        except Exception:  # Final fallback
            return url  # Return original url if all parsing fails


class ActivityType(Enum):
    """
    Enumeration of simplified activity types for UI updates.
    These represent the main steps that the user should see in the UI.
    """

    DECOMPOSE_RESEARCH_TOPIC = "decompose_research_topic"
    LINKEDIN_SEARCH = "linkedin_search"
    ACADEMIC_SEARCH = "academic_search"
    GENERAL_DEEP_SEARCH = "general_deep_search"
    TEXT2SQL_SEARCH = "text2sql_search"
    GENERATE_VISUALIZATION = "generate_visualization"
    GENERATE_INITIAL_REPORT = "generate_initial_report"
    IDENTIFY_GAPS = "identify_gaps"
    FINALIZE_REPORT = "finalize_report"
    NODE_START = "node_start"
    EXECUTE_AGENT_PLAN = "execute_agent_plan"


class ActivityManager:
    """
    Activity management for UI updates with dynamic content based on state data.
    """

    # Track previous activities to avoid repetition
    _previous_activities: Dict[str, Set[str]] = {}
    # Track iteration counts for each activity type
    _iteration_counts: Dict[str, int] = {}
    # Cache lock for thread safety
    _lock = asyncio.Lock()
    # Last time activity cache was cleared
    _last_cache_clear = time.time()
    # Track sent image filenames to avoid resending
    _sent_image_filenames: Set[str] = set()
    # Track sent domain URLs to avoid resending web links
    _sent_domain_urls: Set[str] = set()

    @staticmethod
    def is_important_activity(
        event_type: str, event_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Determine if the event represents an important activity that should be shown in the UI.
        """
        # Basic logging
        logger.info(f"Checking if event is important: {event_type}")

        # First check if this is a node start event
        if event_type == "node_start" and event_data and "node_name" in event_data:
            node_name = event_data.get("node_name", "")

            # Check if this is one of our important nodes
            important_nodes = [
                # Core research steps
                "decompose_research_topic",
                "multi_agents_network",
                "generate_report",
                "reflect_on_report",
                "finalize_report",
                # Search variations
                "linkedin_search",
                "academic_search",
                "general_deep_search",
                "deep_search_parallel",
                # Other important nodes
                "generate_visualization",
                "identify_gaps",
                "route_after_multi_agents",
            ]

            is_important = node_name in important_nodes
            logger.info(f"Node {node_name} important? {is_important}")
            return is_important

        # Check for knowledge gap events
        elif event_type == "knowledge_gap" or event_type == "knowledge_gap_identified":
            return True

        # Check for search_sources_found events
        elif event_type == "search_sources_found":
            # More inclusive - show all search source events as they're important
            return True

        # Check for visualization events
        elif event_type == "visualization_generated":
            return True

        # Check for search result events
        elif event_type in [
            "linkedin_search_results",
            "academic_search_results",
            "general_deep_search_results",
        ]:
            return True

        # Default case - not important
        return False

    @staticmethod
    def _extract_context(event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant context from event data based on the event type.
        Instead of manually extracting specific fields, we now pass the entire event data
        to the context window and let the LLM figure out what's relevant.

        Returns:
            A dictionary of context information useful for activity generation
        """
        logger.info(f"===== EXTRACTING CONTEXT FOR EVENT TYPE: {event_type} =====")
        logger.info(f"Event data keys: {list(event_data.keys())}")

        # Create a context dictionary that includes the entire event_data
        context = {
            # Include the entire event data for the LLM to process
            "event_data": event_data,
            # Include the event type for quick reference
            "event_type": event_type,
            # Extract a few essential fields for backward compatibility
            "research_topic": event_data.get("research_topic", "the topic"),
        }

        # Add iteration count if available - commonly needed
        if "research_loop_count" in event_data:
            context["iteration"] = event_data["research_loop_count"]

        # Extract node name for node_start events - commonly referenced
        if event_type == "node_start":
            context["node_name"] = event_data.get("node_name", "")

        # For identify_gaps activity type with Reflect On Report, include Knowledge Gap and Search Query
        if (
            event_type == "node_start"
            and event_data.get("node_name", "").lower() == "reflect_on_report"
        ):
            # Extract state object which contains Knowledge Gap and Search Query
            state = event_data.get("state", {})

            # Include Knowledge Gap if available
            if "knowledge_gap" in state:
                context["knowledge_gap"] = state["knowledge_gap"]
                logger.info(f"Added knowledge_gap to context: {state['knowledge_gap']}")

            # Include Search Query if available
            if "search_query" in state:
                context["search_query"] = state["search_query"]
                logger.info(f"Added search_query to context: {state['search_query']}")

        # Log that we're using the simplified approach
        logger.info(
            f"Using simplified context extraction. Full event data passed to LLM for interpretation."
        )

        return context

    @staticmethod
    def _build_prompt(activity_type: ActivityType, context: Dict[str, Any]) -> str:
        """
        Build a minimal prompt for the LLM to generate a dynamic activity description.
        This version works with the simplified context structure that includes the full event data.

        Args:
            activity_type: The type of activity to generate content for
            context: Context information including full event_data

        Returns:
            A prompt string for the LLM
        """
        logger.info(f"===== BUILDING PROMPT FOR ACTIVITY TYPE: {activity_type} =====")
        # Log the raw event_data for inspection
        logger.info(f"Available context keys: {list(context.keys())}")

        # Add more detailed logging for search operation results
        event_data = context.get("event_data", {})
        input_data = event_data.get("input", {})
        research_results = input_data.get("research_results", {})
        domains = input_data.get("domains", [])
        sources = input_data.get("formatted_sources", [])

        # Check if we have search results
        if research_results:
            logger.info(f"===== DETAILED SEARCH RESULTS LOGGING =====")
            logger.info(f"Research results keys: {list(research_results.keys())}")

            # Extract and log subtopic results
            subtopic_results = research_results.get("subtopic_results", [])
            if subtopic_results:
                search_tools_used = []
                all_domains = set()
                source_counts = {}

                logger.info(f"Found {len(subtopic_results)} subtopic results")

                # Process each subtopic result
                for i, result in enumerate(subtopic_results):
                    if isinstance(result, dict):
                        subtopic = result.get("subtopic", {})
                        search_result = result.get("search_result", {})

                        # Extract information
                        tool_used = search_result.get("tool_used", "unknown")
                        search_tools_used.append(tool_used)

                        # Log domains
                        domains_list = search_result.get("domains", [])
                        if domains_list:
                            all_domains.update(domains_list)

                        # Count sources
                        sources_list = search_result.get("formatted_sources", [])
                        source_counts[tool_used] = source_counts.get(
                            tool_used, 0
                        ) + len(sources_list)

                        # Log first subtopic details as an example
                        if i == 0:
                            logger.info(
                                f"Example subtopic: {subtopic.get('name', 'unknown')}"
                            )
                            logger.info(
                                f"Example query: {subtopic.get('query', 'unknown')}"
                            )
                            if sources_list and len(sources_list) > 0:
                                logger.info(f"Example source: {sources_list[0]}")

        # Build the prompt in the previous rich, instructive format, using all available domains
        research_topic = context.get("research_topic", "unknown")
        node_name = context.get("node_name", event_data.get("node_name", ""))
        # Build prompt using a Python literal string (triple-quoted f-string) for readability
        context_lines = []
        if node_name:
            context_lines.append(
                f"The current activity is: {node_name.replace('_', ' ').title()}."
            )

        # Count domains that are actually included in the input data
        prompt_domains_count = 0
        if domains:
            prompt_domains_count = len(domains)
            logger.info(
                f"Counted {prompt_domains_count} domains directly from input data for prompt."
            )

        # Add domains count to context if there are any
        if prompt_domains_count > 0:
            context_lines.append(
                f"Processing {prompt_domains_count} websites/sources from input data."
            )

        if domains:
            # Limit displayed domain URLs to avoid overly long prompts
            max_display_domains = 5
            domain_urls_list = [
                d.get("url", d.get("domain", ""))
                for d in domains
                if d.get("url") or d.get("domain")
            ]
            displayed_urls = ", ".join(domain_urls_list[:max_display_domains])
            if len(domain_urls_list) > max_display_domains:
                displayed_urls += ", ..."
            if displayed_urls:
                context_lines.append(f"Websites referenced include: {displayed_urls}.")

        # Optionally, still include sample sources if available and distinct from domains
        if sources:
            # Filter sources to avoid simple domain names already listed
            filtered_sources = [
                s
                for s in sources
                if not any(d.get("domain") in s for d in domains if d.get("domain"))
            ]
            if filtered_sources:
                preview_sources = ", ".join(filtered_sources[:3])
                context_lines.append(f"Sample sources: {preview_sources}")

        # Include Knowledge Gap and Search Query for identify_gaps activity type with Reflect On Report
        if (
            activity_type == ActivityType.IDENTIFY_GAPS
            and node_name.lower() == "reflect_on_report"
        ):
            # Add Knowledge Gap if available
            if "knowledge_gap" in context:
                context_lines.append(f"Knowledge Gap: {context['knowledge_gap']}")
                logger.info(
                    f"Including knowledge_gap in prompt context: {context['knowledge_gap']}"
                )

            # Add Search Query if available
            if "search_query" in context:
                context_lines.append(f"Search Query: {context['search_query']}")
                logger.info(
                    f"Including search_query in prompt context: {context['search_query']}"
                )

        # Include database information for text2sql activities
        if activity_type == ActivityType.TEXT2SQL_SEARCH:
            # Check for database info in event_data
            database_info = event_data.get(
                "database_info", context.get("database_info")
            )
            if database_info:
                if isinstance(database_info, list) and len(database_info) > 0:
                    db_name = database_info[0].get("filename", "database")
                    context_lines.append(f"Querying database: {db_name}")
                    logger.info(f"Including database_info in prompt context: {db_name}")
                elif isinstance(database_info, dict):
                    db_name = database_info.get("filename", "database")
                    context_lines.append(f"Querying database: {db_name}")
                    logger.info(f"Including database_info in prompt context: {db_name}")

            # Check for SQL query in event_data
            sql_query = event_data.get("sql_query", context.get("sql_query"))
            if sql_query:
                context_lines.append(f"Executing SQL query to analyze data")
                logger.info(f"Including SQL query context")

        context_block = "\n".join(context_lines)

        prompt = f"""
Generate a brief, specific, and natural-sounding research activity update about "{research_topic}".
Your response should be 1-2 sentences MAX describing what's happening right now in the research process.
Focus only on the current activity and be specific about what's being done.

Activity type: {activity_type.value}

## Context

{context_block}

## Instructions
Keep your response concise and specific to the current research step.
IMPORTANT: Do not use first person perspective. Describe the activity as if reporting on what the research system is doing.
Just give the activity description with no prefixes or explanations.
Include specific details like source names, domains, or gaps when available.

## Additional Information
You have full access to all context information provided above. Use it to generate a detailed, specific, and informative research activity message.

EXTREMELY IMPORTANT: Use simple, easy-to-understand language that a 10-year-old could understand. DO NOT use technical or internal system terms like "node_start", "Multi Agents Network", "agent plan", "Node_Start activity", or phrases like "The Multi Agents Network is executing the agent plan". 

Instead, use natural, conversational language that describes the research process in plain terms, like:
- "Starting research on [topic] by searching for basic information."
- "Looking through search results to find key details about [topic]."
- "Uncovering important facts about [topic] from several websites."
- "Exploring deeper information about [specific aspect] of [topic]."
- "Putting together all the findings into a complete report about [topic]."
- "Analyzing data from the uploaded database to find insights about [topic]." (for text2sql activities)
- "Querying the database to extract relevant information about [topic]." (for text2sql activities)

When describing search activities, mention the specific domains being searched. When describing database query activities (text2sql_search), mention that data is being extracted from an uploaded database. When describing report generation, mention how many sources are being used, but always in plain, simple language.
"""

        logger.info(f"Generated prompt: {prompt}")
        return prompt

    @staticmethod
    async def _track_activity(activity_type: str, activity_message: str) -> None:
        """
        Track an activity to avoid repetition.

        Args:
            activity_type: The type of activity being tracked
            activity_message: The activity message being tracked
        """
        async with ActivityManager._lock:
            # Initialize sets if needed
            if activity_type not in ActivityManager._previous_activities:
                ActivityManager._previous_activities[activity_type] = set()

            # Add this activity to the tracked set
            ActivityManager._previous_activities[activity_type].add(activity_message)

            # Increment the iteration count
            if activity_type not in ActivityManager._iteration_counts:
                ActivityManager._iteration_counts[activity_type] = 0
            ActivityManager._iteration_counts[activity_type] += 1

            # Periodically clear old activities and image tracking (keep last 30 minutes)
            current_time = time.time()
            if current_time - ActivityManager._last_cache_clear > 1800:  # 30 minutes
                ActivityManager._previous_activities.clear()
                ActivityManager._sent_image_filenames.clear()
                ActivityManager._sent_domain_urls.clear()
                if hasattr(ActivityManager, "_processed_batches"):
                    ActivityManager._processed_batches.clear()
                ActivityManager._last_cache_clear = current_time
        logger.info(
            "Cleared activity cache, image tracking, domain tracking, and batch tracking after 30 minutes of inactivity"
        )

    @staticmethod
    def associate_code_with_visualizations_static(
        code_snippets, visualization_results, event_id=None
    ):
        """
        Match code snippets that generate visualizations with their output.
        Returns updated code snippets with visualization property where applicable,
        and a list of standalone visualizations.

        Args:
            code_snippets: List of code snippet dictionaries
            visualization_results: List of visualization dictionaries (images)
            event_id: Optional event ID for logging purposes

        Returns:
            Tuple: (updated_list_of_code_snippets, list_of_standalone_visualizations)
        """
        log_prefix = f"[EVENT-{event_id}] " if event_id else "[AssociateViz] "
        logger.info(f"{log_prefix}Starting code-visualization association process")
        logger.debug(
            f"{log_prefix}Code snippets count: {len(code_snippets) if code_snippets else 0}, "
            + f"Visualizations count: {len(visualization_results) if visualization_results else 0}"
        )

        if not code_snippets:
            logger.info(
                f"{log_prefix}No code snippets to process. Returning all visualizations as standalone."
            )
            return [], visualization_results if visualization_results else []

        if not visualization_results:
            logger.info(
                f"{log_prefix}No visualization results to associate. Returning snippets as-is."
            )
            return code_snippets if code_snippets else [], []

        snippets_with_visuals = []
        processed_vis_indices = set()
        association_count = 0

        for snippet_idx, snippet in enumerate(code_snippets):
            snippet_id = snippet.get("id", f"snippet_{snippet_idx}")

            if not isinstance(snippet, dict):
                logger.warning(
                    f"{log_prefix}Skipping non-dictionary snippet: {type(snippet)}"
                )
                snippets_with_visuals.append(snippet)
                continue

            if "code" not in snippet:
                logger.warning(
                    f"{log_prefix}Skipping snippet without code: {snippet_id}"
                )
                snippets_with_visuals.append(snippet)
                continue

            code_content = snippet.get("code", "")
            vis_indicators = [
                "matplotlib",
                "plt.",
                "seaborn",
                "sns.",
                "visualize",
                "chart",
                "plot(",
                "figure",
                "savefig",
            ]

            vis_score = 0
            matched_indicators = []

            for indicator in vis_indicators:
                if indicator.lower() in code_content.lower():
                    vis_score += 1
                    matched_indicators.append(indicator)

            is_vis_generator = vis_score > 0

            if is_vis_generator:
                logger.debug(
                    f"{log_prefix}Snippet {snippet_id} likely generates visualization. "
                    + f"Score: {vis_score}, Indicators: {matched_indicators}"
                )
            else:
                logger.debug(
                    f"{log_prefix}Snippet {snippet_id} unlikely to generate visualization"
                )

            if not is_vis_generator:
                snippets_with_visuals.append(snippet)
                continue

            found_match = False
            for idx, vis_item in enumerate(visualization_results):
                vis_id = vis_item.get("id", f"vis_{idx}")

                if idx in processed_vis_indices:
                    logger.debug(
                        f"{log_prefix}Skipping already processed visualization: {vis_id}"
                    )
                    continue

                description_match = False
                if (
                    "description" in vis_item
                    and "purpose" in snippet
                    and vis_item["description"]
                    and snippet["purpose"]
                    and any(
                        word in vis_item["description"].lower()
                        for word in snippet["purpose"].lower().split()
                    )
                ):
                    description_match = True
                    logger.debug(
                        f"{log_prefix}Found description match between snippet {snippet_id} and vis {vis_id}"
                    )

                processed_vis_indices.add(idx)
                association_count += 1
                logger.info(
                    f"{log_prefix}Associating visualization {vis_id} with code snippet {snippet_id}"
                    + (" based on description match" if description_match else "")
                )

                vis_data = {}
                if "src" in vis_item:
                    vis_data["src"] = vis_item["src"]
                    src_info = (
                        "base64 data"
                        if vis_item["src"].startswith("data:image")
                        else "file path"
                    )
                    src_length = (
                        len(vis_item["src"])
                        if vis_item["src"].startswith("data:image")
                        else "N/A"
                    )
                    logger.debug(
                        f"{log_prefix}Using src from visualization: {vis_id} ({src_info}, length: {src_length})"
                    )
                elif "data" in vis_item:
                    img_format = vis_item.get("format", "png")
                    img_data_b64 = vis_item["data"]
                    vis_data["src"] = f"data:image/{img_format};base64,{img_data_b64}"
                    vis_data["data"] = img_data_b64
                    vis_data["format"] = img_format
                    logger.debug(
                        f"{log_prefix}Created data URI for visualization: {vis_id} "
                        + f"(format: {img_format}, data length: {len(img_data_b64) if img_data_b64 else 0})"
                    )

                if "description" in vis_item:
                    vis_data["description"] = vis_item["description"]
                    logger.debug(
                        f"{log_prefix}Added description to visualization: {vis_id}"
                    )

                updated_snippet = snippet.copy()
                updated_snippet["visualization"] = vis_data
                snippets_with_visuals.append(updated_snippet)

                found_match = True
                break

            if not found_match:
                logger.info(
                    f"{log_prefix}No matching visualization found for snippet {snippet_id}"
                )
                snippets_with_visuals.append(snippet)

        standalone_visualizations = []
        if visualization_results:
            for idx, vis_item in enumerate(visualization_results):
                if idx not in processed_vis_indices:
                    standalone_visualizations.append(vis_item)

        logger.info(
            f"{log_prefix}Association complete: {association_count} code snippets associated."
        )
        logger.info(
            f"{log_prefix}{len(standalone_visualizations)} visualizations remain standalone."
        )

        return snippets_with_visuals, standalone_visualizations

    @staticmethod
    @retry(
        stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3)
    )
    async def _generate_dynamic_message(
        activity_type: ActivityType, context: Dict[str, Any]
    ) -> str:
        """
        Check if an activity message is too similar to a previously tracked one.

        Args:
            activity_type: The type of activity to check
            activity_message: The activity message to check

        Returns:
            True if the activity is too similar to a previous one
        """
        if activity_type not in ActivityManager._previous_activities:
            return False

        return activity_message in ActivityManager._previous_activities[activity_type]

    @staticmethod
    def _get_iteration_count(activity_type: str) -> int:
        """Get the current iteration count for an activity type."""
        return ActivityManager._iteration_counts.get(activity_type, 0)

    @staticmethod
    @retry(
        stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3)
    )
    async def _generate_dynamic_message(
        activity_type: ActivityType, context: Dict[str, Any]
    ) -> str:
        """
        Generate a dynamic activity message using the LLM.

        Args:
            activity_type: The type of activity to generate content for
            context: Context information for the activity, now includes full event_data

        Returns:
            A dynamic activity message, or None if generation failed
        """
        try:
            logger.info(
                f"===== GENERATING DYNAMIC MESSAGE FOR ACTIVITY TYPE: {activity_type} ====="
            )

            # Log key context data available for this activity
            logger.info(
                f"Context for dynamic message: research_topic='{context.get('research_topic', 'unknown')}'"
            )

            # Log the entire context structure for debugging
            event_data = context.get("event_data", {})
            research_results = event_data.get("input", {}).get("research_results", {})

            # Detect if event comes after search operations
            subtopic_results = research_results.get("subtopic_results", [])
            search_operations_completed = []
            for result in subtopic_results:
                if isinstance(result, dict) and "search_result" in result:
                    search_result = result.get("search_result", {})
                    tool_used = search_result.get("tool_used", "")
                    if tool_used:
                        search_operations_completed.append(tool_used)

            if search_operations_completed:
                logger.info(
                    f"SEARCH OPERATIONS COMPLETED: {search_operations_completed}"
                )
                logger.info(f"SUBTOPIC COUNT: {len(subtopic_results)}")
                logger.info(f"DETECTED POST-SEARCH EVENT for {activity_type}")

                # Detailed logging of search results structure
                logger.info(
                    f"SEARCH RESULTS STRUCTURE:\n"
                    + f"- Has research_results: {bool(research_results)}\n"
                    + f"- Subtopic results count: {len(subtopic_results)}\n"
                    + f"- Available top-level keys in event_data: {list(event_data.keys())}\n"
                    + f"- Available keys in event_data['input']: {list(event_data.get('input', {}).keys())}"
                )

                # Log first search result as example (truncated to avoid massive logs)
                if subtopic_results:
                    first_result = subtopic_results[0]
                    first_subtopic = first_result.get("subtopic", {})
                    logger.info(
                        f"EXAMPLE SUBTOPIC: {first_subtopic.get('name', 'unknown')}"
                    )
                    logger.info(
                        f"EXAMPLE SUBTOPIC QUERY: {first_subtopic.get('query', 'unknown')}"
                    )

                    # Check sources and domains in this result
                    first_search_result = first_result.get("search_result", {})
                    sources_count = len(
                        first_search_result.get("formatted_sources", [])
                    )
                    domains = first_search_result.get("domains", [])
                    logger.info(
                        f"EXAMPLE RESULT - Sources: {sources_count}, Domains: {domains}"
                    )
            else:
                # For non-search events, just log key structure
                logger.info(f"REGULAR EVENT (not after search) for {activity_type}")
                logger.info(f"EVENT_DATA KEYS: {list(event_data.keys())}")
                logger.info(f"INPUT KEYS: {list(event_data.get('input', {}).keys())}")

            # Continue with original code

            # Log that we're using the simplified context structure
            if "event_data" in context:
                logger.info(
                    f"Using simplified context with full event_data, keys: {list(context['event_data'].keys())[:10]}{'...' if len(context['event_data'].keys()) > 10 else ''}"
                )

            # Build the prompt
            prompt = ActivityManager._build_prompt(activity_type, context)

            # Get the LLM client - use configured provider instead of hardcoded Google
            from src.configuration import Configuration
            config = Configuration.from_runnable_config()
            provider = config.activity_llm_provider.value if hasattr(config.activity_llm_provider, 'value') else config.activity_llm_provider
            model = config.activity_llm_model

            llm = get_llm_client(provider, model)

            # Get the response from the model
            response = get_model_response(llm, "", prompt)

            # Clean up the response
            if isinstance(response, str):
                message = response.strip()
            else:
                # Handle complex response object
                # This assumes your get_model_response may return an object with content
                try:
                    if hasattr(response, "content"):
                        message = response.content.strip()
                    elif hasattr(response, "message") and hasattr(
                        response.message, "content"
                    ):
                        message = response.message.content.strip()
                    else:
                        message = str(response).strip()
                except:
                    message = str(response).strip()

            # Further clean-up: remove quotation marks if they wrap the entire message
            if (message.startswith('"') and message.endswith('"')) or (
                message.startswith("'") and message.endswith("'")
            ):
                message = message[1:-1].strip()

            return message

        except Exception as e:
            logger.error(f"Error generating dynamic activity message: {e}")
            return None

    @staticmethod
    async def create_activity_event(
        event_type: str, event_data: Dict[str, Any]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:  # Return type hint added
        """
        Create an activity event with enriched data about sources and citations.

        This method handles global deduplication of domains and images to prevent
        duplicate web links and visualizations from appearing in the UI.

        - Domains are tracked globally using _sent_domain_urls
        - Images are tracked globally using _sent_image_filenames
        - Only new content is included in activity events
        """
        try:
            # Generate a unique ID for this event processing call for tracing
            import uuid

            event_id = str(uuid.uuid4())[:8]

            # CRITICAL DEBUG SECTION - Remove after fixing duplicate image issue
            logger.info(
                f"===== [EVENT-{event_id}] PROCESSING EVENT: {event_type} ====="
            )

            # Change: Use event_data directly as raw_data
            raw_data = event_data
            logger.info(f"[EVENT-{event_id}] Using event_data directly as raw_data.")
            logger.info(f"[EVENT-{event_id}] raw_data keys: {list(raw_data.keys())}")

            # Log fields relevant to sources and domains
            logger.info(
                f"[EVENT-{event_id}] raw_data['sources'] (for sources_from_state): {raw_data.get('sources')}"
            )
            logger.info(
                f"[EVENT-{event_id}] raw_data['domains'] (for domains_data prio 1): {raw_data.get('domains')}"
            )
            if isinstance(raw_data.get("input"), dict):
                logger.info(
                    f"[EVENT-{event_id}] raw_data['input']['domains'] (for domains_data prio 2): {raw_data['input'].get('domains')}"
                )
            else:
                logger.info(
                    f"[EVENT-{event_id}] raw_data['input'] is not a dict or missing."
                )

            web_research_results_log = raw_data.get("web_research_results")
            logger.info(
                f"[EVENT-{event_id}] raw_data['web_research_results'] (for domains_data prio 3): {type(web_research_results_log)} - Len: {len(web_research_results_log) if isinstance(web_research_results_log, list) else 'N/A'}"
            )
            if isinstance(web_research_results_log, list) and web_research_results_log:
                first_item_str = str(web_research_results_log[0])
                # Truncate if too long (especially for base64 data)
                if len(first_item_str) > 500:
                    logger.info(
                        f"[EVENT-{event_id}] First item of web_research_results: {first_item_str[:200]}... [TRUNCATED - full length: {len(first_item_str)}]"
                    )
                else:
                    logger.info(
                        f"[EVENT-{event_id}] First item of web_research_results: {first_item_str}"
                    )

            formatted_sources_log = raw_data.get("formatted_sources")
            logger.info(
                f"[EVENT-{event_id}] raw_data['formatted_sources'] (for domains_data prio 4): {type(formatted_sources_log)} - Len: {len(formatted_sources_log) if isinstance(formatted_sources_log, list) else 'N/A'}"
            )
            if isinstance(formatted_sources_log, list) and formatted_sources_log:
                first_item_str = str(formatted_sources_log[0])
                # Truncate if too long
                if len(first_item_str) > 500:
                    logger.info(
                        f"[EVENT-{event_id}] First item of formatted_sources: {first_item_str[:200]}... [TRUNCATED - full length: {len(first_item_str)}]"
                    )
                else:
                    logger.info(
                        f"[EVENT-{event_id}] First item of formatted_sources: {first_item_str}"
                    )

            # Check if this event contains visualizations
            # raw_data = event_data.get('data', event_data) # This line is now replaced by the above
            has_results = False
            results_path = None

            # Check all possible paths where visualization data might be
            paths_to_check = [
                "results",
                "search_result.results",
                "code_data.results",
                "visualizations",
                "input.results",
                "input.visualizations",
                "input.visualization_results",
            ]

            for path in paths_to_check:
                parts = path.split(".")
                current = raw_data
                valid_path = True

                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        valid_path = False
                        break

                if valid_path and current:
                    has_results = True
                    results_path = path
                    logger.info(
                        f"[EVENT-{event_id}] Found visualization results at path: {path}"
                    )

                    # Check content type
                    if isinstance(current, list):
                        logger.info(
                            f"[EVENT-{event_id}] Results is a list with {len(current)} items"
                        )
                        # Show first item sample
                        if current and len(current) > 0:
                            sample = current[0]
                            if isinstance(sample, dict):
                                logger.info(
                                    f"[EVENT-{event_id}] Sample keys: {list(sample.keys())}"
                                )
                            else:
                                logger.info(
                                    f"[EVENT-{event_id}] Sample type: {type(sample)}"
                                )
                    elif isinstance(current, dict):
                        logger.info(
                            f"[EVENT-{event_id}] Results is a dict with keys: {list(current.keys())}"
                        )

            if not has_results:
                logger.info(
                    f"[EVENT-{event_id}] No visualization results found in this event"
                )

            # Extract code snippets from the event data
            code_snippets = []
            logger.info(
                f"[EVENT-{event_id}] Initializing code_snippets list (currently empty)."
            )  # Log initialization

            # --- START MODIFICATION ---
            # Prioritize checking the top-level 'code_snippets' key first
            if (
                isinstance(raw_data, dict)
                and "code_snippets" in raw_data
                and isinstance(raw_data["code_snippets"], list)
            ):
                top_level_snippets = raw_data.get("code_snippets", [])
                if top_level_snippets:
                    logger.info(
                        f"[EVENT-{event_id}] Found {len(top_level_snippets)} snippets directly in raw_data['code_snippets']."
                    )
                    code_snippets.extend(top_level_snippets)
                else:
                    logger.info(
                        f"[EVENT-{event_id}] No snippets found directly in raw_data['code_snippets']."
                    )
            # --- END MODIFICATION ---

            if isinstance(raw_data, dict):
                # Check for code snippets in enriched_data (as a fallback or secondary location)
                if "enriched_data" in raw_data and isinstance(
                    raw_data["enriched_data"], dict
                ):
                    found_in_enriched = raw_data["enriched_data"].get(
                        "code_snippets", []
                    )
                    if found_in_enriched:
                        logger.info(
                            f"[EVENT-{event_id}] Found {len(found_in_enriched)} snippets in raw_data['enriched_data']."
                        )
                        code_snippets.extend(
                            found_in_enriched
                        )  # Extend, don't overwrite
                    # else: # No need to log 'not found' here if already checked top-level
                    #      logger.info(f"[EVENT-{event_id}] No snippets found directly in raw_data['enriched_data'].")

                # Recursively search for code snippets in nested structures like subtopic_results or visualizations
                def extract_nested_code_snippets(data, path=""):  # Add path tracking
                    # (rest of the recursive function remains the same)
                    snippets = []
                    current_path = path if path else "root"
                    # logger.debug(f"[EVENT-{event_id}] Searching for snippets in path: {current_path}") # Optional debug log
                    if isinstance(data, dict):
                        # Check the 'code_snippets' key within this dictionary
                        if "code_snippets" in data and isinstance(
                            data["code_snippets"], list
                        ):
                            found_nested = data.get("code_snippets", [])
                            if found_nested:
                                logger.info(
                                    f"[EVENT-{event_id}] Found {len(found_nested)} nested snippets at path: {current_path}.code_snippets"
                                )
                                snippets.extend(found_nested)
                        # Recursively search deeper
                        for key, value in data.items():
                            # Avoid re-searching the top-level 'code_snippets' if we already checked it
                            if key == "code_snippets" and path == "":
                                continue
                            snippets.extend(
                                extract_nested_code_snippets(
                                    value, f"{current_path}.{key}"
                                )
                            )
                    elif isinstance(data, (list, tuple)):
                        for i, item in enumerate(data):
                            snippets.extend(
                                extract_nested_code_snippets(
                                    item, f"{current_path}[{i}]"
                                )
                            )
                    return snippets

                # Extract from raw_data input or results (if not found at top level)
                # Only search these if the top-level check didn't find anything substantial
                if not code_snippets:  # <-- Check if snippets were already found
                    logger.info(
                        f"[EVENT-{event_id}] No top-level snippets found, searching nested 'input' and 'results'..."
                    )
                    if "input" in raw_data:
                        logger.info(
                            f"[EVENT-{event_id}] Searching for snippets within raw_data['input']..."
                        )
                        input_snippets = extract_nested_code_snippets(
                            raw_data["input"], "input"
                        )
                        if input_snippets:
                            logger.info(
                                f"[EVENT-{event_id}] Found {len(input_snippets)} snippets within raw_data['input']."
                            )
                            code_snippets.extend(input_snippets)
                        # else: # Reduce log noise
                        #      logger.info(f"[EVENT-{event_id}] No snippets found within raw_data['input'].")

                    if "results" in raw_data:
                        logger.info(
                            f"[EVENT-{event_id}] Searching for snippets within raw_data['results']..."
                        )
                        results_snippets = extract_nested_code_snippets(
                            raw_data["results"], "results"
                        )
                        if results_snippets:
                            logger.info(
                                f"[EVENT-{event_id}] Found {len(results_snippets)} snippets within raw_data['results']."
                            )
                            code_snippets.extend(results_snippets)
                        # else: # Reduce log noise
                        #      logger.info(f"[EVENT-{event_id}] No snippets found within raw_data['results'].")
                else:
                    logger.info(
                        f"[EVENT-{event_id}] Skipping nested search as top-level snippets were found."
                    )

            else:
                logger.warning(
                    f"[EVENT-{event_id}] raw_data is not a dict, cannot search for snippets."
                )

            # --- Deduplication and adding to enriched_data logic remains the same ---
            if code_snippets:
                # Deduplicate before adding to enriched_data
                unique_snippets_map = {}
                for snippet in code_snippets:
                    if isinstance(snippet, dict):
                        # Create a unique key based on code content
                        code_content = snippet.get("code", "")
                        if code_content:
                            # Use a hash or just the code itself if not too long
                            snippet_key = hash(code_content)
                            if snippet_key not in unique_snippets_map:
                                unique_snippets_map[snippet_key] = snippet
                    else:
                        logger.warning(
                            f"[EVENT-{event_id}] Skipping non-dict item during deduplication: {type(snippet)}"
                        )

                deduplicated_code_snippets = list(unique_snippets_map.values())
                logger.info(
                    f"[EVENT-{event_id}] Found {len(code_snippets)} total code snippets, {len(deduplicated_code_snippets)} unique after deduplication."
                )

                # Ensure enriched_data exists and add snippets
                if isinstance(raw_data, dict):
                    logger.info(
                        f"[EVENT-{event_id}] Raw data is a dict. Proceeding to add snippets."
                    )
                    if "enriched_data" not in raw_data or not isinstance(
                        raw_data.get("enriched_data"), dict
                    ):
                        logger.info(
                            f"[EVENT-{event_id}] Initializing empty enriched_data."
                        )
                        raw_data["enriched_data"] = {}
                    else:
                        logger.info(f"[EVENT-{event_id}] enriched_data already exists.")

                    # Combine potentially pre-existing snippets with newly found unique ones
                    existing_snippet_list = raw_data["enriched_data"].get(
                        "code_snippets", []
                    )
                    # Use a simpler deduplication based on the unique map we already created
                    final_snippets_map = {
                        hash(s.get("code", "")): s
                        for s in existing_snippet_list
                        if isinstance(s, dict) and s.get("code")
                    }
                    added_count = 0
                    for snippet_key, snippet_dict in unique_snippets_map.items():
                        if snippet_key not in final_snippets_map:
                            final_snippets_map[snippet_key] = snippet_dict
                            added_count += 1

                    final_combined_snippets = list(final_snippets_map.values())

                    logger.info(
                        f"[EVENT-{event_id}] Added {added_count} new unique snippets to the list."
                    )
                    raw_data["enriched_data"]["code_snippets"] = final_combined_snippets
                    logger.info(
                        f"[EVENT-{event_id}] Successfully updated enriched_data. Final total code snippets: {len(final_combined_snippets)}"
                    )
                else:
                    logger.warning(
                        f"[EVENT-{event_id}] Raw data is not a dict ({type(raw_data)}), cannot add snippets."
                    )
            else:
                logger.info(
                    f"[EVENT-{event_id}] No code snippets found after searching raw_data."
                )

            # Regular processing continues...
            logger.info(
                f"===== [EVENT-{event_id}] CREATING ACTIVITY EVENT FOR: {event_type} ====="
            )
            logger.info(
                f"[EVENT-{event_id}] Event data keys: {list(event_data.keys())}"
            )

            # --- START OF CLASS VARIABLE INITIALIZATION ---
            # Ensure all class tracking variables are properly initialized
            if not hasattr(ActivityManager, "_current_research_loop"):
                ActivityManager._current_research_loop = 0

            if not hasattr(ActivityManager, "_sent_domain_urls"):
                ActivityManager._sent_domain_urls = set()

            # --- END OF CLASS VARIABLE INITIALIZATION ---

            raw_data = event_data.get("data", event_data)
            logger.info(f"Raw_data keys: {list(raw_data.keys())}")

            # Extract research topic
            research_topic = raw_data.get("research_topic", None)
            if not research_topic and isinstance(raw_data.get("input"), dict):
                research_topic = raw_data["input"].get(
                    "research_topic", "unknown_topic"
                )

            # Extract source information from raw_data
            sources_from_state = raw_data.get(
                "sources", []
            )  # This is likely state.sources_gathered
            citations = raw_data.get("citations", {})
            source_citations = raw_data.get("source_citations", {})

            # Extract domains from raw_data or nested input/web_research_results
            # This will store domain objects {url: ..., title: ...}
            domains_data = []
            seen_domain_urls = set()

            # Priority 1: Top-level 'domains' if already in correct format or list of strings
            if isinstance(raw_data.get("domains"), list):
                potential_domains = raw_data["domains"]
                logger.info(
                    f"[EVENT-{event_id}] Found {len(potential_domains)} domains in raw_data['domains']"
                )
                for d_item in potential_domains:
                    if isinstance(d_item, dict) and "url" in d_item:
                        url = d_item["url"]
                        if url and url not in seen_domain_urls:
                            domains_data.append(
                                {
                                    "url": url,
                                    "title": d_item.get("title", extract_domain(url)),
                                }
                            )
                            seen_domain_urls.add(url)
                    elif isinstance(d_item, str) and d_item.strip():
                        # Assume it's a domain name or full URL string
                        domain_str_cleaned = d_item.strip()
                        url = (
                            domain_str_cleaned
                            if domain_str_cleaned.startswith("http")
                            else f"https://{domain_str_cleaned}"
                        )
                        if url not in seen_domain_urls:
                            domains_data.append(
                                {
                                    "url": url,
                                    "title": extract_domain(domain_str_cleaned),
                                }
                            )
                            seen_domain_urls.add(url)

            # Priority 2: 'input.domains'
            if (
                not domains_data
                and isinstance(raw_data.get("input"), dict)
                and isinstance(raw_data["input"].get("domains"), list)
            ):
                potential_domains = raw_data["input"]["domains"]
                logger.info(
                    f"[EVENT-{event_id}] Found {len(potential_domains)} domains in raw_data['input']['domains']"
                )
                for d_item in potential_domains:
                    if isinstance(d_item, dict) and "url" in d_item:
                        url = d_item["url"]
                        if url and url not in seen_domain_urls:
                            domains_data.append(
                                {
                                    "url": url,
                                    "title": d_item.get("title", extract_domain(url)),
                                }
                            )
                            seen_domain_urls.add(url)
                    elif isinstance(d_item, str) and d_item.strip():
                        domain_str_cleaned = d_item.strip()
                        url = (
                            domain_str_cleaned
                            if domain_str_cleaned.startswith("http")
                            else f"https://{domain_str_cleaned}"
                        )
                        if url not in seen_domain_urls:
                            domains_data.append(
                                {
                                    "url": url,
                                    "title": extract_domain(domain_str_cleaned),
                                }
                            )
                            seen_domain_urls.add(url)

            # Priority 3: Aggregate from 'web_research_results'
            # web_research_results is expected to be at raw_data.web_research_results if passed from service state
            web_results_for_domains = raw_data.get("web_research_results")
            if not domains_data and isinstance(web_results_for_domains, list):
                logger.info(
                    f"[EVENT-{event_id}] Attempting to extract domains from raw_data['web_research_results'] ({len(web_results_for_domains)} items)"
                )
                for result_item in web_results_for_domains:
                    if isinstance(result_item, dict) and isinstance(
                        result_item.get("domains"), list
                    ):
                        for domain_str in result_item["domains"]:
                            if isinstance(domain_str, str) and domain_str.strip():
                                domain_str_cleaned = domain_str.strip()
                                url = (
                                    domain_str_cleaned
                                    if domain_str_cleaned.startswith("http")
                                    else f"https://{domain_str_cleaned}"
                                )
                                if url not in seen_domain_urls:
                                    title = extract_domain(domain_str_cleaned)
                                    domains_data.append({"url": url, "title": title})
                                    seen_domain_urls.add(url)
                if domains_data:
                    logger.info(
                        f"[EVENT-{event_id}] Extracted {len(domains_data)} domains from raw_data['web_research_results']"
                    )

            # Priority 4: Fallback to 'formatted_sources' (less ideal as it's parsing strings)
            # formatted_sources is expected to be at raw_data.formatted_sources if passed from service state
            formatted_sources_for_domains = raw_data.get("formatted_sources")
            if not domains_data and isinstance(formatted_sources_for_domains, list):
                logger.info(
                    f"[EVENT-{event_id}] Attempting to extract domains from raw_data['formatted_sources']"
                )
                for source_str_entry in formatted_sources_for_domains:
                    if isinstance(source_str_entry, str) and " : " in source_str_entry:
                        parts = source_str_entry.split(" : ", 1)
                        if len(parts) >= 2:
                            url = parts[1].strip()
                            if (
                                url and url not in seen_domain_urls
                            ):  # Ensure URL is not empty
                                title = extract_domain(url)
                                domains_data.append({"url": url, "title": title})
                                seen_domain_urls.add(url)
                if domains_data:
                    logger.info(
                        f"[EVENT-{event_id}] Extracted {len(domains_data)} domains from raw_data['formatted_sources']"
                    )

            # Log the extracted information
            logger.info(f"Sources from state count: {len(sources_from_state)}")
            logger.info(f"Citations count: {len(citations)}")
            logger.info(f"Final domains_data count: {len(domains_data)}")
            logger.info(f"Has source_citations: {len(source_citations) > 0}")

            # Extract node name and get loop ID
            node_name = event_data.get("node_name", "")

            # Get the current research loop count
            loop_id = 0
            if "research_loop_count" in raw_data:  # Check raw_data first
                loop_id = raw_data["research_loop_count"]
            elif "input" in raw_data and "research_loop_count" in raw_data["input"]:
                loop_id = raw_data["input"]["research_loop_count"]

            # Check if we need to reset tracking for a new loop or multi_agents_network
            is_new_loop = loop_id != ActivityManager._current_research_loop

            if is_new_loop:
                logger.info(
                    f"RESET EVENT: New loop {loop_id} (was {ActivityManager._current_research_loop})"
                )
                ActivityManager._current_research_loop = loop_id
                # Note: Domain tracking is now handled globally, not per-loop

            # Filter domains using global tracking (similar to image tracking)
            if domains_data:
                logger.info(
                    f"[EVENT-{event_id}] Processing {len(domains_data)} domains for global deduplication"
                )

                # Filter out domains that have already been sent globally
                new_domains_for_event = []
                for domain in domains_data:
                    domain_url = domain.get("url", "")
                    if (
                        domain_url
                        and domain_url not in ActivityManager._sent_domain_urls
                    ):
                        new_domains_for_event.append(domain)
                        # Add to global tracking immediately
                        ActivityManager._sent_domain_urls.add(domain_url)
                        logger.info(f"[EVENT-{event_id}] New domain: {domain_url}")
                    else:
                        logger.info(
                            f"[EVENT-{event_id}] Skipping already sent domain: {domain_url}"
                        )

                domains_data_for_event = new_domains_for_event
                logger.info(
                    f"[EVENT-{event_id}] After global deduplication: {len(domains_data_for_event)} domains to show"
                )
                logger.info(
                    f"[EVENT-{event_id}] Total domains tracked globally: {len(ActivityManager._sent_domain_urls)}"
                )
            else:
                domains_data_for_event = []

            # Always clear domains list for certain activity types that should not show domains
            should_clear_domains = False

            # NEW LOGIC: Only show domains in multi_agents_network stage, filter out from all others
            # Check if this is NOT from the multi_agents_network node
            if node_name and "multi_agents_network" not in node_name.lower():
                should_clear_domains = True
                logger.info(
                    f"Clearing domains for non-multi_agents_network node: {node_name}"
                )

            # Also clear domains for certain event types that should never show domains
            elif event_type in [
                "visualization_generated",
                "knowledge_gap_identified",
                "knowledge_gap",
            ]:
                should_clear_domains = True
                logger.info(
                    f"Clearing domains for event type {event_type} that should never show domains"
                )

            # Special handling for activity_generated events - check the related_event_type and node_name
            elif event_type == "activity_generated":
                related_event_type = event_data.get("related_event_type", "")
                activity_node_name = event_data.get("node_name", "")

                # Only show domains if this activity is related to multi_agents_network
                if (
                    activity_node_name
                    and "multi_agents_network" not in activity_node_name.lower()
                ):
                    should_clear_domains = True
                    logger.info(
                        f"Clearing domains for activity_generated event from non-multi_agents_network node: {activity_node_name}"
                    )
                elif (
                    related_event_type
                    and "multi_agents_network" not in related_event_type.lower()
                ):
                    should_clear_domains = True
                    logger.info(
                        f"Clearing domains for activity_generated event with non-multi_agents_network related_event_type: {related_event_type}"
                    )

            if should_clear_domains:
                domains_data_for_event = []
                logger.info(
                    f"[EVENT-{event_id}] Domains cleared - web links only allowed in multi_agents_network stage"
                )
            else:
                logger.info(
                    f"[EVENT-{event_id}] Domains allowed for multi_agents_network stage"
                )

            # Debug: check for visualization results in raw_data
            logger.info(f"Raw_data 'results' field: {raw_data.get('results')}")
            logger.info(f"Raw_data keys: {list(raw_data.keys())}")

            # Extract visualization images - look in various possible places
            images = []
            visualization_results = []  # Initialize

            # OPTION B: Check multiple possible locations for visualizations
            # Check direct results field
            viz_results_raw = raw_data.get("results", [])
            if (
                isinstance(viz_results_raw, dict) and "results" in viz_results_raw
            ):  # Handle nested structure
                visualization_results = viz_results_raw.get("results", [])
            elif isinstance(viz_results_raw, list):
                visualization_results = viz_results_raw

            # Also check search_result.results if that exists
            if not visualization_results and isinstance(
                raw_data.get("search_result"), dict
            ):
                viz_results_raw = raw_data.get("search_result", {}).get("results", [])
                if isinstance(viz_results_raw, dict) and "results" in viz_results_raw:
                    visualization_results = viz_results_raw.get("results", [])
                elif isinstance(viz_results_raw, list):
                    visualization_results = viz_results_raw

            # Also check code_data.results if that exists
            if not visualization_results and isinstance(
                raw_data.get("code_data"), dict
            ):
                viz_results_raw = raw_data.get("code_data", {}).get("results", [])
                if isinstance(viz_results_raw, dict) and "results" in viz_results_raw:
                    visualization_results = viz_results_raw.get("results", [])
                elif isinstance(viz_results_raw, list):
                    visualization_results = viz_results_raw
                logger.info(f"Checking code_data.results: {visualization_results}")

            # Also check visualizations field
            if not visualization_results:
                viz_results_raw = raw_data.get("visualizations", [])
                if (
                    isinstance(viz_results_raw, dict) and "results" in viz_results_raw
                ):  # Handle potential nesting here too
                    visualization_results = viz_results_raw.get("results", [])
                elif isinstance(viz_results_raw, list):
                    visualization_results = viz_results_raw
                logger.info(f"Checking visualizations field: {visualization_results}")

            # Last attempt - check if there's a nested structure in input
            if not visualization_results and "input" in raw_data:
                input_data = raw_data.get("input", {})
                if isinstance(input_data, dict):
                    # Check various possible paths in the input data
                    for field in [
                        "results",
                        "visualizations",
                        "visualization_results",
                        "base64_encoded_images",
                    ]:
                        if field in input_data:
                            viz_results_raw = input_data.get(field, [])
                            if (
                                isinstance(viz_results_raw, dict)
                                and "results" in viz_results_raw
                            ):
                                visualization_results = viz_results_raw.get(
                                    "results", []
                                )
                            elif isinstance(viz_results_raw, list):
                                visualization_results = viz_results_raw
                            break
            # Process the found visualization results
            if isinstance(visualization_results, list) and visualization_results:
                logger.info(
                    f"Found visualization results list. Processing {len(visualization_results)} items for images."
                )

                # Check for potential duplicates in the raw visualization_results
                image_signatures = set()
                potential_duplicates = 0

                for item in visualization_results:
                    # Generate a simple signature for duplicate detection
                    if isinstance(item, dict):
                        item_sig = (
                            f"{item.get('filepath', '')}_{item.get('description', '')}"
                        )
                        if item_sig in image_signatures:
                            potential_duplicates += 1
                        else:
                            image_signatures.add(item_sig)

                if potential_duplicates > 0:
                    logger.warning(
                        f"Found {potential_duplicates} potential duplicate images in visualization_results!"
                    )

                for item in visualization_results:
                    # Handle both dict and potentially string items (if paths were stored directly)
                    img_src = None
                    filepath = None
                    description = None
                    img_format = "png"  # Default

                    if isinstance(item, dict):
                        img_src = item.get("src")  # Check for pre-formatted src first
                        filepath = item.get("filepath")
                        description = item.get("description")
                        img_format = item.get("format", "png")
                        img_data_b64 = item.get("data")  # Check for base64 data

                        if (
                            not img_src and img_data_b64
                        ):  # Construct src from base64 data if src missing
                            img_src = f"data:image/{img_format};base64,{img_data_b64}"
                            logger.info(
                                f"Constructed 'src' from 'data' field for image: {description or filepath} (base64 length: {len(img_data_b64) if img_data_b64 else 0})"
                            )

                    elif isinstance(item, str) and os.path.exists(
                        item
                    ):  # Handle direct file path string
                        filepath = item
                        description = os.path.basename(
                            filepath
                        )  # Use filename as description fallback

                    # Now, prioritize src, then filepath encoding
                    if img_src:
                        # Log image info without exposing base64 data
                        src_info = (
                            "base64 data"
                            if img_src.startswith("data:image")
                            else "file path"
                        )
                        src_length = (
                            len(img_src) if img_src.startswith("data:image") else "N/A"
                        )
                        images.append(
                            {
                                "src": img_src,
                                "description": description,
                                "format": img_format,
                            }
                        )
                        logger.info(
                            f"Using existing/constructed 'src' for image: {description or filepath} ({src_info}, length: {src_length})"
                        )
                    elif filepath and os.path.exists(filepath):
                        try:
                            with open(filepath, "rb") as img_f:
                                b64 = base64.b64encode(img_f.read()).decode("utf-8")
                            img_src = f"data:image/{img_format};base64,{b64}"
                            images.append(
                                {
                                    "src": img_src,
                                    "description": description,
                                    "format": img_format,
                                }
                            )
                            logger.info(
                                f"Successfully encoded image from filepath: {filepath} (base64 length: {len(b64)})"
                            )
                        except Exception as img_err:
                            logger.error(
                                f"Error encoding image {filepath} from event data: {img_err}"
                            )
                    else:
                        # Log item info without potentially large data
                        item_info = str(item)
                        if len(item_info) > 200:
                            item_info = (
                                item_info[:200]
                                + f"... [TRUNCATED - full length: {len(str(item))}]"
                            )
                        logger.warning(
                            f"Could not process visualization item: {item_info}"
                        )
            else:
                logger.info(
                    f"No valid visualization results found after checking multiple locations in event data for event {event_type}."
                )

            # Generate the main activity text
            activity_text = await ActivityManager._generate_activity_text(
                event_type, event_data
            )

            # Format sources for UI
            formatted_sources_for_ui = []
            if sources_from_state:  # sources_from_state is state.sources_gathered
                for source_str_entry in sources_from_state:
                    if isinstance(source_str_entry, str) and " : " in source_str_entry:
                        parts = source_str_entry.split(" : ", 1)
                        if len(parts) == 2:
                            title, url = parts[0].strip(), parts[1].strip()
                            formatted_sources_for_ui.append(
                                {"title": title, "url": url}
                            )
                    elif (
                        isinstance(source_str_entry, dict)
                        and "title" in source_str_entry
                        and "url" in source_str_entry
                    ):
                        formatted_sources_for_ui.append(source_str_entry)

            logger.info(
                f"[EVENT-{event_id}] Final formatted_sources_for_ui count: {len(formatted_sources_for_ui)}"
            )
            if formatted_sources_for_ui:
                # Log just the first source without potentially large data
                sample_source = formatted_sources_for_ui[0]
                if isinstance(sample_source, dict):
                    sample_str = f"{{title: '{sample_source.get('title', 'N/A')}', url: '{sample_source.get('url', 'N/A')[:50]}...'}}"
                else:
                    sample_str = str(sample_source)[:100] + (
                        "..." if len(str(sample_source)) > 100 else ""
                    )
                logger.info(
                    f"[EVENT-{event_id}] Sample formatted_sources_for_ui[0]: {sample_str}"
                )
            logger.info(
                f"[EVENT-{event_id}] Final domains_data_for_event count: {len(domains_data_for_event)}"
            )
            if domains_data_for_event:
                # Log just the first domain without potentially large data
                sample_domain = domains_data_for_event[0]
                if isinstance(sample_domain, dict):
                    sample_str = f"{{url: '{sample_domain.get('url', 'N/A')[:50]}...', title: '{sample_domain.get('title', 'N/A')}'}}"
                else:
                    sample_str = str(sample_domain)[:100] + (
                        "..." if len(str(sample_domain)) > 100 else ""
                    )
                logger.info(
                    f"[EVENT-{event_id}] Sample domains_data_for_event[0]: {sample_str}"
                )

            # Create the main activity event dictionary
            activity_event = {
                "event_type": "activity_generated",
                "data": {
                    "activity": activity_text,
                    "node_name": event_data.get("node_name"),
                    "related_event_type": event_type,
                    "enriched_data": {
                        "sources": formatted_sources_for_ui,  # Use the formatted list
                        "citations": citations,  # This is fine as is (dict)
                        "source_citations": source_citations,  # This is fine as is (dict)
                        "domains": domains_data_for_event,  # Use the collected and formatted domains_data
                        "domain_count": len(domains_data_for_event),
                        "images": images,
                        "code_snippets": code_snippets,  # Add code snippets to enriched_data
                    },
                },
                "timestamp": event_data.get(
                    "timestamp", datetime.datetime.now().isoformat()
                ),
            }

            # --- START COMPLETELY REVISED VISUALIZATION EVENT CREATION ---
            visualization_events = []  # Initialize visualization events list

            # Process images if there are any - check both global and batch duplicates
            if images and len(images) > 0:
                logger.info(
                    f"[EVENT-{event_id}] Processing {len(images)} raw visualization images"
                )

                # Step 1: Identify images in this batch not yet sent globally
                new_images_to_send = []
                for idx, img in enumerate(images):
                    # Get image identifier (filename or hash)
                    filename = None
                    if "description" in img and img["description"]:
                        filename = img["description"]
                    elif "filename" in img and img["filename"]:
                        filename = img["filename"]
                    if not filename and "src" in img and img["src"]:
                        src = img["src"]
                        if src.startswith("data:image"):
                            parts = src.split("base64,")
                            if len(parts) > 1:
                                base64_data = parts[1]
                                import hashlib

                                hash_obj = hashlib.md5(base64_data.encode())
                                hash_digest = hash_obj.hexdigest()
                                filename = f"base64_{hash_digest[:16]}"
                    if not filename:
                        filename = f"img_hash_{hash(str(img))}"

                    # Log identifier and global check status
                    logger.info(
                        f"[EVENT-{event_id}] Image {idx+1}/{len(images)} - Identifier: {filename}"
                    )
                    is_already_sent_globally = (
                        filename in ActivityManager._sent_image_filenames
                    )
                    logger.info(
                        f"[EVENT-{event_id}] Image {idx+1}/{len(images)} - Already sent globally? {is_already_sent_globally}"
                    )

                    # Collect images that haven't been sent globally yet
                    if not is_already_sent_globally:
                        new_images_to_send.append(
                            (img, filename)
                        )  # Store tuple (image_data, identifier)
                    else:
                        logger.info(
                            f"[EVENT-{event_id}] Skipping globally tracked image: {filename}"
                        )

                # Log the current global tracking state
                logger.info(
                    f"[EVENT-{event_id}] Globally tracking {len(ActivityManager._sent_image_filenames)} images before adding new ones"
                )

                # Step 2: If no globally new images, we're done with visualizations for this event
                if not new_images_to_send:
                    logger.info(
                        f"[EVENT-{event_id}] All images in this batch were already sent globally. No visualization events to create."
                    )
                    # Return the main activity event without visualization events
                    return activity_event

                logger.info(
                    f"[EVENT-{event_id}] Identified {len(new_images_to_send)} potentially new images (not sent globally yet)"
                )

                # Step 3: Check if this specific batch of *new* images was already processed
                # (This handles cases where the exact same *set of new images* appears in a rapidly succeeding event)
                import hashlib

                new_image_identifiers_sorted = sorted(
                    [item[1] for item in new_images_to_send]
                )  # Sort identifiers for consistent hashing
                batch_signature = hashlib.md5(
                    str(new_image_identifiers_sorted).encode()
                ).hexdigest()

                if not hasattr(ActivityManager, "_processed_batches"):
                    ActivityManager._processed_batches = set()

                if batch_signature in ActivityManager._processed_batches:
                    logger.warning(
                        f"[EVENT-{event_id}] DUPLICATE BATCH of new images detected! Signature: {batch_signature[:10]}..."
                    )
                    logger.warning(
                        f"[EVENT-{event_id}] This exact set of {len(new_images_to_send)} new images was processed recently. Skipping."
                    )
                    return activity_event  # Skip creating viz events
                else:
                    # Track this batch of NEW images
                    ActivityManager._processed_batches.add(batch_signature)
                    logger.info(
                        f"[EVENT-{event_id}] New batch of {len(new_images_to_send)} unsent images with signature: {batch_signature[:10]}..."
                    )

                # Step 4: Create visualization events for the new images, up to the limit
                display_limit = 5  # Use the limit you set previously
                images_to_display = new_images_to_send[
                    : min(len(new_images_to_send), display_limit)
                ]

                logger.info(
                    f"[EVENT-{event_id}] Preparing to create events for {len(images_to_display)} new images (limit: {display_limit})"
                )

                for img_data, filename in images_to_display:
                    # Mark this image as globally sent NOW, before creating the event
                    if filename not in ActivityManager._sent_image_filenames:
                        ActivityManager._sent_image_filenames.add(filename)
                        logger.info(
                            f"[EVENT-{event_id}] Added {filename} to global tracking."
                        )

                    # Generate a simple descriptive message
                    viz_message = f"Created a data visualization related to {research_topic or 'this research topic'}."

                    # Create the separate event
                    viz_event = {
                        "event_type": "activity_generated",
                        "eventId": f"viz-{filename}-{event_id}",  # Add unique event ID
                        "data": {
                            "activity": viz_message,
                            "node_name": "visualization_result",
                            "related_event_type": "visualization_generated",
                            "enriched_data": {
                                "images": [img_data],  # Event contains only one image
                                "sources": [],
                                "citations": {},
                                "source_citations": {},
                                "domains": [],
                                "domain_count": 0,
                            },
                        },
                        "timestamp": event_data.get(
                            "timestamp", datetime.datetime.now().isoformat()
                        ),
                    }
                    visualization_events.append(viz_event)
                    logger.info(
                        f"[EVENT-{event_id}] Created visualization event {len(visualization_events)} for image {filename}"
                    )

                # Log final global tracking state after processing this batch
                logger.info(
                    f"[EVENT-{event_id}] Globally tracking {len(ActivityManager._sent_image_filenames)} images after processing this batch."
                )

            # --- END COMPLETELY REVISED VISUALIZATION EVENT CREATION ---

            # Log the final structure before returning
            final_return_value = None
            if visualization_events:
                # Combine main event (without images) and separate viz events
                # Remove images from the main event if we are sending separate viz events
                activity_event["data"]["enriched_data"]["images"] = []
                final_return_value = [activity_event] + visualization_events
                logger.info(
                    f"[EVENT-{event_id}] Returning {len(final_return_value)} events: 1 main activity (images removed) + {len(visualization_events)} visualization events."
                )
            else:
                # Return only the main activity event (which now includes images if found)
                final_return_value = activity_event
                logger.info(
                    f"[EVENT-{event_id}] Returning 1 main activity event. Images included: {len(activity_event['data']['enriched_data']['images'])}"
                )

            return final_return_value

        except Exception as e:
            logger.error(f"Error creating activity event: {e}")
            import traceback

            logger.error(traceback.format_exc())  # Log stack trace
            return None  # Return None on error

    @staticmethod
    async def _generate_activity_text(
        event_type: str, event_data: Dict[str, Any]
    ) -> str:
        """Generate activity text based on event type and data."""
        try:
            # Extract available context from the event data
            context = ActivityManager._extract_context(event_type, event_data)

            # Map event_type to appropriate ActivityType
            activity_type = None

            # Handle different event types
            if event_type == "node_start" and "node_name" in event_data:
                node_name = event_data.get("node_name", "").lower()

                # Match specific node names to appropriate activity types
                if "generate_report" in node_name or "report_generator" in node_name:
                    activity_type = ActivityType.GENERATE_INITIAL_REPORT
                    logger.info(f"Mapped node {node_name} to GENERATE_INITIAL_REPORT")
                elif "finalize_report" in node_name:
                    activity_type = ActivityType.FINALIZE_REPORT
                    logger.info(f"Mapped node {node_name} to FINALIZE_REPORT")
                elif "decompose" in node_name:
                    activity_type = ActivityType.DECOMPOSE_RESEARCH_TOPIC
                    logger.info(f"Mapped node {node_name} to DECOMPOSE_RESEARCH_TOPIC")
                elif "visualiz" in node_name:
                    activity_type = ActivityType.GENERATE_VISUALIZATION
                    logger.info(f"Mapped node {node_name} to GENERATE_VISUALIZATION")
                elif "gap" in node_name or "reflect" in node_name:
                    activity_type = ActivityType.IDENTIFY_GAPS
                    logger.info(f"Mapped node {node_name} to IDENTIFY_GAPS")
                elif "linkedin" in node_name or "linked_in" in node_name:
                    activity_type = ActivityType.LINKEDIN_SEARCH
                    logger.info(f"Mapped node {node_name} to LINKEDIN_SEARCH")
                elif "academic" in node_name:
                    activity_type = ActivityType.ACADEMIC_SEARCH
                    logger.info(f"Mapped node {node_name} to ACADEMIC_SEARCH")
                elif "text2sql" in node_name or "database" in node_name:
                    activity_type = ActivityType.TEXT2SQL_SEARCH
                    logger.info(f"Mapped node {node_name} to TEXT2SQL_SEARCH")
                elif "search" in node_name:
                    activity_type = ActivityType.GENERAL_DEEP_SEARCH
                    logger.info(f"Mapped node {node_name} to GENERAL_DEEP_SEARCH")
                elif "multi_agents_network" in node_name:
                    activity_type = ActivityType.EXECUTE_AGENT_PLAN
                    logger.info(f"Mapped node {node_name} to EXECUTE_AGENT_PLAN")
                else:
                    # For unrecognized nodes, default to NODE_START but log it
                    activity_type = ActivityType.NODE_START
                    logger.info(
                        f"No specific mapping for node: {node_name}, using NODE_START"
                    )
            elif event_type == "search_sources_found":
                activity_type = ActivityType.GENERAL_DEEP_SEARCH
            elif event_type == "linkedin_search_results":
                activity_type = ActivityType.LINKEDIN_SEARCH
            elif event_type == "academic_search_results":
                activity_type = ActivityType.ACADEMIC_SEARCH
            elif (
                event_type == "text2sql_search_results"
                or event_type == "database_query_results"
            ):
                activity_type = ActivityType.TEXT2SQL_SEARCH
            elif event_type == "general_deep_search_results":
                activity_type = ActivityType.GENERAL_DEEP_SEARCH
            elif event_type == "visualization_generated":
                activity_type = ActivityType.GENERATE_VISUALIZATION
            elif (
                event_type == "knowledge_gap"
                or event_type == "knowledge_gap_identified"
            ):
                activity_type = ActivityType.IDENTIFY_GAPS
            else:
                # Default to NODE_START for any other events
                activity_type = ActivityType.NODE_START

            # Use the more sophisticated _generate_dynamic_message for all events
            message = await ActivityManager._generate_dynamic_message(
                activity_type, context
            )
            logger.info(f"Generated dynamic message: {message}")
            return message

        except Exception as e:
            logger.error(f"Error generating activity text: {e}")
            # Print stack trace for debugging
            import traceback

            logger.error(f"Stack trace: {traceback.format_exc()}")

            # Fallback to a simple message if something goes wrong
            research_topic = event_data.get("research_topic", "the topic")
            return f"Researching information about {research_topic}."

    @staticmethod
    def should_process_event(
        event_type: str, event_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Quick check to see if we should even try to process this event type.
        This is a performance optimization to skip processing events we know aren't important.
        """
        # Only process these event types
        processable_events = [
            # Core node events
            "node_start",
            # Knowledge gap events
            "knowledge_gap",
            "knowledge_gap_identified",
            # Search events
            "search_sources_found",
            "linkedin_search_results",
            "academic_search_results",
            "general_deep_search_results",
            # Visualization events
            "visualization_generated",
            # Critical system events
            "research_complete",
            "reconnecting",
            "error",
            "connection_interrupted",
        ]

        return event_type in processable_events

    @staticmethod
    def reset_image_tracking() -> None:
        """
        Reset the image and domain tracking to allow previously sent content to be sent again.
        This should be called when starting a new research session or when explicitly
        requested by the application.
        """
        ActivityManager._sent_image_filenames.clear()
        ActivityManager._sent_domain_urls.clear()
        if hasattr(ActivityManager, "_processed_batches"):
            ActivityManager._processed_batches.clear()
        logger.info(
            "Reset image and domain tracking - previously sent images and web links can be sent again"
        )
