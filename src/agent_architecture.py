"""
Agent Architecture for Deep Research

This module implements a modular, agent-based architecture for research:
- Master Agent: Planning, query decomposition, and coordination
- Search Agent: Specialized for executing search queries
- Visualization Agent: Creates data visualizations from search results
- Result Combiner: Integrates findings from multiple sources

The architecture provides better separation of concerns while
maintaining backward compatibility with the existing codebase.
"""

import os
import json
import logging
import traceback
import re
import base64
from typing import Dict, List, Any, Optional
from openai import OpenAI
import asyncio
import uuid
from src.tools.executor import ToolExecutor
from src.visualization_agent import VisualizationAgent
import time


class MasterResearchAgent:
    """
    Master agent responsible for planning research, decomposing topics,
    and coordinating specialized agents.

    This agent serves as the "central brain" that breaks down complex queries
    into manageable subtasks and delegates them to specialized agents.
    """

    def __init__(self, config=None):
        """
        Initialize the Master Research Agent.

        Args:
            config: Configuration object containing LLM settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def decompose_topic(
        self, query, knowledge_gap, research_loop_count, uploaded_knowledge=None
    ):
        """
        Analyze a topic and decompose it if complex.
        Uses query_writer_instructions as the central brain for decomposition.

        Args:
            query: The main research query or topic
            knowledge_gap: Additional context about knowledge gaps to address
            research_loop_count: Current iteration of research loop
            uploaded_knowledge: User-provided external knowledge (optional)

        Returns:
            Dict containing topic_complexity, query/subtopics, and analysis
        """
        # Import necessary functions
        import os
        import json
        import traceback
        from llm_clients import get_llm_client, MODEL_CONFIGS

        try:
            # Get configuration
            if self.config is not None:
                from src.graph import get_configurable

                configurable = get_configurable(self.config)
                provider = configurable.llm_provider
                model = configurable.llm_model
            else:
                # Try environment variables first
                provider = os.environ.get("LLM_PROVIDER")
                model = os.environ.get("LLM_MODEL")

                # Use defaults if not found in environment
                if not provider:
                    provider = "openai"
                # Will use the default model from llm_clients.py if model is None

            # Get the model name from the configuration or default
            if provider == "openai" and not model:
                model = MODEL_CONFIGS["openai"]["default_model"]
            elif not model and provider in MODEL_CONFIGS:
                model = MODEL_CONFIGS[provider]["default_model"]

            self.logger.info(
                f"[MasterAgent] Using {provider} model {model} for topic decomposition"
            )

            # Generate date constants
            from datetime import datetime

            today = datetime.now()
            CURRENT_DATE = today.strftime("%B %d, %Y")
            CURRENT_YEAR = str(today.year)
            ONE_YEAR_AGO = str(today.year - 1)

            # Create a combined query that incorporates knowledge gaps if available
            combined_topic = query
            research_context = ""
            if knowledge_gap and knowledge_gap.strip():
                combined_topic = f"{query} - {knowledge_gap}"
                research_context = (
                    f"Knowledge gap identified in previous research: {knowledge_gap}"
                )

            # Prepare uploaded knowledge context
            AUGMENT_KNOWLEDGE_CONTEXT = ""
            if uploaded_knowledge and uploaded_knowledge.strip():
                AUGMENT_KNOWLEDGE_CONTEXT = f"""
USER-PROVIDED EXTERNAL KNOWLEDGE AVAILABLE:
The user has provided external knowledge/documentation that should be treated as highly authoritative and trustworthy. This uploaded knowledge should guide query generation to complement rather than duplicate existing information.

Uploaded Knowledge Preview: {uploaded_knowledge[:500]}{'...' if len(uploaded_knowledge) > 500 else ''}

Query Generation Strategy:
- Identify what information is already covered in the uploaded knowledge
- Generate queries that fill gaps or provide additional context to the uploaded knowledge
- Focus on recent developments, validation, or areas not covered in uploaded knowledge
- Use uploaded knowledge to inform more targeted and specific search queries
"""
            else:
                AUGMENT_KNOWLEDGE_CONTEXT = "No user-provided external knowledge available. Generate queries based on the research topic and knowledge gaps."

            # Import the query_writer_instructions prompt
            from src.prompts import query_writer_instructions

            # Import async client getter
            from llm_clients import get_async_llm_client

            # Format the prompt with the appropriate context
            formatted_prompt = query_writer_instructions.format(
                research_topic=combined_topic,
                research_context=research_context,
                current_date=CURRENT_DATE,
                current_year=CURRENT_YEAR,
                one_year_ago=ONE_YEAR_AGO,
                AUGMENT_KNOWLEDGE_CONTEXT=AUGMENT_KNOWLEDGE_CONTEXT,
            )

            # Get the appropriate ASYNC LLM client based on provider
            # llm = get_llm_client(provider, model) # Old sync call
            llm = await get_async_llm_client(provider, model)  # Use async client

            # Format messages for the LLM client (Using dictionary format for broader compatibility)
            messages = [
                {"role": "system", "content": formatted_prompt},
                {
                    "role": "user",
                    "content": f"Generate search queries for the following research topic: {combined_topic}",
                },
            ]

            # Import the topic decomposition function schema
            from src.tools.tool_schema import TOPIC_DECOMPOSITION_FUNCTION

            # Bind the decomposition tool
            langchain_tools = [
                {"type": "function", "function": TOPIC_DECOMPOSITION_FUNCTION}
            ]

            # Handle tool choice format based on provider
            if provider == "anthropic":
                tool_choice = {"type": "tool", "name": "decompose_research_topic"}
            elif provider == "google":
                # Use string format for Google Generative AI
                tool_choice = "decompose_research_topic"
            elif provider == "openai":
                # Use standard dictionary format for OpenAI
                tool_choice = {
                    "type": "function",
                    "function": {"name": "decompose_research_topic"},
                }
            else:
                # Default or fallback - assuming OpenAI-like structure might work for some
                # or default to no specific tool choice if unsure.
                # For now, let's default to the OpenAI format, but log a warning.
                self.logger.warning(
                    f"[MasterAgent] Using default OpenAI tool_choice format for provider '{provider}'. This might not be correct."
                )
                tool_choice = {
                    "type": "function",
                    "function": {"name": "decompose_research_topic"},
                }

            llm_with_tool = llm.bind_tools(
                tools=langchain_tools, tool_choice=tool_choice
            )

            # Call LLM API with function calling ASYNCHRONOUSLY
            model_str = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
            self.logger.info(
                f"[MasterAgent] Making ASYNC tool call with model {model_str} to decompose topic..."
            )
            response = await llm_with_tool.ainvoke(messages)  # Use await and ainvoke
            self.logger.info(
                f"[MasterAgent] Raw ASYNC LLM response for decomposition: {response}"
            )

            # Process the response (consistent across providers due to Langchain abstraction)
            function_args = None  # Initialize to None
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_call = response.tool_calls[0]
                self.logger.info(
                    f"[MasterAgent] Extracted tool call: {tool_call}"
                )  # DEBUG LOG
                try:
                    # Standard Langchain way: access parsed args directly
                    if hasattr(tool_call, "args") and isinstance(tool_call.args, dict):
                        function_args = tool_call.args
                        self.logger.info(
                            f"[MasterAgent] Using pre-parsed args (from tool_call.args): {function_args}"
                        )
                    # Fallback for string arguments (less common)
                    elif (
                        hasattr(tool_call, "function")
                        and hasattr(tool_call.function, "arguments")
                        and isinstance(tool_call.function.arguments, str)
                    ):
                        raw_args = tool_call.function.arguments
                        self.logger.warning(
                            f"[MasterAgent] Received string arguments, attempting JSON parse: {raw_args}"
                        )
                        function_args = json.loads(raw_args)
                    else:
                        # Handle unexpected structures, including the dict case if needed
                        if (
                            isinstance(tool_call, dict)
                            and "args" in tool_call
                            and isinstance(tool_call["args"], dict)
                        ):
                            function_args = tool_call["args"]
                            self.logger.info(
                                f"[MasterAgent] Using args from dict key (standard structure): {function_args}"
                            )
                        else:
                            self.logger.warning(
                                f"[MasterAgent] Tool call structure not recognized or args not found/parsed: {tool_call}"
                            )

                except json.JSONDecodeError as e:
                    self.logger.error(
                        f"[MasterAgent] Failed to parse JSON arguments: {getattr(locals(), 'raw_args', 'N/A')} - Error: {e}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"[MasterAgent] Error processing tool call arguments: {e}"
                    )

            if function_args:
                self.logger.info(
                    f"[MasterAgent] Parsed function_args: {function_args}"
                )  # DEBUG LOG
            else:
                # Fallback if no tool call or parsing failed
                self.logger.warning(
                    "[MasterAgent] No valid tool call arguments found. Falling back to simple search."
                )
                function_args = {
                    "topic_complexity": "simple",
                    "simple_topic": {
                        "query": query,
                        "aspect": "general information",
                        "rationale": "Fallback: No tool call or argument parsing failed",
                        "suggested_tool": "general_search",
                    },
                }

            # Process function call result based on topic complexity
            if function_args.get("topic_complexity") == "simple":
                simple_topic = function_args.get("simple_topic", {})
                return {
                    "topic_complexity": "simple",
                    "query": simple_topic.get("query", query),
                    "aspect": simple_topic.get("aspect", "general information"),
                    "rationale": simple_topic.get("rationale", ""),
                    "suggested_tool": simple_topic.get(
                        "suggested_tool", "general_search"
                    ),
                }
            elif function_args.get("topic_complexity") == "complex":
                complex_topic = function_args.get("complex_topic", {})
                self.logger.info(
                    f"[MasterAgent] Topic decomposed into {len(complex_topic.get('subtopics', []))} subtopics."
                )  # Added log
                return {
                    "topic_complexity": "complex",
                    "main_query": complex_topic.get("main_query", query),
                    "main_tool": complex_topic.get("main_tool", "general_search"),
                    "subtopics": complex_topic.get("subtopics", []),
                }
            else:
                # Fallback if the function call didn't return expected format
                self.logger.warning(
                    f"[MasterAgent] Function call did not return expected format (complexity: {function_args.get('topic_complexity')}). Falling back to simple topic."
                )
                return {
                    "topic_complexity": "simple",
                    "query": query,
                    "aspect": "general information",
                    "rationale": "Fallback due to unexpected function call format",
                    "suggested_tool": "general_search",
                }

        except Exception as e:
            self.logger.error(f"[MasterAgent] Error in topic decomposition: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                "topic_complexity": "simple",
                "query": query,
                "aspect": "general information",
                "rationale": "Error during execution",
                "suggested_tool": "general_search",
            }

    async def plan_research(
        self, query, knowledge_gap, research_loop_count, uploaded_knowledge=None
    ):
        """
        Create a research plan based on the topic decomposition.

        Args:
            query: The main research query or topic
            knowledge_gap: Additional context about knowledge gaps to address
            research_loop_count: Current iteration of research loop
            uploaded_knowledge: User-provided external knowledge (optional)

        Returns:
            Dict containing the research plan with tasks for specialized agents
        """
        # Decompose the topic
        topic_info = await self.decompose_topic(
            query, knowledge_gap, research_loop_count, uploaded_knowledge
        )

        # Log decomposition in benchmark mode
        if hasattr(self, "state") and getattr(self.state, "benchmark_mode", False):
            print(f"[plan_research] Benchmark mode: Question decomposition")
            if "subtopics" in topic_info:
                print(f"[plan_research] Identified subtopics:")
                for i, subtopic in enumerate(topic_info.get("subtopics", [])):
                    print(f"  {i+1}. {subtopic}")
            if "key_entities" in topic_info:
                print(f"[plan_research] Key entities:")
                for entity in topic_info.get("key_entities", []):
                    print(f"  - {entity}")

        # Create a research plan based on the topic complexity
        if topic_info.get("topic_complexity") == "complex":
            # For complex topics, create a plan with subtasks
            subtasks = []

            # Add search tasks based on subtopics
            for i, subtopic in enumerate(topic_info.get("subtopics", [])):
                subtasks.append(
                    {
                        "index": i,
                        "type": "search",
                        "query": subtopic,
                        "description": f"Research information about {subtopic}",
                        "source_type": topic_info.get(
                            "recommended_sources", ["general"]
                        )[
                            min(
                                i,
                                len(topic_info.get("recommended_sources", ["general"]))
                                - 1,
                            )
                        ],
                    }
                )
            # Skip visualization tasks if in benchmark mode
            if not (
                hasattr(self, "state")
                and (
                    getattr(self.state, "benchmark_mode", False)
                    or getattr(self.state, "visualization_disabled", False)
                )
            ):
                # Add visualization tasks based on recommended visualizations
                for i, viz in enumerate(
                    topic_info.get("recommended_visualizations", [])
                ):
                    search_task_index = viz.get("search_task_index")
                    if search_task_index is not None:
                        viz_task = {
                            "index": len(subtasks),
                            "type": "visualization",
                            "description": viz.get(
                                "description",
                                f"Create visualization based on subtopic {search_task_index}",
                            ),
                            "visualization_type": viz.get("type", "default"),
                            "search_task_index": search_task_index,
                        }
                        subtasks.append(viz_task)

            # Create the plan
            plan = {
                "title": topic_info.get("title", query),
                "description": topic_info.get(
                    "description", "A comprehensive research plan"
                ),
                "subtasks": subtasks,
                "subtopics": topic_info.get("subtopics", []),
            }

        else:
            # For simple topics, create a basic plan with a single search task
            search_query = topic_info.get("search_query", query)
            plan = {
                "title": topic_info.get("title", query),
                "description": topic_info.get("description", "A simple research plan"),
                "subtasks": [
                    {
                        "index": 0,
                        "type": "search",
                        "query": search_query,
                        "description": f"Research information about {search_query}",
                        "source_type": topic_info.get(
                            "recommended_sources", ["general"]
                        )[0],
                    }
                ],
                "subtopics": [search_query],
            }

            # Skip visualization tasks if in benchmark mode or visualization mode is disabled
            if not (
                hasattr(self, "state")
                and (
                    getattr(self.state, "benchmark_mode", False)
                    or getattr(self.state, "visualization_disabled", False)
                )
            ):
                # Add a generic visualization task for simple topics
                if topic_info.get("recommended_visualizations"):
                    viz = topic_info.get("recommended_visualizations")[0]
                    viz_task = {
                        "index": 1,
                        "type": "visualization",
                        "description": viz.get(
                            "description", "Create visualization of the topic"
                        ),
                        "visualization_type": viz.get("type", "default"),
                        "search_task_index": 0,  # Reference the single search task
                    }
                    plan["subtasks"].append(viz_task)

        # Log the plan
        subtask_count = len(plan.get("subtasks", []))
        search_task_count = len(
            [t for t in plan.get("subtasks", []) if t.get("type") == "search"]
        )
        visualization_task_count = len(
            [t for t in plan.get("subtasks", []) if t.get("type") == "visualization"]
        )

        self.logger.info(
            f"[MasterAgent] Created research plan with {subtask_count} subtasks:"
        )
        self.logger.info(f"[MasterAgent] - {search_task_count} search tasks")
        self.logger.info(
            f"[MasterAgent] - {visualization_task_count} visualization tasks"
        )

        if hasattr(self, "state") and getattr(self.state, "benchmark_mode", False):
            print(f"[plan_research] Benchmark mode: Research plan created")
            print(f"[plan_research] Plan contains {search_task_count} search tasks")
            if search_task_count > 0:
                print(f"[plan_research] Search queries:")
                for i, task in enumerate(
                    [t for t in plan.get("subtasks", []) if t.get("type") == "search"]
                ):
                    print(
                        f"  {i+1}. \"{task.get('query')}\" (source: {task.get('source_type', 'general')})"
                    )

        return plan

    async def execute_research(self, state, callbacks=None):
        """
        Execute research tasks based on the research plan.
        """
        # Store state for access by all methods in this execution
        self.state = state

        try:
            # Initialize visualization tracking variables
            viz_tasks_created_this_loop = 0
            total_viz_in_state = (
                len(state.visualizations)
                if hasattr(state, "visualizations") and state.visualizations
                else 0
            )
            # 5 or 1
            max_viz_for_this_loop = self._get_max_viz_for_loop(
                state.research_loop_count
            )

            # Always get query, knowledge_gap, and loop_count from state for planning
            # Use refined search_query if available, otherwise the original research_topic
            query = (
                state.search_query
                if hasattr(state, "search_query") and state.search_query
                else state.research_topic
            )
            knowledge_gap = getattr(state, "knowledge_gap", "")
            research_loop_count = getattr(state, "research_loop_count", 0)
            uploaded_knowledge = getattr(state, "uploaded_knowledge", None)
            self.logger.info(
                f"[MasterAgent.execute_research] Planning with query: '{query[:100]}...'"
            )

            # Log uploaded knowledge availability
            if uploaded_knowledge and uploaded_knowledge.strip():
                self.logger.info(
                    f"[MasterAgent.execute_research] Uploaded knowledge available: {len(uploaded_knowledge)} characters"
                )
            else:
                self.logger.info(
                    "[MasterAgent.execute_research] No uploaded knowledge available"
                )

            research_plan = None
            if (
                hasattr(state, "research_plan")
                and state.research_plan
                and research_loop_count == 0
            ):
                # Only use existing plan if it's the very first loop AND a plan was somehow pre-loaded
                # Otherwise, always re-plan to use the latest query/knowledge_gap
                research_plan = state.research_plan
                print(
                    "[execute_research] Using pre-existing research plan for initial loop"
                )

            # If no pre-existing plan for loop 0, or if it's a subsequent loop, always (re)plan.
            if not research_plan or research_loop_count > 0:
                if research_loop_count > 0:
                    print(
                        f"[execute_research] Re-planning research for loop {research_loop_count}."
                    )
                else:
                    print("[execute_research] Creating initial research plan.")

                research_plan = await self.plan_research(
                    query, knowledge_gap, research_loop_count, uploaded_knowledge
                )

                # Add/update the research plan in state if possible
                try:
                    state.research_plan = research_plan
                    print("[execute_research] Updated research plan in state")
                except (ValueError, AttributeError):
                    print(
                        "[execute_research] Unable to update research plan in state, using local copy"
                    )

            # Execute search tasks
            search_results_list = await self._execute_search_tasks(
                research_plan, state
            )  # Renamed to avoid conflict
            successful_search_indices = [
                i
                for i, result in enumerate(search_results_list)
                if result.get("success", False)
            ]

            self.logger.info(
                f"[MasterAgent] Completed {len(successful_search_indices)} successful search tasks out of {len(search_results_list)}."
            )

            # Initialize lists to store visualization outputs from this loop
            visualizations_generated_this_loop = []
            base64_images_generated_this_loop = []
            code_snippets_generated_this_loop = []

            # Skip visualization tasks in QA mode or benchmark mode
            if state.qa_mode or state.benchmark_mode:
                mode_name = "QA mode" if state.qa_mode else "benchmark mode"
                self.logger.info(
                    f"[MasterAgent] Skipping visualization tasks because {mode_name} is enabled."
                )
            elif state.visualization_disabled:
                self.logger.info(
                    "[MasterAgent] Skipping visualization tasks because visualization mode is disabled."
                )
            # do visualization tasks
            else:
                self.logger.info(
                    f"[MasterAgent] Attempting to generate visualizations. Max for this loop: {max_viz_for_this_loop}"
                )
                visualization_agent = VisualizationAgent(self.config)

                for i, search_result_item in enumerate(search_results_list):
                    if not search_result_item.get("success", False):
                        continue  # Skip failed search tasks

                    if viz_tasks_created_this_loop >= max_viz_for_this_loop:
                        self.logger.info(
                            f"[MasterAgent] Reached max visualizations ({max_viz_for_this_loop}) for this loop."
                        )
                        break

                    self.logger.info(
                        f"[MasterAgent] Considering search result index {search_result_item.get('index')} for visualization."
                    )

                    # Pass the actual search result content to determine_visualization_needs
                    # search_result_item is already a dict like {'index': ..., 'query': ..., 'success': ..., 'content': ..., 'sources': ..., 'error': ...}
                    vis_needs = await visualization_agent.determine_visualization_needs(
                        search_result_item
                    )

                    if vis_needs and vis_needs.get("visualization_needed"):
                        self.logger.info(
                            f"[MasterAgent] Visualization needed for search result index {search_result_item.get('index')}. Rationale: {vis_needs.get('rationale')}"
                        )
                        code_data = (
                            await visualization_agent.generate_visualization_code(
                                search_result_item, vis_needs
                            )
                        )

                        if code_data and code_data.get("code"):
                            execution_output = (
                                await visualization_agent.execute_visualization_code(
                                    code_data
                                )
                            )

                            if execution_output and not execution_output.get("error"):
                                viz_tasks_created_this_loop += 1
                                # execution_output["results"] is a list of dicts like:
                                # [{"type": "image", "filepath": "path/to/img.png", "filename": "img.png", "description": "desc", "format": "png", "data": "base64data", "src": "datauri"}]
                                generated_visualizations = execution_output.get(
                                    "results", []
                                )
                                visualizations_generated_this_loop.extend(
                                    generated_visualizations
                                )

                                # Extract base64 data for state
                                for viz_item_detail in generated_visualizations:
                                    if viz_item_detail.get(
                                        "data"
                                    ) and viz_item_detail.get("filename"):
                                        base64_images_generated_this_loop.append(
                                            {
                                                "filename": viz_item_detail.get(
                                                    "filename"
                                                ),
                                                "title": viz_item_detail.get(
                                                    "description"
                                                )
                                                or viz_item_detail.get(
                                                    "filename"
                                                ),  # Use description or filename as title
                                                "base64_data": viz_item_detail.get(
                                                    "data"
                                                ),
                                                "format": viz_item_detail.get(
                                                    "format", "png"
                                                ),
                                            }
                                        )
                                # Collect code snippets
                                if execution_output.get("code_snippets"):
                                    code_snippets_generated_this_loop.extend(
                                        execution_output.get("code_snippets", [])
                                    )
                                elif code_data.get(
                                    "code_snippets"
                                ):  # Fallback to code_data if not in execution_output
                                    code_snippets_generated_this_loop.extend(
                                        code_data.get("code_snippets", [])
                                    )

                                self.logger.info(
                                    f"[MasterAgent] Successfully generated {len(generated_visualizations)} visualizations for search result index {search_result_item.get('index')}."
                                )
                            elif execution_output and execution_output.get("error"):
                                self.logger.error(
                                    f"[MasterAgent] Error executing visualization code for search result index {search_result_item.get('index')}: {execution_output.get('error')}"
                                )
                                # Collect code snippets even on error
                                if execution_output.get("code_snippets"):
                                    code_snippets_generated_this_loop.extend(
                                        execution_output.get("code_snippets", [])
                                    )
                                elif code_data and code_data.get("code_snippets"):
                                    code_snippets_generated_this_loop.extend(
                                        code_data.get("code_snippets", [])
                                    )
                        else:
                            self.logger.info(
                                f"[MasterAgent] No visualization code generated for search result index {search_result_item.get('index')}."
                            )
                    else:
                        self.logger.info(
                            f"[MasterAgent] No visualization deemed necessary for search result index {search_result_item.get('index')}. Rationale: {vis_needs.get('rationale') if vis_needs else 'No vis_needs determined'}"
                        )

            self.logger.info(
                f"[MasterAgent] Generated {viz_tasks_created_this_loop} visualization(s) in this loop. Total viz in state (before this loop): {total_viz_in_state}"
            )

            # Prepare the return dictionary
            # The primary output is still the search_results_list
            # We add new keys for the visualization outputs from this loop.
            return_value = {
                "web_research_results": search_results_list,  # This is what subsequent nodes expect
                "visualizations_generated_this_loop": visualizations_generated_this_loop,
                "base64_images_generated_this_loop": base64_images_generated_this_loop,
                "code_snippets_generated_this_loop": code_snippets_generated_this_loop,
                # Ensure other state fields are preserved if they were part of the input `state`
                # This is crucial if `execute_research` is expected to return a full state-like dict
                # For now, focusing on returning the direct outputs of this agent's actions.
            }
            # Preserve essential state fields that might have been updated by planning
            if hasattr(state, "research_plan"):
                return_value["research_plan"] = state.research_plan

            return return_value

        except Exception as e:
            self.logger.error(f"[MasterAgent] Error in research execution: {str(e)}")
            self.logger.error(f"[MasterAgent] {traceback.format_exc()}")
            # Return search_results_list even in case of error to allow flow to continue if possible
            # And add error information.
            return {
                "web_research_results": getattr(
                    self, "search_results_list", []
                ),  # search_results_list might not be defined if error is early
                "error": f"MasterAgent execution failed: {str(e)}",
                "visualizations_generated_this_loop": [],
                "base64_images_generated_this_loop": [],
                "code_snippets_generated_this_loop": [],
            }

    def _get_max_viz_for_loop(self, research_loop_count):
        """
        Determine the maximum number of visualizations allowed for a given research loop.

        Args:
            research_loop_count: Current research loop iteration

        Returns:
            int: Maximum number of visualizations allowed for this loop
        """
        if research_loop_count == 0:
            # Allow more visualizations in the initial loop
            return 5  # Up to 5 in first loop
        else:
            # Allow fewer visualizations in subsequent loops
            return 1  # Only 1 in later loops

    async def _execute_search_tasks(self, research_plan, state):
        """
        Execute search tasks from the research plan.

        Args:
            research_plan: Dictionary containing the research plan
            state: Current state object

        Returns:
            list: Results from search tasks
        """
        # Start timer for performance logging
        search_start_time = time.time()

        # State is accessible via self.state set in execute_research

        # Track if we're in benchmark mode
        benchmark_mode = getattr(state, "benchmark_mode", False)
        if benchmark_mode:
            print(f"[_execute_search_tasks] Executing search tasks in benchmark mode")

        visualization_disabled = getattr(state, "visualization_disabled", False)
        if visualization_disabled:
            print(
                f"[_execute_search_tasks] Executing search tasks in visualization disabled mode"
            )

        # Initialize specialized agents and tools
        search_agent = SearchAgent(self.config)

        # Initialize task results list to track what's completed
        task_results = []

        # If there are no research tasks, exit early
        if "subtasks" not in research_plan or not research_plan["subtasks"]:
            print("No search tasks found in research plan.")
            return task_results

        # Loop through each search task
        for task in research_plan["subtasks"]:
            if task.get("type") == "search":
                task_index = task.get("index", 0)
                task_query = task.get("query", {})

                # Handle both string and dict query formats
                if isinstance(task_query, str):
                    # If query is a string, use it directly
                    query_text = task_query
                    tool_name = task.get("source_type", "general_search")
                    # Map source_type to tool names
                    if tool_name == "general":
                        tool_name = "general_search"
                    elif tool_name == "academic":
                        tool_name = "academic_search"
                    elif tool_name == "github":
                        tool_name = "github_search"
                    elif tool_name == "linkedin":
                        tool_name = "linkedin_search"
                    else:
                        tool_name = "general_search"  # Default fallback
                elif isinstance(task_query, dict):
                    # If query is a dict, extract the query text and tool
                    query_text = task_query.get("query", "")
                    tool_name = task_query.get("suggested_tool", "general_search")
                else:
                    # Fallback for unexpected format
                    query_text = str(task_query)
                    tool_name = "general_search"

                # Log this to help with tracing
                print(f"Executing search task {task_index} with query: '{query_text}'")
                print(f"Using search tool: {tool_name}")

                try:
                    # Execute the search based on the tool_name
                    search_result = None
                    if tool_name == "general_search":
                        search_result = await search_agent.general_search(query_text)
                    elif tool_name == "academic_search":
                        search_result = await search_agent.academic_search(query_text)
                    elif tool_name == "github_search":
                        search_result = await search_agent.github_search(query_text)
                    elif tool_name == "linkedin_search":
                        search_result = await search_agent.linkedin_search(query_text)
                    else:
                        # Default to general search if tool is unknown
                        search_result = await search_agent.general_search(query_text)

                    # Extract sources from search_result for easy access in results
                    sources = []

                    # Handle different search_result formats
                    if isinstance(search_result, dict):
                        # Format from newer search tools that return a dict with 'formatted_sources'
                        if "formatted_sources" in search_result:
                            formatted_sources = search_result.get(
                                "formatted_sources", []
                            )
                            # Extract source details from formatted_sources
                            for src in formatted_sources:
                                if isinstance(src, str) and " : " in src:
                                    # Parse title and URL from source string (format: "title : url")
                                    parts = src.split(" : ", 1)
                                    if len(parts) == 2:
                                        title, url = parts
                                        sources.append({"title": title, "url": url})

                        # If search_result directly contains 'sources' key with structured data
                        if "sources" in search_result and isinstance(
                            search_result["sources"], list
                        ):
                            for src in search_result["sources"]:
                                if (
                                    isinstance(src, dict)
                                    and "title" in src
                                    and "url" in src
                                ):
                                    sources.append(
                                        {"title": src["title"], "url": src["url"]}
                                    )

                        # Extract content from search_result
                        if "content" in search_result:
                            content = search_result["content"]
                        elif "raw_contents" in search_result:
                            # Join multiple raw contents into a single string
                            raw_contents = search_result.get("raw_contents", [])
                            if isinstance(raw_contents, list):
                                content = "\n\n".join(
                                    [str(item) for item in raw_contents if item]
                                )
                            else:
                                content = str(raw_contents)
                        else:
                            # Fallback: convert the entire result to a string
                            content = str(search_result)
                    else:
                        # Fallback for unexpected search_result type
                        content = str(search_result)
                        # No sources can be extracted in this case

                    # Build the result object with both content and sources
                    task_result = {
                        "index": task_index,
                        "query": query_text,  # Store the actual query text, not the original task_query
                        "success": True,
                        "content": content,
                        "sources": sources,
                        "error": None,
                    }

                    print(
                        f"Search task {task_index} completed with {len(sources)} sources"
                    )
                    task_results.append(task_result)

                except Exception as e:
                    # Log the error and continue with other tasks
                    print(f"Error in search task {task_index}: {str(e)}")
                    error_result = {
                        "index": task_index,
                        "query": query_text,  # Store the actual query text, not the original task_query
                        "success": False,
                        "content": "",
                        "sources": [],
                        "error": str(e),
                    }
                    task_results.append(error_result)

        # Log performance
        search_end_time = time.time()
        search_duration = search_end_time - search_start_time
        print(f"All search tasks completed in {search_duration:.2f} seconds")
        print(f"Total search tasks completed: {len(task_results)}")

        return task_results


class SearchAgent:
    """
    Specialized agent for executing search queries.

    This agent is responsible for executing specific search tasks
    using the appropriate search tool based on the subtopic.
    """

    def __init__(self, config=None):
        """
        Initialize the Search Agent.

        Args:
            config: Configuration object containing search settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def general_search(self, query):
        """Execute a general search query"""
        from src.graph import ToolRegistry, ToolExecutor

        # Debug print statement
        print(f"[DEBUG] SearchAgent.general_search called with query: {query}")
        self.logger.info(f"SearchAgent.general_search called with query: {query}")

        # Create tool registry and executor
        tool_registry = ToolRegistry(self.config)
        tool_executor = ToolExecutor(tool_registry)

        # Execute the search
        return await tool_executor.execute_tool(
            tool_name="general_search", params={"query": query, "top_k": 5}
        )

    async def academic_search(self, query):
        """Execute an academic search query"""
        from src.graph import ToolRegistry, ToolExecutor

        # Debug print statement
        print(f"[DEBUG] SearchAgent.academic_search called with query: {query}")
        self.logger.info(f"SearchAgent.academic_search called with query: {query}")

        # Create tool registry and executor
        tool_registry = ToolRegistry(self.config)
        tool_executor = ToolExecutor(tool_registry)

        # Execute the search
        return await tool_executor.execute_tool(
            tool_name="academic_search", params={"query": query, "top_k": 5}
        )

    async def github_search(self, query):
        """Execute a GitHub search query"""
        from src.graph import ToolRegistry, ToolExecutor

        # Debug print statement
        print(f"[DEBUG] SearchAgent.github_search called with query: {query}")
        self.logger.info(f"SearchAgent.github_search called with query: {query}")

        # Create tool registry and executor
        tool_registry = ToolRegistry(self.config)
        tool_executor = ToolExecutor(tool_registry)

        # Execute the search
        return await tool_executor.execute_tool(
            tool_name="github_search", params={"query": query, "top_k": 5}
        )

    async def linkedin_search(self, query):
        """Execute a LinkedIn search query"""
        from src.graph import ToolRegistry, ToolExecutor

        # Debug print statement
        print(f"[DEBUG] SearchAgent.linkedin_search called with query: {query}")
        self.logger.info(f"SearchAgent.linkedin_search called with query: {query}")

        # Create tool registry and executor
        tool_registry = ToolRegistry(self.config)
        tool_executor = ToolExecutor(tool_registry)

        # Execute the search
        return await tool_executor.execute_tool(
            tool_name="linkedin_search", params={"query": query, "top_k": 5}
        )

    async def execute(self, subtask, tool_executor=None):
        """
        Execute a search for a specific subtask asynchronously using the provided tool executor.
        This method directly uses the query and tool name from the subtask description
        without making an additional LLM call.

        Args:
            subtask: Dict containing task details (query, tool, name, aspect, etc.)
            tool_executor: An initialized ToolExecutor instance.

        Returns:
            Dict containing the search results or an error.
        """
        # Import necessary components
        from src.graph import ToolRegistry, ToolExecutor  # Keep ToolExecutor import
        import traceback

        # Ensure tool_executor is provided (as it's essential now)
        if tool_executor is None:
            self.logger.error("[SearchAgent] ToolExecutor instance must be provided.")
            # Consider raising an error or handling appropriately
            # For now, let's try creating one, but this indicates a potential design issue
            try:
                from src.graph import ToolRegistry

                tool_registry = ToolRegistry(self.config)
                tool_executor = ToolExecutor(tool_registry)
                self.logger.warning(
                    "[SearchAgent] Created a default ToolExecutor instance as none was provided."
                )
            except Exception as te_err:
                self.logger.error(
                    f"[SearchAgent] Failed to create default ToolExecutor: {te_err}"
                )
                return {
                    "error": "ToolExecutor instance required but not provided and creation failed."
                }

        # Directly extract parameters from the subtask dictionary
        query = subtask.get("query")
        # Default to general_search if not specified in subtask
        tool_name = subtask.get("suggested_tool", subtask.get("tool", "general_search"))
        subtask_name = subtask.get("name", "Unnamed subtask")

        # Validate extracted parameters
        if not query:
            self.logger.error(
                f"[SearchAgent] Subtask '{subtask_name}' is missing a query."
            )
            return {"error": f"Subtask '{subtask_name}' missing query."}
        if not tool_name:
            self.logger.error(
                f"[SearchAgent] Subtask '{subtask_name}' is missing a tool name."
            )
            # Defaulting again just in case, but log the error
            tool_name = "general_search"

        self.logger.info(
            f"[SearchAgent] Directly executing tool '{tool_name}' for subtask '{subtask_name}' with query: '{query}'"
        )

        try:
            # --- LLM Call Removed ---
            # No need to call LLM again to determine parameters; use directly from subtask.

            # Define standard parameters (can be extended if MasterAgent provides more)
            # Example: if MasterAgent decided top_k based on decomposition:
            # top_k = subtask.get("parameters", {}).get("top_k", 5)
            params = {
                "query": query,
                "top_k": 5,  # Using a default, could be passed in subtask if needed
            }

            # Execute search with the specified tool asynchronously using ToolExecutor
            self.logger.info(
                f"[SearchAgent] Calling tool_executor.execute_tool with tool_name='{tool_name}', params={params}"
            )
            search_result = await tool_executor.execute_tool(
                tool_name=tool_name, params=params
            )

            # Ensure search_result is a dictionary (ToolExecutor should ideally ensure this)
            if not isinstance(search_result, dict):
                self.logger.warning(
                    f"[SearchAgent] ToolExecutor result is not a dictionary: {type(search_result)}. Converting."
                )
                search_result = {
                    "formatted_sources": str(search_result),
                    "search_string": query,
                    "domains": [],
                }

            # Add metadata to the result
            search_result["subtask"] = subtask  # Include original subtask for context
            search_result["tool_used"] = tool_name  # Log the tool used
            # No "tool_used_llm" as LLM didn't choose the tool here

            self.logger.info(
                f"[SearchAgent] Successfully executed tool '{tool_name}' for subtask '{subtask_name}'"
            )
            return search_result

        except Exception as e:
            self.logger.error(
                f"[SearchAgent] Error executing tool '{tool_name}' for subtask '{subtask_name}': {str(e)}"
            )
            self.logger.error(traceback.format_exc())

            # Return error result
            return {
                "error": str(e),
                "formatted_sources": f"Error executing search: {str(e)}",
                "search_string": query,
                "domains": [],
                "subtask": subtask,  # Include original subtask even on error
                "tool_used": tool_name,
            }


class ResultCombiner:
    """
    Agent for combining results from multiple specialized agents.

    This agent takes the outputs from various specialized agents
    and combines them into a cohesive research result.
    """

    def __init__(self, config=None):
        """
        Initialize the Result Combiner.

        Args:
            config: Configuration object containing combining settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def combine(
        self,
        research_plan,
        subtask_results,
        query,
        research_loop_count,
        original_research_topic,
        current_state=None,
    ):
        """
        Combine results from multiple specialized agents.

        Args:
            research_plan: The original research plan
            subtask_results: List of results from specialized agents
            query: The current search query being used
            research_loop_count: Current research loop count
            original_research_topic: The original research topic from the user (preserved across iterations)
            current_state: The current state object containing existing sources

        Returns:
            Dict containing the combined research results in a format compatible with the existing codebase
        """
        self.logger.info(
            f"[ResultCombiner] Combining {len(subtask_results)} results for loop {research_loop_count}"
        )

        # --- START FIX: Load existing visualizations from state ---
        existing_visualizations = getattr(current_state, "visualizations", [])
        existing_viz_paths = getattr(current_state, "visualization_paths", [])
        existing_base64 = getattr(current_state, "base64_encoded_images", [])
        self.logger.info(
            f"[ResultCombiner] Loaded {len(existing_visualizations)} existing visualizations, {len(existing_viz_paths)} paths, {len(existing_base64)} base64 images from previous state."
        )

        # Initialize collection variables with existing data
        all_formatted_sources = []  # Formatted sources are typically loop-specific
        all_search_strings = []  # Search strings are loop-specific
        all_tools_used = set(
            getattr(current_state, "tools", [])
        )  # Accumulate tools used
        all_visualizations = (
            existing_visualizations.copy()
        )  # Start with existing visualizations
        # Use sets for paths and filenames to handle duplicates easily
        seen_viz_paths = set(existing_viz_paths)
        # Need a way to uniquely identify base64 images if structure allows (e.g., by filename if present)
        # Using filename for now, assuming base64_encoded_images is a list of dicts with 'filename'
        seen_base64_filenames = set(
            img.get("filename")
            for img in existing_base64
            if isinstance(img, dict) and img.get("filename")
        )
        current_loop_base64_images = []  # Temporarily store base64 for this loop
        # --- END FIX ---

        # DEBUG: Check incoming subtask_results
        self.logger.info(
            f"[ResultCombiner] 🔎 DEBUGGING {len(subtask_results)} subtask results:"
        )
        for i, result in enumerate(subtask_results):
            keys = (
                list(result.keys())
                if isinstance(result, dict)
                else ["result is not a dict"]
            )
            if isinstance(result, dict) and "error" in result:
                self.logger.warning(
                    f"[ResultCombiner] 🔎 Result {i+1}: ERROR={result.get('error')}"
                )
            # Check for visualization results (assuming structure from VisualizationAgent.execute)
            elif (
                isinstance(result, dict)
                and "results" in result
                and isinstance(result["results"], dict)
                and "results" in result["results"]
            ):
                viz_count = len(result["results"].get("results", []))
                self.logger.info(
                    f"[ResultCombiner] 🔎 Result {i+1}: VISUALIZATION with {viz_count} images. Keys: {keys}"
                )
            elif isinstance(result, dict):
                self.logger.info(
                    f"[ResultCombiner] 🔎 Result {i+1}: Regular result with keys: {keys}"
                )
            else:
                self.logger.warning(
                    f"[ResultCombiner] 🔎 Result {i+1}: Unexpected type {type(result)}"
                )

        # Process each subtask result
        subtopic_results = []
        all_code_snippets = []  # Initialize list for code snippets

        # NEW: accumulate raw contents for storing in state.web_research_results
        all_raw_contents = []

        # --- FIX: Process results and append NEW visualizations ---
        new_viz_added_count = 0
        for result in subtask_results:
            if not isinstance(result, dict):  # Skip non-dict results
                continue

            # Skip results with errors
            if "error" in result:
                self.logger.warning(
                    f"[ResultCombiner] Skipping result with error: {result.get('error')}"
                )
                continue

            # Check if this is a visualization result
            # Structure expected: {'search_result': {...}, 'code_data': {...}, 'results': {'results': [...] or 'error': ...}}
            is_visualization_result = (
                "results" in result
                and isinstance(result["results"], dict)
                and "results" in result["results"]
                and isinstance(result["results"]["results"], list)
            )
            visualization_failed = (
                "results" in result
                and isinstance(result["results"], dict)
                and "error" in result["results"]
            )

            if visualization_failed:
                # Use single quotes inside the f-string expression
                self.logger.warning(
                    f"[ResultCombiner] Visualization subtask failed: {result['results'].get('error')}"
                )
                # Still process the underlying search result if it exists
                if "search_result" in result and isinstance(
                    result["search_result"], dict
                ):
                    search_result_data = result["search_result"]
                    if "subtask" in search_result_data:
                        # Process search result components
                        if isinstance(
                            search_result_data.get("formatted_sources"), list
                        ):
                            all_formatted_sources.extend(
                                search_result_data.get("formatted_sources", [])
                            )
                        else:
                            all_formatted_sources.append(
                                search_result_data.get("formatted_sources", "")
                            )
                        all_search_strings.append(
                            search_result_data.get("search_string", "")
                        )
                        all_tools_used.add(
                            search_result_data.get(
                                "tool_used",
                                search_result_data.get("subtask", {}).get(
                                    "tool", "general_search"
                                ),
                            )
                        )
                        subtopic_results.append(
                            {
                                "subtopic": search_result_data.get("subtask", {}),
                                "search_result": search_result_data,
                            }
                        )
                continue  # Skip adding visualization parts for failed task

            if is_visualization_result:
                viz_list = result["results"].get("results", [])
                viz_files_from_this_task = []
                for viz_item in viz_list:
                    filepath = None
                    filename = None
                    description = None  # Initialize description
                    if isinstance(viz_item, dict) and viz_item.get("type") == "image":
                        filepath = viz_item.get("filepath")
                        description = viz_item.get("description")  # Get description
                        if filepath and os.path.exists(filepath):
                            filename = viz_item.get(
                                "filename", os.path.basename(filepath)
                            )
                        else:
                            # Use repr() for safer logging
                            self.logger.warning(
                                f"[ResultCombiner] Invalid/missing viz filepath: {repr(filepath)}"
                            )
                            filepath = None  # Ensure invalid path isn't used

                    if filepath and filename:
                        # Check if this visualization path is already tracked
                        if filepath not in seen_viz_paths:
                            viz_files_from_this_task.append(
                                {
                                    "filepath": filepath,
                                    "filename": filename,
                                    "subtask_name": result.get("code_data", {}).get(
                                        "subtask_name", "Visualization"
                                    ),
                                    "description": description,  # Store description
                                }
                            )
                            seen_viz_paths.add(filepath)  # Track the path
                            new_viz_added_count += 1
                        else:
                            # Use repr() for safer logging
                            self.logger.info(
                                f"[ResultCombiner] Skipping duplicate viz filepath: {repr(filepath)}"
                            )
                    else:
                        # Use repr() for safer logging
                        self.logger.warning(
                            f"[ResultCombiner] Skipping invalid/missing visualization item: {repr(viz_item)}"
                        )

                # Extend the main list with new, unique visualizations from this task
                all_visualizations.extend(viz_files_from_this_task)
                self.logger.info(
                    f"[ResultCombiner] Added {len(viz_files_from_this_task)} NEW unique visualizations from this task."
                )
                # Also handle the base64 encoded images if they exist for this viz task
                # (assuming format_visualizations_html populates self._base64_encoded_images)
                # We need to call format_visualizations_html *for this task's visualizations* temporarily
                # to potentially populate the base64 cache within the combiner instance for this call
                self.format_visualizations_html(
                    viz_files_from_this_task
                )  # This populates self._base64_encoded_images
                if (
                    hasattr(self, "_base64_encoded_images")
                    and self._base64_encoded_images
                ):
                    for img_data in self._base64_encoded_images:
                        if (
                            isinstance(img_data, dict)
                            and img_data.get("filename")
                            and img_data["filename"] not in seen_base64_filenames
                        ):
                            current_loop_base64_images.append(img_data)
                            seen_base64_filenames.add(img_data["filename"])
                    self._base64_encoded_images = (
                        []
                    )  # Clear instance cache for next task

            # Process the underlying search result data (even if part of a viz result)
            search_result_data = None
            if "search_result" in result and isinstance(result["search_result"], dict):
                search_result_data = result["search_result"]
            elif (
                "subtask" in result
            ):  # Handle regular search results not nested under 'search_result'
                search_result_data = result

            if search_result_data and "subtask" in search_result_data:
                subtask = search_result_data.get("subtask", {})
                # Extract components for UI state
                if isinstance(search_result_data.get("formatted_sources"), list):
                    all_formatted_sources.extend(
                        search_result_data.get("formatted_sources", [])
                    )
                else:
                    all_formatted_sources.append(
                        search_result_data.get("formatted_sources", "")
                    )
                all_search_strings.append(search_result_data.get("search_string", ""))
                all_tools_used.add(
                    search_result_data.get(
                        "tool_used", subtask.get("tool", "general_search")
                    )
                )
                subtopic_results.append(
                    {"subtopic": subtask, "search_result": search_result_data}
                )

                # Collect raw_contents
                if "raw_contents" in search_result_data and isinstance(
                    search_result_data["raw_contents"], list
                ):
                    for rc in search_result_data["raw_contents"]:
                        if rc:  # Ensure we don't append empty/None items
                            all_raw_contents.append(rc)
            elif not is_visualization_result and not visualization_failed:
                self.logger.warning(
                    f"[ResultCombiner] Skipping result with unrecognized structure: {result.keys() if isinstance(result, dict) else type(result)}"
                )

            # --- START: Aggregate Code Snippets ---
            # Check direct top-level code_snippets (our new addition for visualizations)
            if (
                isinstance(result.get("code_snippets"), list)
                and result["code_snippets"]
            ):
                all_code_snippets.extend(result["code_snippets"])
                self.logger.info(
                    f"[ResultCombiner] Found and added {len(result['code_snippets'])} code snippets from top-level code_snippets."
                )
            # Check for code_snippets in results object (also for visualizations)
            elif isinstance(result.get("results"), dict) and isinstance(
                result["results"].get("code_snippets"), list
            ):
                snippets = result["results"]["code_snippets"]
                if snippets:
                    all_code_snippets.extend(snippets)
                    self.logger.info(
                        f"[ResultCombiner] Found and added {len(snippets)} code snippets from results.code_snippets."
                    )
            # Check top-level enriched_data (for non-visualization results)
            elif isinstance(result.get("enriched_data"), dict):
                snippets = result["enriched_data"].get("code_snippets")
                if isinstance(snippets, list) and snippets:
                    all_code_snippets.extend(snippets)
                    self.logger.info(
                        f"[ResultCombiner] Found and added {len(snippets)} code snippets from top-level enriched_data."
                    )
            # Check nested enriched_data within the 'results' key (for visualization results)
            elif isinstance(result.get("results"), dict) and isinstance(
                result["results"].get("enriched_data"), dict
            ):
                nested_enriched_data = result["results"]["enriched_data"]
                snippets = nested_enriched_data.get("code_snippets")
                if isinstance(snippets, list) and snippets:
                    all_code_snippets.extend(snippets)
                    self.logger.info(
                        f"[ResultCombiner] Found and added {len(snippets)} code snippets from nested results.enriched_data."
                    )
            # --- END: Aggregate Code Snippets ---

        # --- END FIX ---

        self.logger.info(
            f"[ResultCombiner] Total NEW unique visualizations added in this loop: {new_viz_added_count}"
        )
        self.logger.info(
            f"[ResultCombiner] Accumulated total unique visualizations: {len(all_visualizations)}"
        )
        final_base64_images = existing_base64 + current_loop_base64_images
        self.logger.info(
            f"[ResultCombiner] Accumulated total base64 images: {len(final_base64_images)}"
        )

        # --- EXISTING LOGIC TO PARSE FORMATTED SOURCES ---
        parsed_sources = []
        seen_urls = set()  # To handle potential duplicates from formatted_sources
        for source_str in all_formatted_sources:
            if not isinstance(source_str, str):  # Handle potential non-string items
                self.logger.warning(
                    f"[ResultCombiner] Skipping non-string item in all_formatted_sources: {type(source_str)}"
                )
                continue

            # Attempt to parse the string - adjust regex if format is different
            # Common formats: "1. Title: [Actual Title] (Source: [Actual URL])", "Title: ... (Source: ...)", "[Title](URL)"
            # Regex tries to capture title and URL from common patterns
            match = None
            # Pattern 1: Explicit Title and Source labels
            pattern1 = r"(?:\d+\.\s*)?(?:Title|Name|headline):\s*(.*?)\s*(?:\(Source:|\(URL:|URL:|Source:)\s*(https?://.*?)(?:\)|$)"
            # Pattern 2: Markdown link style
            pattern2 = r"\s*\[(.*?)\]\((https?://.*?)\)"
            # Pattern 3: Simpler Title : URL
            pattern3 = r"\s*(.*?)\s*:\s*(https?://\S+)"

            match = re.search(pattern1, source_str.strip(), re.IGNORECASE)
            if not match:
                match = re.search(pattern2, source_str.strip())
            if not match:
                match = re.search(pattern3, source_str.strip())

            if match:
                # Group 1 is usually title, Group 2 is usually URL across patterns
                title = match.group(1).strip() if match.group(1) else "No Title Found"
                url = (
                    match.group(2).strip()
                    if len(match.groups()) > 1 and match.group(2)
                    else None
                )

                if not url and len(match.groups()) > 0:
                    # Check if URL might be the only capture group in some patterns
                    potential_url = match.group(1).strip()
                    if potential_url.startswith("http"):
                        url = potential_url
                        title = "No Title Found"  # Reset title if URL was in group 1

                if url and url not in seen_urls:
                    parsed_sources.append({"title": title, "url": url})
                    seen_urls.add(url)
                elif url:
                    self.logger.info(
                        f"[ResultCombiner] Skipping duplicate URL found in formatted_sources: {url}"
                    )
                else:
                    self.logger.warning(
                        f"[ResultCombiner] Could not extract URL from formatted source: {source_str}"
                    )

            else:
                self.logger.warning(
                    f"[ResultCombiner] Could not parse formatted source string: {source_str}"
                )

        # Create combined results structure based on topic complexity
        if research_plan.get("topic_complexity") == "complex":
            combined_results = {
                "topic_complexity": "complex",
                "main_query": research_plan.get("main_query", query),
                "main_tool": research_plan.get("main_tool", "general_search"),
                "subtopic_results": subtopic_results,
                "visualizations": all_visualizations,
            }
        else:
            # For simple topics, use a simpler structure
            combined_results = {
                "topic_complexity": "simple",
                "query": query,
                "aspect": (
                    subtask_results[0]
                    .get("subtask", {})
                    .get("aspect", "general information")
                    if subtask_results
                    else "general information"
                ),
                "rationale": "Direct search",
                "suggested_tool": (
                    list(all_tools_used)[0] if all_tools_used else "general_search"
                ),
                "search_result": subtask_results[0] if subtask_results else {},
                "visualizations": all_visualizations,
            }

        # --- BUILD OUTPUTS FROM PARSED SOURCES ---
        # Get existing citations if available
        existing_source_citations = {}
        if current_state is not None:
            existing_source_citations = getattr(current_state, "source_citations", {})

        # Create new citations dictionary, starting with existing citations
        citations = existing_source_citations.copy()
        sources_gathered = []
        final_domains = []  # Use parsed_sources for the 'domains' output field

        # Get existing sources to check for duplicates
        existing_sources = []
        if current_state is not None:
            existing_sources = getattr(current_state, "sources_gathered", [])
        existing_urls = set()

        # Extract URLs from existing sources for faster duplicate checking
        for source in existing_sources:
            if " : " in source:
                url = source.split(" : ", 1)[1].strip()
                existing_urls.add(url)

        # Debug log for deduplication and citations
        self.logger.info(
            f"[ResultCombiner] Found {len(existing_sources)} existing sources with {len(existing_urls)} unique URLs for deduplication"
        )
        self.logger.info(
            f"[ResultCombiner] Starting with {len(existing_source_citations)} existing citations"
        )

        # Determine the next available citation number
        next_citation_num = 1
        if citations:
            # Extract highest existing citation number
            existing_nums = [int(num) for num in citations.keys() if num.isdigit()]
            if existing_nums:
                next_citation_num = max(existing_nums) + 1

        # Add new sources that aren't duplicates
        for source_info in parsed_sources:
            title = source_info.get("title", "N/A")
            url = source_info.get("url", "")

            # Check if URL already exists in existing citations to avoid duplicates
            url_already_in_citations = False
            for citation in citations.values():
                if citation.get("url") == url:
                    url_already_in_citations = True
                    break

            if url and not url_already_in_citations and url not in existing_urls:
                # Add as a new citation with incrementing number
                citations[str(next_citation_num)] = {"title": title, "url": url}
                sources_gathered.append(f"{title} : {url}")
                final_domains.append(
                    {"title": title, "url": url}
                )  # Populate domains with parsed info
                existing_urls.add(
                    url
                )  # Add to existing URLs to prevent duplicates within this batch
                next_citation_num += 1

        self.logger.info(
            f"[ResultCombiner] Finished with {len(citations)} total citations after adding new ones"
        )

        # Format visualizations for inclusion in the report
        # Ensure all_visualizations contains unique items before formatting
        unique_visualizations = []
        seen_paths = set()
        for viz_data in all_visualizations:
            if (
                isinstance(viz_data, dict)
                and viz_data.get("filepath") not in seen_paths
            ):
                unique_visualizations.append(viz_data)
                seen_paths.add(viz_data.get("filepath"))
            elif isinstance(viz_data, str) and viz_data not in seen_paths:
                unique_visualizations.append(
                    viz_data
                )  # Keep strings if they represent valid paths
                seen_paths.add(viz_data)

        self.logger.info(
            f"[ResultCombiner] Processing {len(unique_visualizations)} unique visualizations for HTML embedding."
        )
        visualization_html = self.format_visualizations_html(unique_visualizations)

        # --- START MODIFICATION: Create visualization_paths list ---
        visualization_paths = []
        for viz_data in unique_visualizations:  # Use unique list
            if isinstance(viz_data, str):
                # Direct filepath string
                if os.path.exists(viz_data):
                    visualization_paths.append(viz_data)
            elif isinstance(viz_data, dict) and viz_data.get("filepath"):
                # Dictionary with filepath attribute
                if os.path.exists(viz_data.get("filepath")):
                    visualization_paths.append(viz_data.get("filepath"))

        # DEBUG: Log information about extracted visualization paths
        self.logger.info(
            f"[ResultCombiner] 🖼️ Extracted {len(visualization_paths)} visualization paths from {len(all_visualizations)} visualization entries"
        )
        for i, path in enumerate(visualization_paths):
            self.logger.info(f"[ResultCombiner] 🖼️ Visualization path {i+1}: {path}")
        if not visualization_paths and all_visualizations:
            self.logger.warning(
                f"[ResultCombiner] ⚠️ Found {len(all_visualizations)} visualizations but extracted 0 paths!"
            )
            # Debug the structure of all_visualizations
            for i, viz in enumerate(all_visualizations):
                if isinstance(viz, dict):
                    self.logger.warning(
                        f"[ResultCombiner] ⚠️ Visualization {i+1} keys: {viz.keys() if hasattr(viz, 'keys') else 'No keys (not a dict)'}"
                    )
                    self.logger.warning(
                        f"[ResultCombiner] ⚠️ Visualization {i+1} filepath: {viz.get('filepath') if hasattr(viz, 'get') else 'No get method'}"
                    )
                else:
                    self.logger.warning(
                        f"[ResultCombiner] ⚠️ Visualization {i+1} type: {type(viz)}, value: {viz}"
                    )
        # --- END MODIFICATION ---

        # Return both new and old format results for backward compatibility
        final_result = {
            "research_results": combined_results,
            "citations": citations,  # New citations from parsed sources
            "source_citations": citations,  # Use the same merged citations dictionary for source_citations
            "formatted_sources": all_formatted_sources,  # Keep original formatted strings
            "search_string": " | ".join(all_search_strings),
            "research_topic": original_research_topic,  # Use the original research topic instead of the query
            "research_loop_count": research_loop_count,
            "tools": list(all_tools_used),
            "domains": final_domains,  # New domains list from parsed sources
            # We store the raw contents in web_research_results
            "web_research_results": all_raw_contents,
            "sources_gathered": sources_gathered,  # New sources_gathered from parsed sources
            "selected_search_tool": (
                list(all_tools_used)[-1] if all_tools_used else "general_search"
            ),
            "visualizations": all_visualizations,  # Keep original visualization dicts for potential other uses
            "visualization_paths": visualization_paths,  # <-- ADDED: List of path strings for finalize_report
            "visualization_html": visualization_html,  # HTML string for visualization embedding
            "base64_encoded_images": final_base64_images,  # Include base64 encoded images if they exist
            "code_snippets": all_code_snippets,  # Add aggregated code snippets
        }

        # --- Add debug logging before return ---
        self.logger.info(f"[ResultCombiner] COMBINE FINAL CHECK:")
        self.logger.info(
            f"[ResultCombiner]   - Returning combined_summary length: {len(final_result.get('research_results', {}))}"
        )
        self.logger.info(
            f"[ResultCombiner]   - Returning sources_gathered count: {len(final_result.get('sources_gathered', []))}"
        )
        self.logger.info(
            f"[ResultCombiner]   - Returning source_citations count: {len(final_result.get('source_citations', {}))}"
        )
        self.logger.info(
            f"[ResultCombiner]   - Returning visualization_html length: {len(final_result.get('visualization_html', ''))}"
        )
        self.logger.info(
            f"[ResultCombiner]   - Returning visualizations count: {len(final_result.get('visualizations', []))}"
        )
        self.logger.info(
            f"[ResultCombiner]   - Returning visualization_paths count: {len(final_result.get('visualization_paths', []))}"
        )
        self.logger.info(
            f"[ResultCombiner]   - Returning base64_encoded_images count: {len(final_result.get('base64_encoded_images', []))}"
        )
        self.logger.info(
            f"[ResultCombiner]   - Returning code_snippets count: {len(final_result.get('code_snippets', []))}"
        )  # Log snippet count
        if len(final_result.get("base64_encoded_images", [])) > 0:
            self.logger.info(
                f"[ResultCombiner]   - First base64 image src start: {final_result['base64_encoded_images'][0].get('src', '')[:100]}..."
            )

        return final_result

    def format_visualizations_html(self, visualizations):
        """
        Format visualizations as HTML for inclusion in the research report.

        Args:
            visualizations: List of visualization metadata dictionaries or strings

        Returns:
            HTML string with embedded visualizations
        """
        if not visualizations:
            print(
                "WORKING@@@@@@@@[format_visualizations_html] No visualizations to format"
            )
            return ""

        html_parts = ["<div class='visualizations'>"]
        base64_encoded_images = []

        for viz in visualizations:
            # Handle different types of visualization items
            if isinstance(viz, str):
                # Direct string filepath
                filepath = viz
                filename = os.path.basename(filepath)
                title = "Visualization"
            else:
                # Dictionary with metadata
                filename = viz.get("filename", "")
                filepath = viz.get("filepath", "")
                title = (
                    viz.get("subtask_name", "Visualization").replace("_", " ").title()
                )

            # Check if filepath exists and is not empty
            if not filepath or not os.path.exists(filepath):
                self.logger.warning(
                    f"[ResultCombiner] Visualization file not found: {filepath}"
                )
                continue

            # Convert image to base64 for embedding
            try:
                with open(filepath, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode("utf-8")

                # Determine image format from filename
                img_format = os.path.splitext(filename)[1][1:].lower()
                if img_format not in ["png", "jpg", "jpeg", "gif", "svg"]:
                    img_format = "png"  # Default to png

                # Store base64 data for this image
                base64_encoded_images.append(
                    {
                        "filename": filename,
                        "title": title,
                        "base64_data": img_data,
                        "format": img_format,
                    }
                )

                # Debug: Log base64 data length to help diagnose issues
                self.logger.info(
                    f"[ResultCombiner] Embedding visualization {filepath} with base64 data length: {len(img_data)}"
                )

                # Use data URI for reliable embedding
                html_parts.append(
                    f"""
                <div class='visualization-container'>
                    <h3>{title}</h3>
                    <img src="data:image/{img_format};base64,{img_data}" alt="{title}" style="max-width:100%; height:auto;" />
                </div>
                """
                )
                self.logger.info(
                    f"[ResultCombiner] Successfully embedded visualization: {filepath}"
                )
            except Exception as e:
                self.logger.error(
                    f"[ResultCombiner] Error embedding visualization {filepath}: {str(e)}"
                )
                self.logger.error(traceback.format_exc())

        html_parts.append("</div>")

        # Add CSS for visualization styling
        css = """
        <style>
        .visualizations {
            margin: 20px 0;
        }
        .visualization-container {
            margin-bottom: 30px;
        }
        .visualization-container h3 {
            margin-bottom: 10px;
            font-weight: bold;
        }
        .visualization-container img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            height: auto;
            display: block;
        }
        </style>
        """

        # Store base64 data in a custom attribute for access by finalize_report
        self._base64_encoded_images = base64_encoded_images

        return css + "\n".join(html_parts)
