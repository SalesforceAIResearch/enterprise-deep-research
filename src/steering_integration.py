"""
Integration layer to make research agent read todo.md before each loop
"""

import logging
from typing import Dict, Any, List
from src.simple_steering import ResearchTodoManager

logger = logging.getLogger(__name__)


def integrate_steering_with_research_loop(
    state, search_queries: List[str]
) -> List[str]:
    """
    Called before each research loop to filter/modify search queries based on todo.md

    This is where the magic happens - agent reads its todo.md and adapts!
    """

    if not hasattr(state, "steering_todo") or not state.steering_todo:
        # No steering enabled, return queries as-is
        return search_queries

    todo_manager: ResearchTodoManager = state.steering_todo

    # logger.info(f"[STEERING] Checking todo.md before research loop...")
    # logger.info(f"[STEERING] Current todo version: {todo_manager.todo_version}")

    # Log current todo for debugging
    # current_todo = todo_manager.get_todo_md()
    # logger.info(f"[STEERING] Current todo.md:\n{current_todo}")

    # STEP 1: Filter out duplicate queries first
    # logger.info(f"[STEERING] Input queries: {len(search_queries)}")
    deduplicated_queries = todo_manager.filter_duplicate_queries(search_queries)
    # logger.info(f"[STEERING] After deduplication: {len(deduplicated_queries)}")

    # STEP 2: Filter search queries based on constraints
    filtered_queries = []
    cancelled_queries = []
    boosted_queries = []

    for query in deduplicated_queries:
        # Check if this query should be cancelled
        if todo_manager.should_cancel_search(query):
            cancelled_queries.append(query)
            continue

        # Check if this query should get priority boost
        boost = todo_manager.get_search_priority_boost(query)
        if boost > 0:
            boosted_queries.append((query, boost))

        filtered_queries.append(query)

    # Log what happened
    # if cancelled_queries:
    #     logger.info(
    #         f"[STEERING] Cancelled {len(cancelled_queries)} queries due to todo constraints:"
    #     )
    #     for query in cancelled_queries:
    #         logger.info(f"  - CANCELLED: {query}")

    # if boosted_queries:
    #     logger.info(f"[STEERING] Boosted priority for {len(boosted_queries)} queries:")
    #     for query, boost in boosted_queries:
    #         logger.info(f"  - BOOSTED (+{boost}): {query}")

    # logger.info(
    #     f"[STEERING] Final queries: {len(filtered_queries)}/{len(search_queries)} (filtered {len(cancelled_queries)})"
    # )

    # Mark relevant todo tasks as in-progress
    _mark_relevant_tasks_in_progress(todo_manager, filtered_queries)

    return filtered_queries


def _mark_relevant_tasks_in_progress(
    todo_manager: ResearchTodoManager, queries: List[str]
):
    """Mark todo tasks as in-progress when we start working on them"""

    for task in todo_manager.get_pending_tasks():
        # Check if any query relates to this task
        task_desc = task.description.lower()

        for query in queries:
            query_lower = query.lower()

            # Simple matching - if query relates to task, mark as in-progress
            if any(word in query_lower for word in task_desc.split() if len(word) > 3):
                task.status = task.status.IN_PROGRESS
                # logger.info(f"[STEERING] Started working on task: {task.description}")
                break


def generate_search_queries_with_steering(state, base_topic: str) -> List[str]:
    """
    Generate search queries that respect steering constraints.
    Called when creating new search queries.
    """

    if not hasattr(state, "steering_todo") or not state.steering_todo:
        # No steering, generate normal queries
        return _generate_default_queries(base_topic)

    todo_manager: ResearchTodoManager = state.steering_todo
    constraints = todo_manager.get_current_constraints()

    # logger.info(f"[STEERING] Generating queries with constraints: {constraints}")

    queries = []

    # If we have focus constraints, generate focused queries
    if constraints["focus_on"]:
        for focus_item in constraints["focus_on"]:
            queries.extend(
                [
                    f"{base_topic} {focus_item}",
                    f"{focus_item} {base_topic}",
                    f"{focus_item} information research",
                ]
            )

    # If we have prioritize constraints, generate priority queries
    if constraints["prioritize"]:
        for priority_item in constraints["prioritize"]:
            queries.extend(
                [f"{base_topic} {priority_item}", f"{priority_item} {base_topic}"]
            )

    # If no specific constraints, generate default but avoid excluded terms
    if not constraints["focus_on"] and not constraints["prioritize"]:
        queries = _generate_default_queries(base_topic)

    # Filter out excluded terms
    if constraints["exclude"]:
        filtered_queries = []
        for query in queries:
            exclude_query = any(
                exclude_term.lower() in query.lower()
                for exclude_term in constraints["exclude"]
            )
            if not exclude_query:
                filtered_queries.append(query)
            # else:
            # logger.info(f"[STEERING] Excluded query due to constraints: {query}")
        queries = filtered_queries

    # Remove duplicates and limit
    queries = list(set(queries))[:10]  # Max 10 queries

    # logger.info(f"[STEERING] Generated {len(queries)} steering-aware queries")

    return queries


def _generate_default_queries(topic: str) -> List[str]:
    """Generate default search queries for a topic"""
    return [
        f"{topic} information",
        f"{topic} background",
        f"{topic} overview",
        f"{topic} details",
        f"{topic} recent news",
        f"{topic} research",
    ]


def complete_steering_tasks(state, search_results: List[Dict[str, Any]]):
    """
    Called after search results are obtained to mark relevant todo tasks as complete
    """

    if not hasattr(state, "steering_todo") or not state.steering_todo:
        return

    todo_manager: ResearchTodoManager = state.steering_todo

    # Mark in-progress tasks as completed if we got good results
    for task in todo_manager.tasks.values():
        if task.status == task.status.IN_PROGRESS:
            # Simple heuristic - if we got results, mark as completed
            if search_results and len(search_results) > 0:
                todo_manager.mark_task_completed(task.id)
                # logger.info(f"[STEERING] Completed task: {task.description}")


def get_steering_summary_for_agent(state) -> str:
    """
    Get a summary of current steering instructions for the research agent.
    This can be included in the agent's prompt.
    """

    if not hasattr(state, "steering_todo") or not state.steering_todo:
        return ""

    todo_manager: ResearchTodoManager = state.steering_todo
    constraints = todo_manager.get_current_constraints()

    summary_parts = []

    if constraints["focus_on"]:
        summary_parts.append(f"FOCUS ONLY ON: {', '.join(constraints['focus_on'])}")

    if constraints["exclude"]:
        summary_parts.append(f"EXCLUDE: {', '.join(constraints['exclude'])}")

    if constraints["prioritize"]:
        summary_parts.append(f"PRIORITIZE: {', '.join(constraints['prioritize'])}")

    if constraints["stop_searching"]:
        summary_parts.append(
            f"STOP SEARCHING FOR: {', '.join(constraints['stop_searching'])}"
        )

    if summary_parts:
        return (
            f"\n\n**CURRENT RESEARCH STEERING:**\n"
            + "\n".join(f"- {part}" for part in summary_parts)
            + "\n"
        )

    return ""
