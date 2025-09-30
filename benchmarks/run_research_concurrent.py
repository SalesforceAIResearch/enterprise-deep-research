#!/usr/bin/env python3
"""
Multi-threaded research script that processes queries concurrently.
"""

import asyncio
import argparse
import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import threading
from typing import List, Dict, Optional, Tuple
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("research_concurrent.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Add the deep-research directory to the Python path
script_dir = Path(__file__).parent
deep_research_dir = script_dir.parent
sys.path.insert(0, str(deep_research_dir))

# Load environment variables from the correct .env file
env_file_path = deep_research_dir / ".env"
logger.info(f"Loading environment from: {env_file_path}")
load_dotenv(dotenv_path=env_file_path)

STATS_LOCK = threading.Lock()
GLOBAL_STATS = {
    "completed": 0,
    "failed": 0,
    "start_time": None,
    "tasks_started": 0,
    "tasks_in_progress": 0,
}


class BenchmarkDatasetManager(ABC):
    """Abstract base class for managing different benchmark datasets."""

    @abstractmethod
    def load_queries(
        self,
        file_path: str,
        task_ids: Optional[List[int]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Load queries from the dataset file."""
        pass

    @abstractmethod
    def get_query_field(self) -> str:
        """Return the field name that contains the query text."""
        pass

    @abstractmethod
    def get_processing_config(self) -> Dict:
        """Return the configuration for processing this benchmark."""
        pass

    @abstractmethod
    def format_result(self, task_data: Dict, result_data: Dict) -> Dict:
        """Format the result according to benchmark requirements."""
        pass

    @abstractmethod
    def get_output_filename(self, task_id: str) -> str:
        """Get the output filename for a task."""
        pass


class DRBDatasetManager(BenchmarkDatasetManager):
    """Dataset manager for Deep Research Bench (DRB)."""

    def load_queries(
        self,
        file_path: str,
        task_ids: Optional[List[int]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Load queries from JSONL file."""
        queries = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            query_data = json.loads(line)
                            queries.append(query_data)
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON on line {line_num}: {e}")

            logger.info(f"📋 Loaded {len(queries)} queries from {file_path}")

            # Filter by task IDs if specified
            if task_ids:
                queries = [q for q in queries if q["id"] in task_ids]
                logger.info(f"🎯 Filtered to {len(queries)} specific tasks: {task_ids}")

            # Apply limit if specified
            if limit:
                queries = queries[:limit]
                logger.info(f"🔢 Limited to first {len(queries)} tasks")

            return queries

        except FileNotFoundError:
            logger.error(f"Query file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading queries: {e}")
            raise

    def get_query_field(self) -> str:
        return "prompt"

    def get_processing_config(self) -> Dict:
        return {
            "max_loops_default": 5,
            "benchmark_mode": False,
            "qa_mode": False,
            "visualization_disabled": True,
        }

    def format_result(self, task_data: Dict, result_data: Dict) -> Dict:
        """Format result for DRB - keep existing structure."""
        return result_data

    def get_output_filename(self, task_id: str) -> str:
        return f"{task_id}.json"


class RQADatasetManager(BenchmarkDatasetManager):
    """Dataset manager for ResearchQA (RQA)."""

    def load_queries(
        self,
        file_path: str,
        task_ids: Optional[List[int]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Load queries from JSON file (not JSONL)."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"📋 Loaded {len(data)} queries from {file_path}")

            # Convert to list format with indices if needed
            queries = []
            for i, item in enumerate(data):
                # Add index if not present
                if "index" not in item:
                    item["index"] = i

                # For RQA, append word count instruction to the query
                if "query" in item:
                    original_query = item["query"]
                    if not original_query.endswith(
                        "\nPlease answer in around 240-260 words."
                    ):
                        item["query"] = (
                            original_query + "\nPlease answer in around 240-260 words."
                        )

                queries.append(item)

            # Filter by task IDs if specified (using indices)
            if task_ids:
                queries = [q for i, q in enumerate(queries) if i in task_ids]
                logger.info(f"🎯 Filtered to {len(queries)} specific tasks: {task_ids}")

            # Apply limit if specified
            if limit:
                queries = queries[:limit]
                logger.info(f"🔢 Limited to first {len(queries)} tasks")

            return queries

        except FileNotFoundError:
            logger.error(f"Query file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading queries: {e}")
            raise

    def get_query_field(self) -> str:
        return "query"

    def get_processing_config(self) -> Dict:
        return {
            "max_loops_default": 2,
            "benchmark_mode": True,
            "qa_mode": False,
            "visualization_disabled": True,
        }

    def format_result(self, task_data: Dict, result_data: Dict) -> Dict:
        """Format result for RQA - extract answer field with metadata."""
        answer = result_data.get("article", "")
        task_id = result_data.get(
            "id", task_data.get("id", task_data.get("index", "unknown"))
        )

        return {"id": task_id, "answer": answer, "answer_length": len(answer.split())}

    def get_output_filename(self, task_id: str) -> str:
        return f"rqa_{task_id}.json"


class DeepConsultDatasetManager(BenchmarkDatasetManager):
    """Dataset manager for DeepConsult CSV dataset."""

    def load_queries(
        self,
        file_path: str,
        task_ids: Optional[List[int]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Load queries from CSV file."""
        import csv

        queries = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    # Create query dict with index as ID
                    query_data = {"id": i, "index": i, "query": row["query"].strip()}
                    queries.append(query_data)

            logger.info(f"📋 Loaded {len(queries)} queries from {file_path}")

            # Filter by task IDs if specified
            if task_ids:
                queries = [q for q in queries if q["id"] in task_ids]
                logger.info(f"🎯 Filtered to {len(queries)} specific tasks: {task_ids}")

            # Apply limit if specified
            if limit:
                queries = queries[:limit]
                logger.info(f"🔢 Limited to first {len(queries)} tasks")

            return queries

        except FileNotFoundError:
            logger.error(f"Query file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading queries: {e}")
            raise

    def get_query_field(self) -> str:
        return "query"

    def get_processing_config(self) -> Dict:
        return {
            "max_loops_default": 10,
            "benchmark_mode": False,
            "qa_mode": False,
            "visualization_disabled": True,
        }

    def format_result(self, task_data: Dict, result_data: Dict) -> Dict:
        """Format result for DeepConsult - keep existing structure."""
        return result_data

    def get_output_filename(self, task_id: str) -> str:
        return f"deepconsult_{task_id}.json"


def get_dataset_manager(benchmark_type: str) -> BenchmarkDatasetManager:
    """Factory function to get the appropriate dataset manager."""
    if benchmark_type == "drb":
        return DRBDatasetManager()
    elif benchmark_type == "rqa":
        return RQADatasetManager()
    elif benchmark_type == "deepconsult":
        return DeepConsultDatasetManager()
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")


def get_default_file_paths(benchmark_type: str) -> Dict[str, str]:
    """Get default file paths for different benchmarks."""
    base_dir = Path(__file__).parent.parent.parent

    if benchmark_type == "drb":
        return {
            "queries_file": str(
                base_dir
                / "benchmarks"
                / "deep_research_bench"
                / "data"
                / "prompt_data"
                / "query.jsonl"
            ),
            "output_dir": str(
                base_dir
                / "benchmarks"
                / "deep_research_bench"
                / "results"
                / "edr_reports_gemini"
            ),
        }
    elif benchmark_type == "rqa":
        return {
            "queries_file": str(
                base_dir / "benchmarks" / "ResearchQA" / "researchqa.json"
            ),
            "output_dir": str(
                base_dir
                / "benchmarks"
                / "ResearchQA"
                / "results"
                / "edr_reports_gemini"
            ),
        }
    elif benchmark_type == "deepconsult":
        return {
            "queries_file": "/Users/akshara.prabhakar/Documents/deep_research/benchmarks/ydc-deep-research-evals/datasets/DeepConsult/queries.csv",
            "output_dir": str(
                base_dir
                / "benchmarks"
                / "ydc-deep-research-evals"
                / "results"
                / "edr_reports_gemini"
            ),
        }
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")


class TaskManager:
    """Manages concurrent research tasks with resource limiting."""

    def __init__(self, max_workers: int = 4, rate_limit_delay: float = 1.0):
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.semaphore = asyncio.Semaphore(max_workers)
        self.last_request_time = 0
        self.request_lock = asyncio.Lock()

    async def rate_limit(self):
        """Apply rate limiting to prevent API overload."""
        async with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
            self.last_request_time = time.time()

    async def exponential_backoff_retry(self, func, max_retries=3):
        """Retry function with exponential backoff for rate limit errors."""
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    if attempt < max_retries - 1:
                        backoff_time = (2**attempt) * self.rate_limit_delay
                        logger.warning(
                            f"Rate limit hit, backing off for {backoff_time:.2f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(backoff_time)
                        continue
                raise e


async def run_single_research_task(
    task_data: Dict,
    dataset_manager: BenchmarkDatasetManager,
    max_web_search_loops: int = 1,
    visualization_disabled: bool = True,
    extra_effort: bool = False,
    minimum_effort: bool = False,
    benchmark_mode: bool = False,
    qa_mode: bool = False,
    provider: str = None,
    model: str = None,
    output_dir: str = None,
    task_manager: TaskManager = None,
) -> Tuple[bool, Dict, str]:
    """
    Run a single research task.

    Returns:
        Tuple of (success: bool, result: Dict, error_message: str)
    """
    # Get task ID and query using dataset manager
    task_id = task_data.get("id", task_data.get("index", "unknown"))
    query_field = dataset_manager.get_query_field()
    query = task_data[query_field]

    # Start timing for this task
    task_start_time = datetime.now()

    # Apply rate limiting
    if task_manager:
        await task_manager.rate_limit()

    logger.info(f"[Task {task_id}]")

    # Update global stats
    with STATS_LOCK:
        GLOBAL_STATS["tasks_started"] += 1
        GLOBAL_STATS["tasks_in_progress"] += 1

    try:
        # Import here to avoid import conflicts in concurrent execution
        from src.state import SummaryState
        from src.graph import create_graph

        # Set up provider and model with defaults
        if not provider:
            provider = os.environ.get("LLM_PROVIDER", "openai")
        if not model:
            model = os.environ.get("LLM_MODEL", "o3-mini")

        # Create a fresh graph for this task (each task gets its own instance)
        fresh_graph = create_graph()

        # Determine benchmark type from dataset manager
        if isinstance(dataset_manager, RQADatasetManager):
            benchmark_type = "RQA"
        elif isinstance(dataset_manager, DeepConsultDatasetManager):
            benchmark_type = "DEEPCONSULT"
        else:
            benchmark_type = "DRB"

        # Generate unique run reference for this task
        run_ref = f"EVAL_{benchmark_type}_{task_id}"

        # Create graph configuration
        graph_config = {
            "configurable": {
                "llm_provider": provider,
                "llm_model": model,
                "max_web_research_loops": max_web_search_loops,
                "user_prompt": query,
            },
            "recursion_limit": 100,
            "tags": [
                f"provider:{provider}",
                f"model:{model}",
                f"loops:{max_web_search_loops}",
                f"task_id:{task_id}",
                f"benchmark:{benchmark_type}",
            ],
            "metadata": {
                "run_ref": run_ref,
                "query": query,
                "provider": provider,
                "model": model,
                "max_loops": max_web_search_loops,
                "benchmark": benchmark_type,
            },
        }

        # Create initial state as SummaryState object
        initial_state = SummaryState(
            research_topic=query,
            search_query=query,
            running_summary="",
            research_complete=False,
            knowledge_gap="",
            research_loop_count=0,
            sources_gathered=[],
            web_research_results=[],
            search_results_empty=False,
            selected_search_tool="general_search",
            source_citations={},
            subtopic_queries=[],
            subtopics_metadata=[],
            extra_effort=extra_effort,
            minimum_effort=minimum_effort,
            qa_mode=qa_mode,
            benchmark_mode=benchmark_mode,
            visualization_disabled=visualization_disabled,
            llm_provider=provider,
            llm_model=model,
            uploaded_knowledge=None,
            uploaded_files=[],
            analyzed_files=[],
        )

        logger.info(f"[Task {task_id}] Executing research graph...")

        # Time the graph execution
        graph_start_time = datetime.now()

        # Run the graph using async method
        result = await fresh_graph.ainvoke(initial_state, config=graph_config)

        graph_end_time = datetime.now()
        graph_duration = graph_end_time - graph_start_time

        # Extract the result based on benchmark mode
        final_summary = result.get("running_summary", "No summary generated")

        # Handle different modes appropriately
        if qa_mode or benchmark_mode:
            # For QA and benchmark modes, use the benchmark_result if available
            benchmark_result = result.get("benchmark_result", {})
            if benchmark_result:
                # Use the full_response (which includes citations for benchmark mode)
                final_content = benchmark_result.get("full_response", "")
                if not final_content:
                    # Fallback to structured answer if full_response is not available
                    answer = benchmark_result.get("answer", "")
                    confidence = benchmark_result.get("confidence_level", "")
                    evidence = benchmark_result.get("evidence", "")
                    limitations = benchmark_result.get("limitations", "")

                    final_content = f"**Answer:** {answer}\n\n"
                    if confidence:
                        final_content += f"**Confidence:** {confidence}\n\n"
                    if evidence:
                        final_content += f"**Supporting Evidence:** {evidence}\n\n"
                    if limitations:
                        final_content += f"**Limitations:** {limitations}\n\n"

                mode_name = "benchmark mode" if benchmark_mode else "QA mode"
                logger.info(
                    f"[Task {task_id}] Using {mode_name} result with {'citations' if benchmark_mode else 'basic sources'}"
                )
            else:
                final_content = final_summary
                logger.info(
                    f"[Task {task_id}] No benchmark result available, using running summary"
                )
        else:
            # Regular mode - prioritize markdown_report if available
            markdown_report = result.get("markdown_report", "")
            if markdown_report and markdown_report.strip():
                # Find the start of Executive Summary section
                exec_summary_start = markdown_report.find("## Executive Summary\n")
                if exec_summary_start >= 0:
                    final_content = markdown_report[exec_summary_start:]
                    logger.info(
                        f"[Task {task_id}] Using clean markdown report (from Executive Summary)"
                    )
                else:
                    final_content = markdown_report
                    logger.info(
                        f"[Task {task_id}] Using complete markdown report (no Executive Summary found)"
                    )
            else:
                final_content = final_summary
                logger.info(
                    f"[Task {task_id}] Using running summary (no markdown report available)"
                )

        # Calculate total execution time
        task_end_time = datetime.now()
        total_duration = task_end_time - task_start_time

        # Extract debugging information from result
        debug_info = {
            "research_loops": result.get("research_loop_count", 0),
            "sources_gathered": len(result.get("sources_gathered", [])),
            "knowledge_gap": result.get("knowledge_gap", ""),
            "selected_search_tool": result.get("selected_search_tool", "unknown"),
            "research_complete": result.get("research_complete", False),
        }

        # Create comprehensive result JSON structure
        result_data = {
            "id": task_id,
            query_field: query,  # Use the correct field name for the benchmark
            "article": final_content,
            "summary": final_summary,
            "timing": {
                "start_time": task_start_time.isoformat(),
                "end_time": task_end_time.isoformat(),
                "total_duration_seconds": total_duration.total_seconds(),
                "graph_execution_seconds": graph_duration.total_seconds(),
                "setup_time_seconds": (
                    graph_start_time - task_start_time
                ).total_seconds(),
            },
            "debug_info": debug_info,
            "content_stats": {
                "final_content_length": len(final_content.split()),
                "final_summary_length": len(final_summary.split()),
                "benchmark_result_available": (
                    bool(result.get("benchmark_result", {}))
                    if (qa_mode or benchmark_mode)
                    else False
                ),
                "content_type": (
                    "benchmark_result"
                    if (qa_mode or benchmark_mode)
                    and result.get("benchmark_result", {})
                    else "running_summary"
                ),
            },
        }

        # Format result according to benchmark requirements
        formatted_result = dataset_manager.format_result(task_data, result_data)

        # Save to individual JSON file using benchmark-specific filename
        output_filename = dataset_manager.get_output_filename(str(task_id))
        output_file = os.path.join(output_dir, output_filename)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(formatted_result, f, indent=2, ensure_ascii=False)

        # For RQA, also update the consolidated results file
        if isinstance(dataset_manager, RQADatasetManager):
            consolidated_file = os.path.join(output_dir, "edr_answers.json")

            # Load existing consolidated results
            consolidated_results = {}
            if os.path.exists(consolidated_file):
                try:
                    with open(consolidated_file, "r", encoding="utf-8") as f:
                        consolidated_results = json.load(f)
                except (json.JSONDecodeError, IOError):
                    logger.warning(
                        f"Could not load existing consolidated file: {consolidated_file}"
                    )
                    consolidated_results = {}

            # Add this result to consolidated results
            consolidated_results[str(task_id)] = {"answer": formatted_result["answer"]}

            # Save updated consolidated results
            with open(consolidated_file, "w", encoding="utf-8") as f:
                json.dump(consolidated_results, f, indent=2, ensure_ascii=False)

            logger.info(
                f"[Task {task_id}] 📋 Updated consolidated results: {consolidated_file}"
            )

        # Performance metrics
        throughput = (
            len(final_content) / total_duration.total_seconds()
            if total_duration.total_seconds() > 0
            else 0
        )

        logger.info(
            f"[Task {task_id}] ✅ SUCCESS - Completed in {total_duration.total_seconds():.2f}s"
        )
        logger.info(
            f"[Task {task_id}] 📊 Metrics: {debug_info['research_loops']} loops, {debug_info['sources_gathered']} sources, {len(final_content):,} chars"
        )
        logger.info(f"[Task {task_id}] 📈 Throughput: {throughput:.0f} chars/second")
        logger.info(f"[Task {task_id}] 💾 Saved to {output_file}")

        # Update global stats
        with STATS_LOCK:
            GLOBAL_STATS["completed"] += 1
            GLOBAL_STATS["tasks_in_progress"] -= 1

        return True, result_data, ""

    except Exception as e:
        error_msg = f"Task {task_id} failed: {str(e)}"
        logger.error(f"[Task {task_id}] ❌ ERROR: {error_msg}")
        logger.exception(e)  # Log full traceback

        # Update global stats
        with STATS_LOCK:
            GLOBAL_STATS["failed"] += 1
            GLOBAL_STATS["tasks_in_progress"] -= 1

        return False, {}, error_msg


async def process_tasks_concurrently(
    tasks: List[Dict],
    dataset_manager: BenchmarkDatasetManager,
    max_workers: int = 4,
    max_web_search_loops: int = 1,
    visualization_disabled: bool = True,
    extra_effort: bool = False,
    minimum_effort: bool = False,
    benchmark_mode: bool = False,
    qa_mode: bool = False,
    provider: str = None,
    model: str = None,
    output_dir: str = None,
    rate_limit_delay: float = 1.0,
):
    """Process multiple research tasks concurrently."""

    logger.info(f"🚀 Starting concurrent processing of {len(tasks)} tasks")
    logger.info(f"📊 Max workers: {max_workers}")
    logger.info(f"⏱️  Rate limit delay: {rate_limit_delay}s")
    logger.info(f"🤖 Using {provider}/{model}")

    # Initialize global stats
    with STATS_LOCK:
        GLOBAL_STATS["start_time"] = time.time()
        GLOBAL_STATS["completed"] = 0
        GLOBAL_STATS["failed"] = 0
        GLOBAL_STATS["tasks_started"] = 0
        GLOBAL_STATS["tasks_in_progress"] = 0

    # Create task manager for resource control
    task_manager = TaskManager(max_workers, rate_limit_delay)

    # Create semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(max_workers)

    async def run_with_semaphore(task_data):
        async with semaphore:  # Limit concurrent tasks
            return await run_single_research_task(
                task_data=task_data,
                dataset_manager=dataset_manager,
                max_web_search_loops=max_web_search_loops,
                visualization_disabled=visualization_disabled,
                extra_effort=extra_effort,
                minimum_effort=minimum_effort,
                benchmark_mode=benchmark_mode,
                qa_mode=qa_mode,
                provider=provider,
                model=model,
                output_dir=output_dir,
                task_manager=task_manager,
            )

    # Start progress monitoring task
    async def monitor_progress():
        """Monitor and log progress periodically."""
        while True:
            await asyncio.sleep(10)  # Update every 10 seconds
            with STATS_LOCK:
                if GLOBAL_STATS["start_time"]:
                    elapsed = time.time() - GLOBAL_STATS["start_time"]
                    total_tasks = len(tasks)
                    completed = GLOBAL_STATS["completed"]
                    failed = GLOBAL_STATS["failed"]
                    in_progress = GLOBAL_STATS["tasks_in_progress"]

                    completion_rate = completed / elapsed if elapsed > 0 else 0
                    eta = (
                        (total_tasks - completed - failed) / completion_rate
                        if completion_rate > 0
                        else 0
                    )

                    logger.info(
                        f"📈 Progress: {completed}/{total_tasks} completed, "
                        f"{failed} failed, {in_progress} in progress, "
                        f"ETA: {eta/60:.1f}min"
                    )

                    if completed + failed >= total_tasks:
                        break

    # Start monitoring
    monitor_task = asyncio.create_task(monitor_progress())

    try:
        # Create all tasks
        task_coroutines = [run_with_semaphore(task_data) for task_data in tasks]

        # Run all tasks concurrently
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)

        # Process results
        successful_results = []
        failed_results = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append(f"Task {tasks[i]['id']}: {str(result)}")
                logger.error(f"Task {tasks[i]['id']} failed with exception: {result}")
            else:
                success, data, error_msg = result
                if success:
                    successful_results.append(data)
                else:
                    failed_results.append(error_msg)

        # Cancel monitoring
        monitor_task.cancel()

        # Final statistics
        with STATS_LOCK:
            total_time = time.time() - GLOBAL_STATS["start_time"]

        logger.info(f"\n{'='*80}")
        logger.info(f"🎉 CONCURRENT PROCESSING COMPLETE!")
        logger.info(f"{'='*80}")
        logger.info(f"✅ Successful: {len(successful_results)}")
        logger.info(f"❌ Failed: {len(failed_results)}")
        logger.info(f"⏱️  Total time: {total_time/60:.2f} minutes")
        logger.info(f"📊 Average time per task: {total_time/len(tasks):.2f} seconds")
        logger.info(
            f"🚀 Throughput: {len(successful_results)/total_time*60:.2f} tasks/minute"
        )

        if failed_results:
            logger.warning(f"\n❌ Failed tasks:")
            for error in failed_results:
                logger.warning(f"  - {error}")

        return successful_results, failed_results

    except Exception as e:
        monitor_task.cancel()
        logger.error(f"Critical error in concurrent processing: {e}")
        raise


async def main():
    """Main function to handle command line arguments and run concurrent research."""

    parser = argparse.ArgumentParser(
        description="Run deep research agent concurrently on multiple queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run DRB benchmark (default)
  python run_research_concurrent.py --benchmark-type drb --max-workers 4 --max-loops 1
  
  # Run ResearchQA benchmark  
  python run_research_concurrent.py --benchmark-type rqa --max-workers 2 --limit 10
  
  # Run DeepConsult benchmark with regular mode and max 5 loops
  python run_research_concurrent.py --benchmark-type deepconsult --max-workers 4 --max-loops 5
  
  # Custom configuration for RQA
  python run_research_concurrent.py --benchmark-type rqa --provider anthropic --model claude-3-5-sonnet --max-workers 4
  
  # Process specific task IDs
  python run_research_concurrent.py --benchmark-type rqa --task-ids "0,1,2,3,4" --max-workers 2
        """,
    )

    # Configuration arguments
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent workers (default: 4)",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=1,
        help="Maximum number of web search loops per task (default: 1)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Delay between API requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--disable-visualizations",
        action="store_true",
        default=True,
        help="Disable visualization generation (default: True)",
    )
    parser.add_argument(
        "--extra-effort",
        action="store_true",
        help="Use extra effort mode for more thorough research",
    )
    parser.add_argument(
        "--minimum-effort",
        action="store_true",
        help="Use minimum effort mode for faster research",
    )
    parser.add_argument(
        "--benchmark-mode", action="store_true", help="Run in benchmark mode"
    )
    parser.add_argument(
        "--benchmark-type",
        choices=["drb", "rqa", "deepconsult"],
        default="drb",
        help="Benchmark type: 'drb' for Deep Research Bench (default), 'rqa' for ResearchQA, or 'deepconsult' for DeepConsult CSV dataset",
    )

    # LLM configuration
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "groq", "google"],
        default="google",
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="LLM model name (e.g., 'o3-mini', 'claude-3-5-sonnet', 'gemini-2.5-pro')",
    )

    # File paths (optional - defaults based on benchmark type)
    parser.add_argument(
        "--queries-file",
        type=str,
        help="Path to queries file (defaults based on benchmark type)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for storing generated reports (defaults based on benchmark type)",
    )

    # Task filtering
    parser.add_argument(
        "--task-ids",
        type=str,
        help="Comma-separated list of specific task IDs to process (e.g., '1,5,10')",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of tasks to process (for testing)"
    )

    args = parser.parse_args()

    # Get dataset manager and default paths
    try:
        dataset_manager = get_dataset_manager(args.benchmark_type)
        default_paths = get_default_file_paths(args.benchmark_type)
    except ValueError as e:
        logger.error(f"❌ ERROR: {e}")
        return 1

    # Use provided paths or defaults
    queries_file = args.queries_file or default_paths["queries_file"]
    output_dir = args.output_dir or default_paths["output_dir"]

    # Get processing configuration from dataset manager
    processing_config = dataset_manager.get_processing_config()

    # Override max_loops with benchmark-specific default if not specified
    if (
        args.max_loops == 1
    ):  # Default value, might want to use benchmark-specific default
        max_loops = processing_config["max_loops_default"]
    else:
        max_loops = args.max_loops

    # Set benchmark mode based on dataset manager config (can be overridden by args)
    benchmark_mode = args.benchmark_mode or processing_config["benchmark_mode"]
    qa_mode = processing_config["qa_mode"]

    logger.info(f"🎯 Benchmark type: {args.benchmark_type.upper()}")
    logger.info(f"📋 Queries file: {queries_file}")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"🔄 Max loops: {max_loops}")
    logger.info(f"🧪 Benchmark mode: {benchmark_mode}")
    logger.info(f"❓ QA mode: {qa_mode}")

    # Check for required environment variables
    required_vars = []
    provider = args.provider or os.environ.get("LLM_PROVIDER", "openai")

    if provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif provider == "anthropic":
        required_vars.append("ANTHROPIC_API_KEY")
    elif provider == "groq":
        required_vars.append("GROQ_API_KEY")
    elif provider == "google":
        required_vars.extend(["GOOGLE_CLOUD_PROJECT"])

    # Always need search API
    required_vars.append("TAVILY_API_KEY")

    missing_vars = [var for var in required_vars if not os.environ.get(var)]

    if missing_vars:
        logger.error("❌ ERROR: Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"   {var}")
        logger.error("\nPlease set these environment variables before running.")
        return 1

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load queries using dataset manager
        task_ids_list = None
        if args.task_ids:
            task_ids_list = [int(x.strip()) for x in args.task_ids.split(",")]

        queries = dataset_manager.load_queries(queries_file, task_ids_list, args.limit)

        if not queries:
            logger.error("No queries found to process")
            return 1

        # Set environment variables for configuration
        original_max_loops = os.environ.get("MAX_WEB_RESEARCH_LOOPS")
        os.environ["MAX_WEB_RESEARCH_LOOPS"] = str(max_loops)
        if args.provider:
            os.environ["LLM_PROVIDER"] = args.provider
        if args.model:
            os.environ["LLM_MODEL"] = args.model

        try:
            # Process tasks concurrently
            successful_results, failed_results = await process_tasks_concurrently(
                tasks=queries,
                dataset_manager=dataset_manager,
                max_workers=args.max_workers,
                max_web_search_loops=max_loops,
                visualization_disabled=args.disable_visualizations,
                extra_effort=args.extra_effort,
                minimum_effort=args.minimum_effort,
                benchmark_mode=benchmark_mode,
                qa_mode=qa_mode,
                provider=args.provider,
                model=args.model,
                output_dir=output_dir,
                rate_limit_delay=args.rate_limit,
            )

            # Save summary report as log file
            summary_file = os.path.join(output_dir, "processing_summary.log")
            success_rate = len(successful_results) / len(queries) * 100

            # Calculate RQA-specific statistics if applicable
            rqa_stats = {}
            if args.benchmark_type == "rqa" and successful_results:
                total_answer_length = 0
                answer_lengths = []
                for result in successful_results:
                    if isinstance(result, dict) and "answer_length" in result:
                        length = result["answer_length"]
                        total_answer_length += length
                        answer_lengths.append(length)

                if answer_lengths:
                    rqa_stats = {
                        "total_answer_chars": total_answer_length,
                        "avg_answer_length": total_answer_length / len(answer_lengths),
                        "min_answer_length": min(answer_lengths),
                        "max_answer_length": max(answer_lengths),
                    }

            with open(summary_file, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("DEEP RESEARCH CONCURRENT PROCESSING SUMMARY\n")
                f.write("=" * 80 + "\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Total tasks: {len(queries)}\n")
                f.write(f"Successful: {len(successful_results)}\n")
                f.write(f"Failed: {len(failed_results)}\n")
                f.write(f"Success rate: {success_rate:.2f}%\n")
                f.write("\n")
                f.write("CONFIGURATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Benchmark type: {args.benchmark_type}\n")
                f.write(f"Max workers: {args.max_workers}\n")
                f.write(f"Max loops: {max_loops}\n")
                f.write(f"Rate limit: {args.rate_limit}\n")
                f.write(f"Provider: {provider}\n")
                f.write(f"Model: {args.model}\n")
                f.write(f"Extra effort: {args.extra_effort}\n")
                f.write(f"Minimum effort: {args.minimum_effort}\n")
                f.write(f"Benchmark mode: {benchmark_mode}\n")
                f.write(f"QA mode: {qa_mode}\n")

                # Add RQA-specific statistics
                if rqa_stats:
                    f.write("\n")
                    f.write("RQA STATISTICS:\n")
                    f.write("-" * 40 + "\n")
                    f.write(
                        f"Total answer characters: {rqa_stats['total_answer_chars']:,}\n"
                    )
                    f.write(
                        f"Average answer length: {rqa_stats['avg_answer_length']:.1f} chars\n"
                    )
                    f.write(
                        f"Min answer length: {rqa_stats['min_answer_length']:,} chars\n"
                    )
                    f.write(
                        f"Max answer length: {rqa_stats['max_answer_length']:,} chars\n"
                    )
                    f.write(f"Consolidated results: edr_answers.json\n")

                if failed_results:
                    f.write("\n")
                    f.write("FAILED TASKS:\n")
                    f.write("-" * 40 + "\n")
                    for error in failed_results:
                        f.write(f"- {error}\n")

                f.write("\n")
                f.write("=" * 80 + "\n")

            logger.info(f"📊 Summary saved to: {summary_file}")

            if len(successful_results) == len(queries):
                logger.info("🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
                return 0
            else:
                logger.warning(f"⚠️  {len(failed_results)} tasks failed")
                return 1

        finally:
            # Restore original environment variables
            if original_max_loops is not None:
                os.environ["MAX_WEB_RESEARCH_LOOPS"] = original_max_loops
            elif "MAX_WEB_RESEARCH_LOOPS" in os.environ:
                del os.environ["MAX_WEB_RESEARCH_LOOPS"]

    except KeyboardInterrupt:
        logger.info("\n🛑 Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ UNEXPECTED ERROR: {e}")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
