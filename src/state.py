import operator
from dataclasses import dataclass, field
from typing_extensions import TypedDict, Annotated
from typing import List, Dict, Optional, Any, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime
import json


def replace_list(old_list, new_list):
    """Custom reducer that replaces the old list with the new list completely.
    This allows us to completely clear or replace lists in the state.
    """
    # If the new list is explicitly set to an empty list, we want to clear the data
    return new_list


class SummaryState(BaseModel):
    """
    State for the research summary graph.
    """

    research_topic: str = Field(description="The main research topic")
    search_query: str = Field(description="Current search query")
    running_summary: str = Field(default="", description="Running summary of research")
    research_complete: bool = Field(
        default=False, description="Whether research is complete"
    )
    knowledge_gap: str = Field(default="", description="Identified knowledge gaps")
    research_loop_count: int = Field(
        default=0, description="Number of research loops completed"
    )
    sources_gathered: List[str] = Field(
        default_factory=list, description="List of sources gathered"
    )
    web_research_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Web research results"
    )
    search_results_empty: bool = Field(
        default=False, description="Whether search results were empty"
    )
    selected_search_tool: str = Field(
        default="general_search", description="Selected search tool"
    )
    source_citations: Dict[str, Any] = Field(
        default_factory=dict, description="Source citations"
    )
    subtopic_queries: List[str] = Field(
        default_factory=list, description="Subtopic queries"
    )
    subtopics_metadata: List[Dict[str, Any]] = Field(
        default_factory=list, description="Subtopics metadata"
    )
    extra_effort: bool = Field(default=False, description="Whether to use extra effort")
    minimum_effort: bool = Field(
        default=False, description="Whether to use minimum effort"
    )
    qa_mode: bool = Field(
        default=False, description="Whether in QA mode (simple question-answering)"
    )
    benchmark_mode: bool = Field(
        default=False,
        description="Whether in benchmark mode (with full citation processing)",
    )

    llm_provider: Optional[str] = Field(default=None, description="LLM provider")
    llm_model: Optional[str] = Field(default=None, description="LLM model")
    uploaded_knowledge: Optional[str] = Field(
        default=None, description="User-uploaded external knowledge"
    )
    uploaded_files: List[str] = Field(
        default_factory=list, description="List of uploaded file IDs"
    )
    analyzed_files: List[Dict[str, Any]] = Field(
        default_factory=list, description="Analysis results from uploaded files"
    )

    # Additional fields for enhanced functionality
    formatted_sources: List[Dict[str, Any]] = Field(
        default_factory=list, description="Formatted sources for UI"
    )
    useful_information: str = Field(
        default="", description="Useful information extracted"
    )
    missing_information: str = Field(
        default="", description="Missing information identified"
    )
    needs_refinement: bool = Field(
        default=False, description="Whether query needs refinement"
    )
    current_refined_query: str = Field(default="", description="Current refined query")
    refinement_reasoning: str = Field(
        default="", description="Reasoning for refinement"
    )
    previous_answers: List[str] = Field(
        default_factory=list, description="Previous answers"
    )
    reflection_history: List[str] = Field(
        default_factory=list, description="Reflection history"
    )

    # Visualization fields
    visualization_disabled: bool = Field(
        default=True, description="Whether to have visualizations"
    )
    visualizations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Generated visualizations"
    )
    base64_encoded_images: List[str] = Field(
        default_factory=list, description="Base64 encoded images"
    )
    visualization_html: str = Field(
        default="", description="Visualization HTML content"
    )
    visualization_paths: List[str] = Field(
        default_factory=list, description="Paths to visualization files"
    )

    # Code snippet fields
    code_snippets: List[Dict[str, Any]] = Field(
        default_factory=list, description="Generated code snippets"
    )

    # Report format fields
    markdown_report: Optional[str] = Field(
        default="",
        description="Plain markdown version of the report without HTML elements",
    )

    # Benchmark mode fields
    benchmark_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Benchmark result"
    )

    # Configuration
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Configuration settings"
    )

    def __init__(self, **data):
        super().__init__(**data)
        # Log the uploaded_knowledge when state is created
        if hasattr(self, "uploaded_knowledge") and self.uploaded_knowledge:
            print(f"[UPLOAD_TRACE] SummaryState.__init__: uploaded_knowledge set")
            print(
                f"[UPLOAD_TRACE] SummaryState.__init__: uploaded_knowledge length: {len(self.uploaded_knowledge)}"
            )
            print(
                f"[UPLOAD_TRACE] SummaryState.__init__: uploaded_knowledge preview: {self.uploaded_knowledge[:100]}..."
            )
        else:
            print(
                f"[UPLOAD_TRACE] SummaryState.__init__: uploaded_knowledge not set (value: {getattr(self, 'uploaded_knowledge', 'MISSING_ATTR')})"
            )


class SummaryStateInput(BaseModel):
    """
    Input model for the research process.
    """

    research_topic: str
    extra_effort: bool = False
    minimum_effort: bool = False
    qa_mode: bool = Field(
        default=False,
        description="Whether to run in QA mode (simple question-answering)",
    )
    benchmark_mode: bool = Field(
        default=False,
        description="Whether to run in benchmark mode (with full citation processing)",
    )
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    uploaded_knowledge: Optional[str] = None
    uploaded_files: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None


class SummaryStateOutput(BaseModel):
    """
    Output model for the research process.
    """

    running_summary: str
    research_complete: bool
    research_loop_count: int
    sources_gathered: List[str]
    web_research_results: List[Dict[str, Any]] = []
    source_citations: Dict[str, Dict[str, str]]
    qa_mode: bool = Field(default=False, description="Whether ran in QA mode")
    benchmark_mode: bool = Field(
        default=False, description="Whether ran in benchmark mode"
    )
    benchmark_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Results from benchmark testing"
    )
    visualizations: List[Dict[str, Any]] = []
    base64_encoded_images: List[Dict[str, Any]] = []
    visualization_paths: List[str] = []
    code_snippets: List[Dict[str, Any]] = []
    markdown_report: str = ""
    uploaded_knowledge: Optional[str] = None
    analyzed_files: List[Dict[str, Any]] = []
