from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class ResearchRequest(BaseModel):
    """
    Model for the research request.
    """

    query: str = Field(..., description="The research query or topic to investigate")
    extra_effort: bool = Field(
        False, description="Whether to perform more extensive research (more loops)"
    )
    minimum_effort: bool = Field(
        False, description="Whether to force minimum (1 loop) research"
    )
    streaming: bool = Field(False, description="Whether to stream the response")
    provider: Optional[str] = Field(
        None,
        description="The LLM provider to use (e.g., 'openai', 'google', 'anthropic')",
    )
    model: Optional[str] = Field(
        None,
        description="The specific model to use (e.g., 'o3-mini', 'gemini-2.5-pro')",
    )
    benchmark_mode: bool = Field(
        False, description="Whether to run in benchmark Q&A mode for testing accuracy"
    )
    uploaded_data_content: Optional[str] = Field(
        None, description="Content of the uploaded external data source"
    )
    uploaded_files: Optional[List[str]] = Field(
        None, description="List of uploaded file IDs to include in research"
    )
    steering_enabled: bool = Field(
        False, description="Whether to enable real-time steering functionality"
    )
    parallel_search_enabled: bool = Field(
        False, description="Whether to enable parallel search execution"
    )
    parallel_search_max_concurrency: int = Field(
        2, description="Max concurrent search subtasks when parallelized"
    )
    database_info: Optional[List[Dict[str, Any]]] = Field(
        None, description="Information about uploaded databases for text2sql functionality"
    )


class ResearchResponse(BaseModel):
    """
    Model for the research response.
    """

    running_summary: str = Field(..., description="The comprehensive research summary")
    research_complete: bool = Field(
        ..., description="Whether the research process is complete"
    )
    research_loop_count: int = Field(
        ..., description="Number of research loops performed"
    )
    sources_gathered: List[str] = Field(
        default_factory=list, description="List of sources used in research"
    )
    web_research_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Raw web research results"
    )
    source_citations: Dict[str, Dict[str, str]] = Field(
        default_factory=dict, description="Source citations mapping"
    )
    benchmark_mode: bool = Field(
        default=False, description="Whether ran in benchmark Q&A mode"
    )
    benchmark_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Results from benchmark testing"
    )
    visualizations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Generated visualizations"
    )
    base64_encoded_images: List[Dict[str, Any]] = Field(
        default_factory=list, description="Base64 encoded images"
    )
    visualization_paths: List[str] = Field(
        default_factory=list, description="Paths to visualization files"
    )
    code_snippets: List[Dict[str, Any]] = Field(
        default_factory=list, description="Generated code snippets"
    )
    uploaded_knowledge: Optional[str] = Field(
        None, description="User-provided external knowledge"
    )
    analyzed_files: List[Dict[str, Any]] = Field(
        default_factory=list, description="Analysis results from uploaded files"
    )


class ResearchEvent(BaseModel):
    """
    Model for research events during streaming.
    """

    event_type: str = Field(..., description="Type of the event")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: Optional[str] = Field(None, description="Event timestamp")


class StreamResponse(BaseModel):
    """
    Model for streaming response.
    """

    stream_url: str = Field(..., description="URL to connect for streaming updates")
    message: str = Field(..., description="Status message")
