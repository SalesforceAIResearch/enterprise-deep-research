from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class FileStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class AnalysisType(str, Enum):
    QUICK = "quick"
    COMPREHENSIVE = "comprehensive"
    CUSTOM = "custom"

class FileUploadResponse(BaseModel):
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    filename: str = Field(..., description="Sanitized filename")
    original_name: str = Field(..., description="Original filename as uploaded")
    file_type: str = Field(..., description="File extension/type")
    file_size: int = Field(..., description="File size in bytes")
    status: FileStatus = Field(..., description="Current processing status")
    upload_timestamp: datetime = Field(..., description="When the file was uploaded")
    analysis_eta: Optional[str] = Field(None, description="Estimated time for analysis completion")

class FileAnalysisRequest(BaseModel):
    analysis_type: AnalysisType = Field(default=AnalysisType.COMPREHENSIVE, description="Type of analysis to perform")
    custom_prompt: Optional[str] = Field(None, description="Custom prompt for analysis if using custom type")

class FileAnalysisResponse(BaseModel):
    file_id: str = Field(..., description="Unique identifier for the analyzed file")
    status: FileStatus = Field(..., description="Analysis status")
    content_description: str = Field(..., description="Detailed description of the file content")
    analysis_timestamp: datetime = Field(..., description="When the analysis was completed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="File-specific metadata and insights")
    processing_time: float = Field(..., description="Time taken for analysis in seconds")
    error_message: Optional[str] = Field(None, description="Error message if analysis failed")

class FileMetadata(BaseModel):
    file_id: str
    filename: str
    original_name: str
    file_type: str
    file_size: int
    upload_timestamp: datetime
    file_path: str
    status: FileStatus

class ContentInsights(BaseModel):
    """Structured insights extracted from file content"""
    key_topics: List[str] = Field(default_factory=list, description="Main topics identified in the content")
    entities: List[str] = Field(default_factory=list, description="Named entities found in the content")
    summary: str = Field("", description="Brief summary of the content")
    language: Optional[str] = Field(None, description="Detected language of the content")
    sentiment: Optional[str] = Field(None, description="Overall sentiment if applicable")
    confidence_score: float = Field(0.0, description="Confidence score for the analysis (0-1)")

class DocumentMetadata(BaseModel):
    """Metadata specific to document files"""
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    character_count: Optional[int] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None

class ImageMetadata(BaseModel):
    """Metadata specific to image files"""
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    has_text: bool = False
    detected_objects: List[str] = Field(default_factory=list)
    color_palette: List[str] = Field(default_factory=list)

class DataFileMetadata(BaseModel):
    """Metadata specific to structured data files (CSV, Excel, etc.)"""
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns: List[str] = Field(default_factory=list)
    data_types: Dict[str, str] = Field(default_factory=dict)
    missing_values: Dict[str, int] = Field(default_factory=dict)
    sample_data: Optional[Dict[str, Any]] = None

class AudioVideoMetadata(BaseModel):
    """Metadata specific to audio and video files"""
    duration: Optional[float] = None  # in seconds
    format: Optional[str] = None
    bitrate: Optional[int] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    has_speech: bool = False
    transcript_available: bool = False

class FileAnalysisStatus(BaseModel):
    """Detailed status information for file analysis"""
    file_id: str
    status: FileStatus
    progress_percentage: float = Field(0.0, description="Analysis progress (0-100)")
    current_stage: str = Field("", description="Current processing stage")
    started_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    error_details: Optional[str] = None

class BatchAnalysisRequest(BaseModel):
    """Request model for batch file analysis"""
    file_ids: List[str] = Field(..., description="List of file IDs to analyze")
    analysis_type: AnalysisType = Field(default=AnalysisType.COMPREHENSIVE)
    custom_prompt: Optional[str] = None
    parallel_processing: bool = Field(True, description="Whether to process files in parallel")

class BatchAnalysisResponse(BaseModel):
    """Response model for batch file analysis"""
    batch_id: str = Field(..., description="Unique identifier for the batch operation")
    total_files: int = Field(..., description="Total number of files in the batch")
    status: str = Field(..., description="Overall batch status")
    individual_results: List[FileAnalysisResponse] = Field(default_factory=list)
    batch_started_at: datetime
    estimated_completion: Optional[datetime] = None