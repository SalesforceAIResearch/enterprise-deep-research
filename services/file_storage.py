import os
import uuid
import shutil
import logging
import mimetypes
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import aiofiles
from fastapi import UploadFile
try:
    import magic
except ImportError:
    magic = None

from models.file_analysis import FileMetadata, FileStatus

logger = logging.getLogger(__name__)

class FileStorageService:
    """Service for managing file storage and metadata"""
    
    # Configuration
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))  # 100MB default
    CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL_HOURS", 24))  # 24 hours
    
    # Supported file types
    SUPPORTED_FORMATS = {
        # Documents
        "pdf", "doc", "docx", "txt", "md", "rtf",
        # Spreadsheets
        "csv", "xlsx", "xls", "ods",
        # Images
        "jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "svg",
        # Audio
        "mp3", "wav", "flac", "aac", "ogg", "m4a",
        # Video
        "mp4", "avi", "mov", "mkv", "wmv", "flv", "webm",
        # Data
        "json", "xml", "yaml", "yml",
        # Archives
        "zip", "tar", "gz", "rar", "7z"
    }
    
    # In-memory storage for file metadata (in production, use a database)
    _file_registry = {}
    
    @classmethod
    def _ensure_upload_directory(cls):
        """Ensure upload directory exists"""
        os.makedirs(cls.UPLOAD_DIR, exist_ok=True)
    
    @classmethod
    async def store_file(cls, upload_file: UploadFile) -> Dict[str, Any]:
        """
        Store an uploaded file and return metadata
        
        Args:
            upload_file: FastAPI UploadFile object
        
        Returns:
            Dictionary with file metadata
        """
        try:
            cls._ensure_upload_directory()
            
            # Validate file
            validation_result = await cls._validate_file(upload_file)
            if not validation_result["valid"]:
                raise ValueError(validation_result["error"])
            
            # Generate unique file ID and sanitized filename
            file_id = str(uuid.uuid4())
            original_name = upload_file.filename
            file_extension = Path(original_name).suffix.lower()
            sanitized_name = cls._sanitize_filename(original_name)
            stored_filename = f"{file_id}_{sanitized_name}"
            
            # Create file path
            file_path = os.path.join(cls.UPLOAD_DIR, stored_filename)
            
            # Store file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await upload_file.read()
                await f.write(content)
            
            # Get file stats
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            
            # Detect MIME type safely
            try:
                mime_type = magic.from_file(file_path, mime=True)
            except:
                mime_type = "application/octet-stream"
            
            # Create metadata
            metadata = {
                "file_id": file_id,
                "filename": stored_filename,
                "original_name": original_name,
                "file_type": file_extension[1:] if file_extension else "",
                "file_size": file_size,
                "mime_type": mime_type,
                "file_path": file_path,
                "upload_timestamp": datetime.now(),
                "status": FileStatus.UPLOADED
            }
            
            # Store in registry
            cls._file_registry[file_id] = metadata
            
            logger.info(f"File stored: {file_id} ({original_name}, {file_size} bytes)")
            return metadata
            
        except Exception as e:
            logger.error(f"Error storing file {upload_file.filename}: {str(e)}")
            raise
    
    @classmethod
    async def _validate_file(cls, upload_file: UploadFile) -> Dict[str, Any]:
        """Validate uploaded file"""
        try:
            # Check file size
            if upload_file.size and upload_file.size > cls.MAX_FILE_SIZE:
                return {
                    "valid": False,
                    "error": f"File size ({upload_file.size} bytes) exceeds maximum allowed size ({cls.MAX_FILE_SIZE} bytes)"
                }
            
            # Check file extension
            if upload_file.filename:
                file_extension = Path(upload_file.filename).suffix[1:].lower()
                if file_extension not in cls.SUPPORTED_FORMATS:
                    return {
                        "valid": False,
                        "error": f"File type '{file_extension}' not supported. Supported formats: {', '.join(sorted(cls.SUPPORTED_FORMATS))}"
                    }
            else:
                return {
                    "valid": False,
                    "error": "No filename provided"
                }
            
            return {"valid": True}
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}"
            }
    
    @classmethod
    def _sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove or replace dangerous characters
        dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        sanitized = filename
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Limit length
        if len(sanitized) > 100:
            name_part = Path(sanitized).stem[:90]
            ext_part = Path(sanitized).suffix
            sanitized = name_part + ext_part
        
        return sanitized
    
    @classmethod
    async def get_file_metadata(cls, file_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a file"""
        return cls._file_registry.get(file_id)
    
    @classmethod
    async def get_file_path(cls, file_id: str) -> Optional[str]:
        """Get file path for a file ID"""
        metadata = cls._file_registry.get(file_id)
        if metadata:
            return metadata["file_path"]
        return None
    
    @classmethod
    async def file_exists(cls, file_id: str) -> bool:
        """Check if a file exists"""
        metadata = cls._file_registry.get(file_id)
        if metadata:
            return os.path.exists(metadata["file_path"])
        return False
    
    @classmethod
    async def delete_file(cls, file_id: str) -> bool:
        """
        Delete a file and its metadata
        
        Args:
            file_id: Unique identifier for the file
        
        Returns:
            True if file was deleted, False if not found
        """
        try:
            metadata = cls._file_registry.get(file_id)
            if not metadata:
                return False
            
            file_path = metadata["file_path"]
            
            # Delete physical file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Remove from registry
            del cls._file_registry[file_id]
            
            logger.info(f"File deleted: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file {file_id}: {str(e)}")
            return False
    
    @classmethod
    async def list_files(
        cls, 
        limit: int = 50, 
        offset: int = 0, 
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List uploaded files with pagination and filtering
        
        Args:
            limit: Maximum number of files to return
            offset: Number of files to skip
            status_filter: Filter by file status
        
        Returns:
            List of file metadata dictionaries
        """
        try:
            files = list(cls._file_registry.values())
            
            # Apply status filter
            if status_filter:
                files = [f for f in files if f["status"] == status_filter]
            
            # Sort by upload timestamp (newest first)
            files.sort(key=lambda x: x["upload_timestamp"], reverse=True)
            
            # Apply pagination
            start_idx = offset
            end_idx = offset + limit
            paginated_files = files[start_idx:end_idx]
            
            return paginated_files
            
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return []
    
    @classmethod
    async def update_file_status(cls, file_id: str, status: FileStatus) -> bool:
        """Update file status"""
        try:
            if file_id in cls._file_registry:
                cls._file_registry[file_id]["status"] = status
                logger.info(f"Status updated for {file_id}: {status}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating status for {file_id}: {str(e)}")
            return False
    
    @classmethod
    async def cleanup_old_files(cls, max_age_hours: int = None) -> int:
        """
        Clean up old files
        
        Args:
            max_age_hours: Maximum age in hours (default: CLEANUP_INTERVAL)
        
        Returns:
            Number of files cleaned up
        """
        if max_age_hours is None:
            max_age_hours = cls.CLEANUP_INTERVAL
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            files_to_delete = []
            
            for file_id, metadata in cls._file_registry.items():
                if metadata["upload_timestamp"] < cutoff_time:
                    files_to_delete.append(file_id)
            
            cleaned_count = 0
            for file_id in files_to_delete:
                if await cls.delete_file(file_id):
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return 0
    
    @classmethod
    async def get_storage_stats(cls) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            total_files = len(cls._file_registry)
            total_size = sum(metadata["file_size"] for metadata in cls._file_registry.values())
            
            # Count by status
            status_counts = {}
            for metadata in cls._file_registry.values():
                status = metadata["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by file type
            type_counts = {}
            for metadata in cls._file_registry.values():
                file_type = metadata["file_type"]
                type_counts[file_type] = type_counts.get(file_type, 0) + 1
            
            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "status_distribution": status_counts,
                "type_distribution": type_counts,
                "upload_directory": cls.UPLOAD_DIR
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {}

# Initialize storage on module import
def init_storage():
    """Initialize file storage system"""
    try:
        FileStorageService._ensure_upload_directory()
        logger.info(f"File storage initialized: {FileStorageService.UPLOAD_DIR}")
    except Exception as e:
        logger.error(f"Failed to initialize file storage: {str(e)}")

# Auto-initialize
init_storage()