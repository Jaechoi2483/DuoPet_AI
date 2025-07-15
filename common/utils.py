"""
Utility functions for DuoPet AI Service

This module provides common utility functions used across the application.
"""

import os
import hashlib
import mimetypes
from typing import List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import cv2

from common.config import get_settings
from common.exceptions import InvalidFileFormatError, FileTooLargeError

settings = get_settings()


def validate_image_file(
    filename: str,
    file_size: int,
    allowed_extensions: Optional[List[str]] = None
) -> None:
    """
    Validate image file format and size
    
    Args:
        filename: Name of the file
        file_size: Size of file in bytes
        allowed_extensions: List of allowed extensions (defaults to settings)
    
    Raises:
        InvalidFileFormatError: If file format is not allowed
        FileTooLargeError: If file exceeds size limit
    """
    if allowed_extensions is None:
        allowed_extensions = settings.ALLOWED_IMAGE_EXTENSIONS
    
    # Check file extension
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        raise InvalidFileFormatError(ext, allowed_extensions)
    
    # Check file size
    if file_size > settings.max_upload_size_bytes:
        raise FileTooLargeError(file_size, settings.max_upload_size_bytes)


def validate_video_file(
    filename: str,
    file_size: int,
    allowed_extensions: Optional[List[str]] = None
) -> None:
    """
    Validate video file format and size
    
    Args:
        filename: Name of the file
        file_size: Size of file in bytes
        allowed_extensions: List of allowed extensions (defaults to settings)
    
    Raises:
        InvalidFileFormatError: If file format is not allowed
        FileTooLargeError: If file exceeds size limit
    """
    if allowed_extensions is None:
        allowed_extensions = settings.ALLOWED_VIDEO_EXTENSIONS
    
    # Check file extension
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        raise InvalidFileFormatError(ext, allowed_extensions)
    
    # Check file size
    if file_size > settings.max_upload_size_bytes:
        raise FileTooLargeError(file_size, settings.max_upload_size_bytes)


def generate_file_hash(file_content: bytes) -> str:
    """
    Generate SHA256 hash of file content
    
    Args:
        file_content: File content as bytes
    
    Returns:
        Hex string of file hash
    """
    return hashlib.sha256(file_content).hexdigest()


def generate_unique_filename(original_filename: str, prefix: str = "") -> str:
    """
    Generate unique filename with timestamp
    
    Args:
        original_filename: Original filename
        prefix: Optional prefix for filename
    
    Returns:
        Unique filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(original_filename)
    
    if prefix:
        return f"{prefix}_{timestamp}_{name}{ext}"
    return f"{timestamp}_{name}{ext}"


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    maintain_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image as numpy array
        target_size: Target size (width, height)
        maintain_aspect_ratio: Whether to maintain aspect ratio
    
    Returns:
        Resized image
    """
    if maintain_aspect_ratio:
        # Calculate aspect ratio
        h, w = image.shape[:2]
        aspect = w / h
        
        # Calculate new dimensions
        if aspect > 1:  # Landscape
            new_w = target_size[0]
            new_h = int(new_w / aspect)
        else:  # Portrait
            new_h = target_size[1]
            new_w = int(new_h * aspect)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to target size
        top = (target_size[1] - new_h) // 2
        bottom = target_size[1] - new_h - top
        left = (target_size[0] - new_w) // 2
        right = target_size[0] - new_w - left
        
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        return padded
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(
    image: np.ndarray,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> np.ndarray:
    """
    Normalize image for model input
    
    Args:
        image: Input image as numpy array (0-255)
        mean: Mean values for normalization
        std: Standard deviation values for normalization
    
    Returns:
        Normalized image
    """
    # Convert to float32
    image = image.astype(np.float32) / 255.0
    
    # Apply normalization
    if mean is not None and std is not None:
        mean = np.array(mean).reshape(1, 1, 3)
        std = np.array(std).reshape(1, 1, 3)
        image = (image - mean) / std
    
    return image


def extract_video_metadata(video_path: str) -> dict:
    """
    Extract metadata from video file
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video metadata
    """
    cap = cv2.VideoCapture(video_path)
    
    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return metadata


def extract_frames_from_video(
    video_path: str,
    fps: int = 1,
    max_frames: Optional[int] = None
) -> List[np.ndarray]:
    """
    Extract frames from video at specified FPS
    
    Args:
        video_path: Path to video file
        fps: Frames per second to extract
        max_frames: Maximum number of frames to extract
    
    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)
            
            if max_frames and len(frames) >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    return frames


def calculate_confidence_threshold(
    base_threshold: float,
    num_detections: int,
    min_threshold: float = 0.5
) -> float:
    """
    Calculate dynamic confidence threshold based on number of detections
    
    Args:
        base_threshold: Base confidence threshold
        num_detections: Number of detections
        min_threshold: Minimum threshold value
    
    Returns:
        Adjusted confidence threshold
    """
    # Increase threshold if too many detections
    if num_detections > 10:
        adjustment = 0.05 * (num_detections - 10)
        return min(base_threshold + adjustment, 0.95)
    
    # Decrease threshold if too few detections
    elif num_detections < 3:
        adjustment = 0.05 * (3 - num_detections)
        return max(base_threshold - adjustment, min_threshold)
    
    return base_threshold


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted duration string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def is_within_time_range(
    timestamp: datetime,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> bool:
    """
    Check if timestamp is within specified time range
    
    Args:
        timestamp: Timestamp to check
        start_time: Start of time range (inclusive)
        end_time: End of time range (inclusive)
    
    Returns:
        True if timestamp is within range
    """
    if start_time and timestamp < start_time:
        return False
    if end_time and timestamp > end_time:
        return False
    return True


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to remove potentially dangerous characters
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    # Remove path separators and other dangerous characters
    dangerous_chars = ['/', '\\', '..', '~', '$', '`', '|', '<', '>', ':', '*', '?', '"']
    
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 100:
        name = name[:100]
    
    return name + ext