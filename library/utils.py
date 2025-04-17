import os
import cv2
import torch
import logging
import gc
from datetime import datetime
from django.conf import settings

logger = logging.getLogger(__name__)

class FileManager:
    @staticmethod
    def create_directory(path):
        """Create directory if it doesn't exist"""
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_media_path(subdir, filename):
        """Get path in media directory"""
        path = os.path.join(settings.MEDIA_ROOT, subdir)
        FileManager.create_directory(path)
        return os.path.join(path, filename)

    @staticmethod
    def generate_timestamp_filename(prefix, extension):
        """Generate filename with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{prefix}_{timestamp}.{extension}"

class MemoryManager:
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

class ErrorHandler:
    @staticmethod
    def handle_error(error, context, raise_exception=False):
        """Unified error handling with logging"""
        error_msg = f"Error in {context}: {str(error)}"
        logger.error(error_msg)
        if raise_exception:
            raise Exception(error_msg)
        return error_msg

class FrameProcessor:
    @staticmethod
    def read_frame(camera):
        """Unified frame reading with retries"""
        try:
            if camera is None or not camera.isOpened():
                return None
                
            success, frame = camera.read()
            if not success or frame is None:
                return None
                
            return frame
        except Exception as e:
            ErrorHandler.handle_error(e, "Reading frame")
            return None

    @staticmethod
    def encode_frame(frame, quality=90):
        """Unified frame encoding"""
        try:
            if frame is None:
                return None
                
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ret:
                return None
                
            return jpeg.tobytes()
        except Exception as e:
            ErrorHandler.handle_error(e, "Encoding frame")
            return None

    @staticmethod
    def draw_text(frame, text, position, color=(0, 255, 0), scale=0.7, thickness=2):
        """Unified text drawing on frame"""
        try:
            if frame is None:
                return
                
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        except Exception as e:
            ErrorHandler.handle_error(e, "Drawing text on frame")

    @staticmethod
    def draw_rectangle(frame, start_point, end_point, color=(0, 255, 0), thickness=2):
        """Unified rectangle drawing on frame"""
        try:
            if frame is None:
                return
                
            cv2.rectangle(frame, start_point, end_point, color, thickness)
        except Exception as e:
            ErrorHandler.handle_error(e, "Drawing rectangle on frame") 