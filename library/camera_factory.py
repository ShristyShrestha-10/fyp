from abc import ABC, abstractmethod
from .cameras import IDCardScanner, BorrowCamera, ReturnCamera, StudentLoginCamera
from .utils import ErrorHandler
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class CameraStrategy(ABC):
    @abstractmethod
    def process_frame(self, frame):
        pass

class IDCardStrategy(CameraStrategy):
    def __init__(self, processor):
        self.processor = processor

    def process_frame(self, frame):
        # ID card specific processing
        pass

class BorrowStrategy(CameraStrategy):
    def __init__(self, book_detector):
        self.book_detector = book_detector

    def process_frame(self, frame):
        # Borrow specific processing
        pass

class ReturnStrategy(CameraStrategy):
    def __init__(self, book_detector):
        self.book_detector = book_detector

    def process_frame(self, frame):
        # Return specific processing
        pass

class StudentLoginStrategy(CameraStrategy):
    def __init__(self):
        pass

    def process_frame(self, frame):
        # Student login specific processing
        pass

class CameraFactory:
    """Factory class for creating and managing camera instances"""
    
    # Dictionary to store active camera instances
    _active_cameras = {}
    
    @classmethod
    def create_camera(cls, camera_type='id_card'):
        """Create a new camera instance of the specified type"""
        try:
            # If there's already an active camera of this type, return it
            if camera_type in cls._active_cameras:
                if cls._active_cameras[camera_type].is_running:
                    logger.info(f"Returning existing active camera of type {camera_type}")
                    return cls._active_cameras[camera_type]
                else:
                    # If camera exists but not running, remove it
                    logger.info(f"Removing inactive camera of type {camera_type}")
                    cls.remove_camera(camera_type)
            
            # Create new camera instance based on type
            if camera_type == 'id_card':
                camera = IDCardScanner()
            elif camera_type == 'borrow':
                camera = BorrowCamera()
            elif camera_type == 'return':
                camera = ReturnCamera()
            elif camera_type == 'student_login':
                camera = StudentLoginCamera()
            else:
                raise ValueError(f"Invalid camera type: {camera_type}")
            
            # Store the camera instance
            cls._active_cameras[camera_type] = camera
            logger.info(f"Created new camera of type {camera_type}")
            return camera
            
        except Exception as e:
            logger.error(f"Error creating camera: {str(e)}")
            return None
    
    @classmethod
    def get_active_camera(cls, camera_type):
        """Get the active camera instance of the specified type"""
        return cls._active_cameras.get(camera_type)
    
    @classmethod
    def remove_camera(cls, camera_type):
        """Remove a camera instance from active cameras"""
        try:
            if camera_type in cls._active_cameras:
                # Get the camera instance
                camera = cls._active_cameras[camera_type]
                
                # Stop the camera if it's running
                if camera.is_running:
                    camera.stop()
                
                # Release video capture if it exists
                if hasattr(camera, 'video') and camera.video is not None:
                    camera.video.release()
                    camera.video = None
                
                # Remove from active cameras
                del cls._active_cameras[camera_type]
                logger.info(f"Removed camera of type {camera_type} from active cameras")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing camera: {str(e)}")
            return False
    
    @classmethod
    def stop_all_cameras(cls):
        """Stop and remove all active cameras"""
        try:
            camera_types = list(cls._active_cameras.keys())
            for camera_type in camera_types:
                cls.remove_camera(camera_type)
            logger.info("All cameras stopped and removed")
            return True
        except Exception as e:
            logger.error(f"Error stopping all cameras: {str(e)}")
            return False
        
    @staticmethod
    def create_strategy(camera_type, **kwargs):
        try:
            if camera_type == 'id_card':
                return IDCardStrategy(kwargs.get('processor'))
            elif camera_type == 'borrow':
                return BorrowStrategy(kwargs.get('book_detector'))
            elif camera_type == 'return':
                return ReturnStrategy(kwargs.get('book_detector'))
            elif camera_type == 'student_login':
                return StudentLoginStrategy()
            else:
                raise ValueError(f"Unknown camera type: {camera_type}")
        except Exception as e:
            ErrorHandler.handle_error(e, f"Creating strategy for camera type {camera_type}", raise_exception=True) 