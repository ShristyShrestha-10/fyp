from abc import ABC, abstractmethod
from .cameras import IDCardScanner, BorrowCamera, ReturnCamera
from .utils import ErrorHandler

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

class CameraFactory:
    @staticmethod
    def create_camera(camera_type, **kwargs):
        try:
            if camera_type == 'id_card':
                return IDCardScanner()
            elif camera_type == 'borrow':
                return BorrowCamera()
            elif camera_type == 'return':
                return ReturnCamera()
            else:
                raise ValueError(f"Unknown camera type: {camera_type}")
        except Exception as e:
            ErrorHandler.handle_error(e, f"Creating camera of type {camera_type}", raise_exception=True)

    @staticmethod
    def create_strategy(camera_type, **kwargs):
        try:
            if camera_type == 'id_card':
                return IDCardStrategy(kwargs.get('processor'))
            elif camera_type == 'borrow':
                return BorrowStrategy(kwargs.get('book_detector'))
            elif camera_type == 'return':
                return ReturnStrategy(kwargs.get('book_detector'))
            else:
                raise ValueError(f"Unknown camera type: {camera_type}")
        except Exception as e:
            ErrorHandler.handle_error(e, f"Creating strategy for camera type {camera_type}", raise_exception=True) 