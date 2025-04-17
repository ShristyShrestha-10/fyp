import cv2
import numpy as np
import logging
from datetime import datetime
import time
from .book_detector import BookDetector
from django.conf import settings
import os
import face_recognition
from .id_card_processor import IDCardProcessor
from .utils import (
    FileManager,
    MemoryManager,
    ErrorHandler,
    FrameProcessor
)
import pickle
from django.contrib.auth.models import User
from library.models import Student

# Set up logging
logger = logging.getLogger(__name__)

class BaseCamera:
    def __init__(self):
        try:
            self.video = None
            self.is_running = False
            self._initialize_camera()
            self._configure_camera()
            self.is_running = True
            logger.info("Camera initialized successfully")
        except Exception as e:
            ErrorHandler.handle_error(e, "Camera initialization", raise_exception=True)

    def _initialize_camera(self):
        """Initialize camera with multiple attempts"""
        # First try to list available cameras
        available_cameras = []
        for i in range(10):  # Check up to 10 possible camera indices
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append(i)
                        logger.info(f"Found working camera at index {i}")
                    cap.release()
            except Exception as e:
                logger.debug(f"Camera index {i} not available: {str(e)}")
                continue

        if not available_cameras:
            raise Exception("No cameras found. Please check your camera connection and permissions.")

        # Try to open the first available camera
        for cam_index in available_cameras:
            try:
                if self.video is not None:
                    self.video.release()
                self.video = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
                if self.video.isOpened():
                    ret, frame = self.video.read()
                    if ret and frame is not None:
                        logger.info(f"Successfully opened camera at index {cam_index}")
                        return
                    else:
                        self.video.release()
                        self.video = None
            except Exception as e:
                ErrorHandler.handle_error(e, f"Opening camera at index {cam_index}")
                if self.video is not None:
                    self.video.release()
                    self.video = None

        if self.video is None or not self.video.isOpened():
            raise Exception("Could not open any camera. Please check your camera connection and permissions.")

    def _configure_camera(self):
        """Configure camera properties"""
        try:
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.video.set(cv2.CAP_PROP_FPS, 30)
            time.sleep(1)  # Wait for camera to stabilize
        except Exception as e:
            ErrorHandler.handle_error(e, "Configuring camera")

    def get_frame(self):
        try:
            if not self.is_running or self.video is None:
                return None

            frame = FrameProcessor.read_frame(self.video)
            if frame is None:
                return None

            return FrameProcessor.encode_frame(frame)
        except Exception as e:
            ErrorHandler.handle_error(e, "Getting frame")
            return None

    def stop(self):
        """Stop the camera and release resources"""
        self.is_running = False
        if self.video is not None:
            self.video.release()
            self.video = None
        logger.info("Camera stopped and resources released")

    def __del__(self):
        self.stop()

class IDCardScanner(BaseCamera):
    def __init__(self):
        super().__init__()
        self.processor = IDCardProcessor()
        self.capture_complete = False
        self.ocr_results = None
        self.error_message = None
        self.face_detected = False
        self.face_image = None
        self.id_card_captured = False
        self.box_coords = None
        self.face_saved = False

    def detect_and_save_face(self, frame):
        try:
            self.box_coords = self.processor.box_coords
            if self.box_coords is None:
                return False

            x, y, box_width, box_height = self.box_coords
            face_roi = frame[y:y + box_height, x:x + box_width]
            rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_roi)

            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_image = face_roi[top:bottom, left:right]

                if (self.processor.should_capture() or not self.face_saved) and not self.face_detected:
                    face_path = FileManager.get_media_path(
                        'faces',
                        FileManager.generate_timestamp_filename('face', 'jpg')
                    )
                    cv2.imwrite(face_path, face_image)
                    self.face_detected = True
                    self.face_image = face_path
                    self.face_saved = True

                FrameProcessor.draw_rectangle(
                    frame,
                    (x + left, y + top),
                    (x + right, y + bottom)
                )
                return True
            return False
        except Exception as e:
            ErrorHandler.handle_error(e, "Face detection")
            return False

    def get_frame(self):
        try:
            if not self.is_running or self.video is None:
                return None

            frame = FrameProcessor.read_frame(self.video)
            if frame is None:
                return None

            if not self.capture_complete:
                self._process_capture(frame)

            self._draw_overlays(frame)
            return FrameProcessor.encode_frame(frame)
        except Exception as e:
            ErrorHandler.handle_error(e, "ID card scanner frame processing")
            return None

    def _process_capture(self, frame):
        self.processor.start_capture_timer()
        if self.processor.should_capture():
            if not self.id_card_captured:
                self._process_id_card(frame)
            else:
                self._process_face(frame)

    def _process_id_card(self, frame):
        full_path, cropped_path = self.processor.save_images(frame)
        if cropped_path:
            self.ocr_results, text_file = self.processor.perform_ocr(cropped_path)
            if self.ocr_results:
                self.id_card_captured = True
                self.processor.switch_to_face_mode()
                self.face_saved = False
            else:
                self.error_message = "Failed to extract text from ID card"
                self.processor.start_time = None
        else:
            self.error_message = "Failed to save ID card images"
            self.processor.start_time = None

    def _process_face(self, frame):
        if self.detect_and_save_face(frame):
            self.capture_complete = True
        else:
            self.error_message = "No face detected. Please position your face in the green box."
            self.processor.start_time = None

    def _draw_overlays(self, frame):
        if self.id_card_captured and not self.capture_complete and self.box_coords:
            x, y, box_width, box_height = self.box_coords
            FrameProcessor.draw_rectangle(frame, (x, y), (x + box_width, y + box_height))
            FrameProcessor.draw_text(frame, "Place your face here", (x, y - 10))

        if self.error_message:
            FrameProcessor.draw_text(frame, self.error_message, (10, 60), color=(0, 0, 255))

        status_text = (
            "Capturing ID card..." if not self.id_card_captured else
            "Now capturing face..." if not self.face_saved else
            "Capture complete!"
        )
        FrameProcessor.draw_text(frame, status_text, (10, 30))

class BorrowCamera(BaseCamera):
    def __init__(self):
        super().__init__()
        self.book_detector = BookDetector()
        self.last_detection_time = {}
        self.detection_cooldown = 2.0
        self.detected_books = set()  # Keep track of detected books in current session

    def get_frame(self):
        try:
            if not self.is_running or self.video is None:
                return None

            frame = FrameProcessor.read_frame(self.video)
            if frame is None:
                return None

            # Process AprilTag detections
            current_time = time.time()
            detections = self.book_detector.detect_books(frame)
            
            for detection in detections:
                tag_id = detection['tag_id']
                # Draw bounding box around detected tag
                if 'bbox' in detection:
                    bbox = detection['bbox']
                    FrameProcessor.draw_rectangle(
                        frame,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        color=(0, 255, 0)
                    )
                
                # Process book if cooldown has elapsed
                if (tag_id not in self.last_detection_time or 
                    current_time - self.last_detection_time[tag_id] >= self.detection_cooldown):
                    self.last_detection_time[tag_id] = current_time
                    self._process_book_detection(tag_id)
                    if tag_id not in self.detected_books:
                        self.detected_books.add(tag_id)
                        # Draw book info on frame
                        from library.views import tag_to_book_mapping
                        if tag_id in tag_to_book_mapping:
                            book_info = tag_to_book_mapping[tag_id]
                            y_pos = 90  # Start position for text
                            FrameProcessor.draw_text(
                                frame,
                                f"Detected Book: {book_info.get('title', 'Unknown')}",
                                (10, y_pos),
                                color=(0, 255, 0)
                            )
                            if 'author' in book_info:
                                FrameProcessor.draw_text(
                                    frame,
                                    f"Author: {book_info['author']}",
                                    (10, y_pos + 30),
                                    color=(0, 255, 0)
                                )

            self._draw_overlays(frame)
            return FrameProcessor.encode_frame(frame)
        except Exception as e:
            ErrorHandler.handle_error(e, "Borrow camera frame processing")
            return None

    def _draw_overlays(self, frame):
        height, width = frame.shape[:2]
        FrameProcessor.draw_text(frame, "Borrow Book Scanner", (10, 30))
        FrameProcessor.draw_text(frame, "Press 'Q' to close", (10, height - 10))
        FrameProcessor.draw_text(frame, f"Books detected: {len(self.detected_books)}", (10, 60))

    def _process_book_detection(self, tag_id):
        from library.views import store_detected_book, tag_to_book_mapping
        if tag_id in tag_to_book_mapping:
            book_info = tag_to_book_mapping[tag_id]
            store_detected_book(tag_id, book_info)
            logger.info(f"Detected and processed book with tag ID: {tag_id}")
        else:
            logger.warning(f"Detected unknown tag ID: {tag_id}")

class ReturnCamera(BorrowCamera):
    def __init__(self):
        super().__init__()
        self.detection_cooldown = 2.0

    def _draw_overlays(self, frame):
        height, width = frame.shape[:2]
        FrameProcessor.draw_text(frame, "Return Book Scanner", (10, 30))
        FrameProcessor.draw_text(frame, "Press 'Q' to close", (10, height - 10))

    def _process_book_detection(self, tag_id):
        from library.views import store_detected_book, tag_to_book_mapping
        if tag_id in tag_to_book_mapping:
            book_info = tag_to_book_mapping[tag_id]
            store_detected_book(tag_id, book_info)
            logger.info(f"Detected and processed returned book with tag ID: {tag_id}")
        else:
            logger.warning(f"Detected unknown tag ID: {tag_id}")

class StudentLoginCamera(BaseCamera):
    def __init__(self):
        super().__init__()
        self.processor = IDCardProcessor()
        self.capture_complete = False
        self.ocr_results = None
        self.error_message = None
        self.face_detected = False
        self.face_image = None
        self.box_coords = None
        self.face_saved = False
        self.id_card_captured = False
        
    def detect_and_save_face(self, frame):
        try:
            self.box_coords = self.processor.box_coords
            if self.box_coords is None:
                return False

            x, y, box_width, box_height = self.box_coords
            face_roi = frame[y:y + box_height, x:x + box_width]
            rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_roi)

            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_image = face_roi[top:bottom, left:right]

                if (self.processor.should_capture() or not self.face_saved) and not self.face_detected:
                    face_path = FileManager.get_media_path(
                        'faces',
                        FileManager.generate_timestamp_filename('face', 'jpg')
                    )
                    cv2.imwrite(face_path, face_image)
                    self.face_detected = True
                    self.face_image = face_path
                    self.face_saved = True

                FrameProcessor.draw_rectangle(
                    frame,
                    (x + left, y + top),
                    (x + right, y + bottom)
                )
                return True
            return False
        except Exception as e:
            ErrorHandler.handle_error(e, "Face detection")
            return False

    def get_frame(self):
        try:
            if not self.is_running or self.video is None:
                return None

            frame = FrameProcessor.read_frame(self.video)
            if frame is None:
                return None

            if not self.capture_complete:
                self._process_capture(frame)

            self._draw_overlays(frame)
            return FrameProcessor.encode_frame(frame)
        except Exception as e:
            ErrorHandler.handle_error(e, "Student login camera frame processing")
            return None

    def _process_capture(self, frame):
        self.processor.start_capture_timer()
        if self.processor.should_capture():
            if not self.id_card_captured:
                self._process_id_card(frame)
            else:
                self._process_face(frame)

    def _process_id_card(self, frame):
        full_path, cropped_path = self.processor.save_images(frame)
        if cropped_path:
            self.ocr_results, text_file = self.processor.perform_ocr(cropped_path)
            if self.ocr_results:
                self.id_card_captured = True
                self.processor.switch_to_face_mode()
                self.face_saved = False
            else:
                self.error_message = "Failed to extract text from ID card"
                self.processor.start_time = None
        else:
            self.error_message = "Failed to save ID card images"
            self.processor.start_time = None

    def _process_face(self, frame):
        if self.detect_and_save_face(frame):
            self.capture_complete = True
        else:
            self.error_message = "No face detected. Please position your face in the green box."
            self.processor.start_time = None

    def _draw_overlays(self, frame):
        if self.id_card_captured and not self.capture_complete and self.box_coords:
            x, y, box_width, box_height = self.box_coords
            FrameProcessor.draw_rectangle(frame, (x, y), (x + box_width, y + box_height))
            FrameProcessor.draw_text(frame, "Place your face here", (x, y - 10))

        if self.error_message:
            FrameProcessor.draw_text(frame, self.error_message, (10, 60), color=(0, 0, 255))

        status_text = (
            "Capturing ID card..." if not self.id_card_captured else
            "Now capturing face..." if not self.face_saved else
            "Capture complete!"
        )
        FrameProcessor.draw_text(frame, status_text, (10, 30)) 