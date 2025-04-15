import cv2
import numpy as np
import logging
from datetime import datetime
import time
from .book_detector import BookDetector
from django.conf import settings
import os

# Set up logging
logger = logging.getLogger(__name__)

class BaseCamera:
    def __init__(self):
        try:
            # Initialize camera with default index (0)
            self.video = None
            self.is_running = False
            
            # Try to open the camera with different indices
            for i in range(3):  # Try first 3 indices
                try:
                    self.video = cv2.VideoCapture(i)
                    if self.video.isOpened():
                        # Test if we can read from the camera
                        ret, frame = self.video.read()
                        if ret:
                            logger.info(f"Successfully opened camera at index {i}")
                            break
                        else:
                            self.video.release()
                            self.video = None
                except Exception as e:
                    logger.warning(f"Failed to open camera at index {i}: {str(e)}")
                    if self.video is not None:
                        self.video.release()
                        self.video = None
            
            if self.video is None or not self.video.isOpened():
                raise Exception("Could not open any camera")
            
            # Set camera properties
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.video.set(cv2.CAP_PROP_FPS, 30)
            
            # Wait for camera to stabilize
            time.sleep(1)
            
            self.is_running = True
            logger.info("Camera initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing camera: {str(e)}")
            if self.video is not None:
                self.video.release()
                self.video = None
            raise

    def get_frame(self):
        try:
            if not self.is_running or self.video is None:
                return None

            # Try to read frame multiple times if first attempt fails
            for _ in range(3):
                success, frame = self.video.read()
                if success and frame is not None:
                    break
                logger.warning("Failed to read frame, retrying...")
                time.sleep(0.1)  # Small delay between retries
                
            if not success or frame is None:
                logger.error("Failed to read frame from camera after multiple attempts")
                return None
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes() if ret else None
            
        except Exception as e:
            logger.error(f"Error getting frame: {str(e)}")
            return None

    def stop(self):
        """Stop the camera and release resources"""
        self.is_running = False
        if self.video is not None:
            self.video.release()
            self.video = None
        cv2.destroyAllWindows()
        logger.info("Camera stopped and resources released")

    def __del__(self):
        self.stop()

class IDCardScanner(BaseCamera):
    def get_frame(self):
        try:
            if not self.is_running or self.video is None:
                return None

            success, frame = self.video.read()
            if not success or frame is None:
                return None
            
            # Add ID card scanning overlay
            height, width = frame.shape[:2]
            overlay = frame.copy()
            
            # Draw ID card rectangle guide
            card_width = int(width * 0.6)
            card_height = int(height * 0.4)
            x = (width - card_width) // 2
            y = (height - card_height) // 2
            cv2.rectangle(overlay, (x, y), (x + card_width, y + card_height), (0, 255, 0), 2)
            
            # Add semi-transparent overlay
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Add instructions
            cv2.putText(frame, "Place ID card here", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'Q' to close scanner", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes() if ret else None
            
        except Exception as e:
            logger.error(f"Error in ID card scanner frame: {str(e)}")
            return None

class BorrowCamera(BaseCamera):
    def __init__(self):
        super().__init__()
        self.book_detector = BookDetector()
        self.last_detection_time = {}  # Track when each tag was last detected
        self.detection_cooldown = 2.0  # Seconds to wait before detecting same tag again
        
    def get_frame(self):
        try:
            if not self.is_running or self.video is None:
                return None

            success, frame = self.video.read()
            if not success or frame is None:
                return None
            
            # Add borrow book overlay
            height, width = frame.shape[:2]
            cv2.putText(frame, "Borrow Book Scanner", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'Q' to close", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Detect AprilTags
            current_time = time.time()
            detections = self.book_detector.detect_books(frame)
            
            for detection in detections:
                tag_id = detection['tag_id']
                
                # Check if enough time has passed since last detection of this tag
                if (tag_id not in self.last_detection_time or 
                    current_time - self.last_detection_time[tag_id] >= self.detection_cooldown):
                    
                    # Update last detection time
                    self.last_detection_time[tag_id] = current_time
                    
                    # Process the detected book
                    from library.views import store_detected_book, tag_to_book_mapping
                    if tag_id in tag_to_book_mapping:
                        book_info = tag_to_book_mapping[tag_id]
                        store_detected_book(tag_id, book_info)
                        logger.info(f"Detected and processed book with tag ID: {tag_id}")
                    else:
                        logger.warning(f"Detected unknown tag ID: {tag_id}")
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes() if ret else None
            
        except Exception as e:
            logger.error(f"Error in borrow camera frame: {str(e)}")
            return None

class ReturnCamera(BaseCamera):
    def __init__(self):
        super().__init__()
        self.book_detector = BookDetector()
        self.last_detection_time = {}
        self.detection_cooldown = 2.0
        
    def get_frame(self):
        try:
            if not self.is_running or self.video is None:
                return None

            success, frame = self.video.read()
            if not success or frame is None:
                return None
            
            # Add return book overlay
            height, width = frame.shape[:2]
            cv2.putText(frame, "Return Book Scanner", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'Q' to close", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Detect AprilTags
            current_time = time.time()
            detections = self.book_detector.detect_books(frame)
            
            for detection in detections:
                tag_id = detection['tag_id']
                
                # Check if enough time has passed since last detection of this tag
                if (tag_id not in self.last_detection_time or 
                    current_time - self.last_detection_time[tag_id] >= self.detection_cooldown):
                    
                    # Update last detection time
                    self.last_detection_time[tag_id] = current_time
                    
                    # Process the detected book
                    from library.views import store_detected_book, tag_to_book_mapping
                    if tag_id in tag_to_book_mapping:
                        book_info = tag_to_book_mapping[tag_id]
                        store_detected_book(tag_id, book_info)
                        logger.info(f"Detected and processed returned book with tag ID: {tag_id}")
                    else:
                        logger.warning(f"Detected unknown tag ID: {tag_id}")
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes() if ret else None
            
        except Exception as e:
            logger.error(f"Error in return camera frame: {str(e)}")
            return None 