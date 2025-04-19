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
from django.utils import timezone

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
        """Initialize camera with multiple attempts and better device handling"""
        # First try to list available cameras
        available_cameras = []
        
        # Check common video device paths
        device_paths = [
            '/dev/video0',
            '/dev/video1',
            '/dev/video2',
            '/dev/video3',
            '/dev/video4'
        ]
        
        # Try device paths first
        for device_path in device_paths:
            try:
                if os.path.exists(device_path):
                    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            available_cameras.append(device_path)
                            logger.info(f"Found working camera at {device_path}")
                        cap.release()
            except Exception as e:
                logger.debug(f"Camera at {device_path} not available: {str(e)}")

        # If no devices found by path, try indices
        if not available_cameras:
            for i in range(5):  # Check first 5 indices
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            available_cameras.append(i)
                            logger.info(f"Found working camera at index {i}")
                        cap.release()
                except Exception as e:
                    logger.debug(f"Camera index {i} not available: {str(e)}")

        if not available_cameras:
            raise Exception("No cameras found. Please check your camera connection and permissions.")

        # Try to open cameras from the available list
        for camera in available_cameras:
            try:
                if self.video is not None:
                    self.video.release()
                
                if isinstance(camera, str):
                    # Open by device path
                    self.video = cv2.VideoCapture(camera, cv2.CAP_V4L2)
                else:
                    # Open by index
                    self.video = cv2.VideoCapture(camera)
                
                if self.video.isOpened():
                    ret, frame = self.video.read()
                    if ret and frame is not None:
                        logger.info(f"Successfully opened camera: {camera}")
                        # Set permissions if using device path
                        if isinstance(camera, str):
                            try:
                                import subprocess
                                subprocess.run(['sudo', 'chmod', '666', camera], check=True)
                                logger.info(f"Set permissions for {camera}")
                            except Exception as e:
                                logger.warning(f"Could not set permissions for {camera}: {str(e)}")
                        return
                    else:
                        self.video.release()
                        self.video = None
            except Exception as e:
                ErrorHandler.handle_error(e, f"Opening camera {camera}")
                if self.video is not None:
                    self.video.release()
                    self.video = None

        if self.video is None or not self.video.isOpened():
            raise Exception("Could not open any camera. Please check your camera connection and permissions.")

    def _configure_camera(self):
        """Configure camera properties with error handling"""
        try:
            # Wait for camera to initialize
            time.sleep(2)
            
            # Try to set camera properties
            properties = {
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_BUFFERSIZE: 1,
                cv2.CAP_PROP_FPS: 30,
                cv2.CAP_PROP_AUTOFOCUS: 1,  # Enable autofocus if available
                cv2.CAP_PROP_BRIGHTNESS: 128,  # Set middle brightness
                cv2.CAP_PROP_CONTRAST: 128,  # Set middle contrast
            }
            
            for prop, value in properties.items():
                try:
                    success = self.video.set(prop, value)
                    if not success:
                        logger.warning(f"Could not set camera property {prop} to {value}")
                except Exception as e:
                    logger.warning(f"Error setting camera property {prop}: {str(e)}")
            
            # Verify camera is still working after configuration
            ret, frame = self.video.read()
            if not ret or frame is None:
                raise Exception("Camera stopped working after configuration")
                
            logger.info("Camera configured successfully")
        except Exception as e:
            ErrorHandler.handle_error(e, "Configuring camera")
            raise

    def get_frame(self):
        try:
            if not self.is_running or self.video is None:
                return None

            # Add retry logic for frame capture
            max_retries = 3
            for attempt in range(max_retries):
                frame = FrameProcessor.read_frame(self.video)
                if frame is not None:
                    return FrameProcessor.encode_frame(frame)
                time.sleep(0.1)  # Short delay between retries
                
            logger.error("Failed to capture frame after multiple attempts")
            return None
        except Exception as e:
            ErrorHandler.handle_error(e, "Getting frame")
            return None

    def stop(self):
        """Stop the camera and release resources with proper cleanup"""
        try:
            # First, set is_running to False to stop any ongoing frame generation
            self.is_running = False
            
            # Then release camera if it exists
            if self.video is not None:
                try:
                    # Flush any pending frames
                    for _ in range(5):
                        self.video.grab()
                    
                    self.video.release()
                    logger.info("Camera resources released successfully")
                except Exception as e:
                    logger.error(f"Error releasing camera resources: {str(e)}")
                finally:
                    # Always set video to None
                    self.video = None
            
            logger.info("Camera stopped and resources released")
        except Exception as e:
            logger.error(f"Error stopping camera: {str(e)}")
            # Still mark as not running even if there was an error
            self.is_running = False
            self.video = None

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
        self.student_created = None
        self.is_duplicate = False  # New flag to track duplicate status

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
                    # Save the real person's face image
                    face_path = FileManager.get_media_path(
                        'faces',
                        FileManager.generate_timestamp_filename('face', 'jpg')
                    )
                    cv2.imwrite(face_path, face_image)
                    self.face_detected = True
                    self.face_image = face_path
                    self.face_saved = True
                    
                    # If we have a student record, update it with the face data
                    if self.student_created:
                        try:
                            # Get face encoding
                            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                            encodings = face_recognition.face_encodings(rgb_face)
                            if encodings:
                                # Save face encoding to student record
                                face_encoding_bytes = pickle.dumps(encodings[0])
                                self.student_created.face_encoding = face_encoding_bytes
                                self.student_created.face_verified = True
                                
                                # Set the real person's face image path
                                relative_path = face_path.replace(settings.MEDIA_ROOT + '/', '')
                                
                                # Store in both face_image and id_card_image for backward compatibility
                                if hasattr(self.student_created, 'face_image'):
                                    self.student_created.face_image = relative_path
                                
                                # Also update id_card_image as fallback
                                self.student_created.id_card_image = relative_path
                                
                                # Save the student record
                                self.student_created.save()
                                
                                logger.info(f"Updated student {self.student_created.name} with face image at {relative_path}")
                        except Exception as e:
                            ErrorHandler.handle_error(e, "Saving face image to student record")

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

            # If duplicate detected, stop processing
            if self.is_duplicate:
                self.stop()
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
        # If duplicate detected, don't process further
        if self.is_duplicate:
            return

        self.processor.start_capture_timer()
        if self.processor.should_capture():
            if not self.id_card_captured:
                self._process_id_card(frame)
            else:
                self._process_face(frame)

    def _process_id_card(self, frame):
        full_path, cropped_path = self.processor.save_images(frame)
        if cropped_path:
            # Perform OCR on the cropped image
            ocr_results, text_file = self.processor.perform_ocr(cropped_path)
            self.id_card_captured = True
                
            # Process the ID card to create a student record
            try:
                self.student_created = self.processor.process_id_card(cropped_path)
                if self.student_created:
                    # Check if this was a duplicate registration
                    if hasattr(self.student_created, 'is_duplicate') and self.student_created.is_duplicate:
                        logger.info(f"Duplicate student detected: {self.student_created.name}")
                        self.is_duplicate = True
                        self.capture_complete = True
                        self.stop()
                        return
                    
                    logger.info(f"Created student record: {self.student_created.name}")
                    # Only switch to face mode for new registrations
                    self.processor.switch_to_face_mode()
                else:
                    self.error_message = "Failed to process ID card"
                    self.processor.start_time = None
            except Exception as e:
                logger.error(f"Error processing ID card: {str(e)}")
                self.error_message = "Failed to process ID card"
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
            FrameProcessor.draw_rectangle(frame, (x, y), (x + box_width, y + box_height), color=(0, 255, 0))
            FrameProcessor.draw_text(frame, "Place your face here", (x, y - 10))

        if self.error_message:
            FrameProcessor.draw_text(frame, self.error_message, (10, 60), color=(0, 0, 255))

        if self.student_created:
            FrameProcessor.draw_text(frame, f"Name: {self.student_created.name}", (10, 90), color=(0, 255, 0))

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
        self.student_session = None  # Store student session

    def set_student_session(self, student_id):
        """Set the student session for book borrowing"""
        self.student_session = student_id
        logger.info(f"Set student session in camera: {student_id}")

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
        if self.student_session:
            FrameProcessor.draw_text(frame, f"Student ID: {self.student_session}", (10, 90), color=(0, 255, 0))

    def _process_book_detection(self, tag_id):
        """Process a detected book tag and trigger the borrowing process"""
        from library.views import store_detected_book, tag_to_book_mapping
        from django.http import HttpRequest
        from django.contrib.sessions.middleware import SessionMiddleware
        import json
        
        if tag_id in tag_to_book_mapping:
            book_info = tag_to_book_mapping[tag_id]
            book = store_detected_book(tag_id, book_info)
            if book:
                logger.info(f"Detected and processed book with tag ID: {tag_id}")
                
                if not self.student_session:
                    logger.warning("No student session available for book borrowing")
                    return
                
                # Create a request with the student session
                request = HttpRequest()
                request.method = 'POST'
                request.POST = {'tag_id': str(tag_id)}
                
                # Add session middleware
                middleware = SessionMiddleware(lambda x: None)
                middleware.process_request(request)
                
                # Set the student session
                request.session['student_id'] = self.student_session
                request.session.save()
                
                # Import the view function here to avoid circular imports
                from library.views import process_detected_book
                try:
                    # Call the process_detected_book view
                    response = process_detected_book(request)
                    response_data = json.loads(response.content)
                    logger.info(f"Book detection processed: {response_data.get('message', 'Unknown response')}")
                except Exception as e:
                    logger.error(f"Error processing book detection: {str(e)}")
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
        self.matched_student = None
        self.id_card_captured = False  # Flag to track if ID card was captured
        self.best_similarity_score = 0  # Track similarity score for face matches
        self.id_card_capture_time = None  # To track when ID card was captured for delay
        self.ready_for_face_recognition = False  # Flag to indicate when face recognition should start
        
        # Set initial box coordinates for face detection in the center of the frame
        height, width = 480, 640  # Default camera resolution
        box_size = min(height, width) // 2
        x = (width - box_size) // 2
        y = (height - box_size) // 2
        self.processor.set_box_coords(x, y, box_size, box_size)
        
    def detect_and_match_face(self, frame):
        """Detect face and match with registered students"""
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
                    # Save the face image temporarily for matching
                    face_path = FileManager.get_media_path(
                        'faces',
                        FileManager.generate_timestamp_filename('login_face', 'jpg')
                    )
                    cv2.imwrite(face_path, face_image)
                    self.face_detected = True
                    self.face_image = face_path
                    self.face_saved = True
                    
                    # Try to match this face against our database of students
                    match_result = self._match_face(face_image)
                    
                    # Only set capture_complete if we have a match with sufficient similarity score
                    if match_result and hasattr(self, 'best_similarity_score') and self.best_similarity_score >= 0.5:
                        return True
                
                # Draw a rectangle around the detected face
                FrameProcessor.draw_rectangle(
                    frame,
                    (x + left, y + top),
                    (x + right, y + bottom),
                    color=(0, 255, 0)
                )
                return True
            return False
        except Exception as e:
            ErrorHandler.handle_error(e, "Face detection during login")
            return False
    
    def process_id_card(self, frame):
        """Process ID card to extract student details during login"""
        try:
            # Save the ID card image
            full_path, cropped_path = self.processor.save_images(frame)
            if cropped_path:
                # Extract text from the ID card
                ocr_results, text_file = self.processor.perform_ocr(cropped_path)
                if ocr_results:
                    # Extract student name only - we don't create a new record but look for matching student
                    student_details = self.processor.extract_student_details(ocr_results)
                    
                    if student_details.get('name'):
                        logger.info(f"Extracted name from ID card: {student_details['name']}")
                        
                        # Try to find matching student in the database by name
                        try:
                            from .models import Student
                            name_parts = student_details['name'].split()
                            # Try exact name match first
                            student = Student.objects.filter(name__iexact=student_details['name']).first()
                            
                            # If no exact match, try with first_name and last_name
                            if not student and len(name_parts) >= 2:
                                first_name = name_parts[0]
                                last_name = ' '.join(name_parts[1:])
                                student = Student.objects.filter(
                                    first_name__iexact=first_name, 
                                    last_name__iexact=last_name
                                ).first()
                            
                            if student:
                                self.matched_student = student
                                logger.info(f"Found matching student: {student.name} (ID: {student.student_id})")
                                return True
                            else:
                                self.error_message = "No matching student found in the database."
                                logger.warning(f"No student found matching name: {student_details['name']}")
                                return False
                        except Exception as e:
                            logger.error(f"Error finding student by name: {e}")
                            return False
                    else:
                        self.error_message = "Could not extract name from ID card."
                        return False
                else:
                    self.error_message = "Could not extract text from ID card."
                    return False
            else:
                self.error_message = "Failed to save ID card images."
                return False
        except Exception as e:
            ErrorHandler.handle_error(e, "Processing ID card during login")
            return False
            
    def _match_face(self, face_image):
        """Try to match the detected face against verified students"""
        try:
            # If we already matched a student by name, verify their face
            if self.matched_student:
                if not self.matched_student.face_encoding:
                    self.error_message = "Student has no registered face. Please visit the library desk."
                    return False
                
                # Get face encoding for the detected face
                rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                unknown_encoding = face_recognition.face_encodings(rgb_face)
                
                if not unknown_encoding:
                    self.error_message = "Could not extract face features. Please try again."
                    return False
                    
                # Compare with stored encoding
                try:
                    known_encoding = pickle.loads(self.matched_student.face_encoding)
                    face_distance = face_recognition.face_distance([known_encoding], unknown_encoding[0])[0]
                    
                    # Calculate similarity score (1 - distance)
                    similarity_score = 1.0 - face_distance
                    self.best_similarity_score = similarity_score  # Store for API access
                    
                    # Check if the face matches (lower distance is better)
                    if face_distance < 0.5:  # Threshold for face matching is 0.5
                        logger.info(f"Face verification successful for student: {self.matched_student.name} with similarity score: {similarity_score:.2f}")
                        
                        # Record successful login in StudentLogin table
                        from .models import StudentLogin
                        from django.utils import timezone
                        
                        # Create or update login record
                        StudentLogin.objects.create(
                            student=self.matched_student,
                            login_time=timezone.now(),
                            status="Verified",
                            similarity_score=similarity_score,
                            is_active=True
                        )
                        
                        return True
                    else:
                        self.error_message = f"Face verification failed. Similarity score: {similarity_score:.2f} (required: > 0.5)"
                        self.matched_student = None  # Clear the match as face didn't verify
                        return False
                except Exception as e:
                    logger.error(f"Error comparing face encodings: {e}")
                    self.error_message = "Error during face verification."
                    return False
            
            # Otherwise, search all verified students
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            unknown_encoding = face_recognition.face_encodings(rgb_face)
            
            if not unknown_encoding:
                self.error_message = "Could not extract face features. Please try again."
                return False
                
            unknown_encoding = unknown_encoding[0]
            
            # Get all students with verified faces
            from .models import Student, StudentLogin
            from django.utils import timezone
            verified_students = Student.objects.filter(face_verified=True).exclude(face_encoding__isnull=True)
            
            best_match = None
            best_match_distance = 0.5  # Threshold for face matching is 0.5
            best_similarity_score = 0
            
            for student in verified_students:
                try:
                    # Extract the stored encoding
                    if student.face_encoding:
                        known_encoding = pickle.loads(student.face_encoding)
                        
                        # Calculate face distance
                        face_distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
                        similarity_score = 1.0 - face_distance
                        
                        # Lower distance means better match
                        if face_distance < best_match_distance:
                            best_match = student
                            best_match_distance = face_distance
                            best_similarity_score = similarity_score
                except Exception as e:
                    logger.warning(f"Error matching face for student {student.name}: {str(e)}")
                    continue
            
            if best_match:
                self.matched_student = best_match
                self.best_similarity_score = best_similarity_score  # Store for API access
                logger.info(f"Face matched to student: {best_match.name} (ID: {best_match.student_id}) with similarity score: {best_similarity_score:.2f}")
                
                # Record login in StudentLogin table with appropriate status based on similarity score
                status = "Verified" if best_similarity_score >= 0.5 else "Not Verified"
                
                StudentLogin.objects.create(
                    student=best_match,
                    login_time=timezone.now(),
                    status=status,
                    similarity_score=best_similarity_score,
                    is_active=True
                )
                
                return True
            else:
                self.error_message = "Face not recognized. Please try again or contact library staff."
                logger.info("No matching student found for the detected face")
                return False
                
        except Exception as e:
            ErrorHandler.handle_error(e, "Matching face during login")
            self.error_message = "Error during face recognition. Please try again."
            return False

    def get_frame(self):
        try:
            if not self.is_running or self.video is None:
                return None

            frame = FrameProcessor.read_frame(self.video)
            if frame is None:
                return None

            if not self.capture_complete:
                self._process_login(frame)

            self._draw_overlays(frame)
            return FrameProcessor.encode_frame(frame)
        except Exception as e:
            ErrorHandler.handle_error(e, "Student login camera frame processing")
            return None
            
    def _process_login(self, frame):
        self.processor.start_capture_timer()
        
        current_time = time.time()
        
        # First capture ID card if not already done
        if not self.id_card_captured and self.processor.should_capture():
            result = self.process_id_card(frame)
            if result:
                self.id_card_captured = True
                self.id_card_capture_time = current_time  # Record when ID card was captured
                self.processor.switch_to_face_mode()  # Switch to face mode after ID captured
                self.processor.start_time = None  # Reset timer for face capture
                self.ready_for_face_recognition = False  # Not ready for face recognition yet
            else:
                self.processor.start_time = None  # Reset timer if ID card capture failed
        
        # Check if 3 seconds have passed since ID card capture
        elif self.id_card_captured and not self.ready_for_face_recognition:
            if self.id_card_capture_time and (current_time - self.id_card_capture_time >= 3.0):
                self.ready_for_face_recognition = True
                logger.info("3-second delay complete, starting face recognition")
        
        # Then capture face for verification if ID card was captured and 3-second delay has passed
        elif self.id_card_captured and self.ready_for_face_recognition and (self.processor.should_capture() or not self.face_saved):
            face_match_result = self.detect_and_match_face(frame)
            
            if face_match_result:
                # Check if we have a matched student with sufficient similarity score
                if self.matched_student and hasattr(self, 'best_similarity_score') and self.best_similarity_score >= 0.5:
                    self.capture_complete = True
                elif self.face_saved and (not hasattr(self, 'best_similarity_score') or self.best_similarity_score < 0.5):
                    # Face was detected but similarity score was too low
                    self.error_message = f"Face verification failed. Similarity score: {getattr(self, 'best_similarity_score', 0):.2f} (required: >= 0.5)"
                    self.processor.start_time = None
            elif self.error_message is None:
                self.error_message = "No face detected. Please position your face in the green box."
                self.processor.start_time = None

    def _draw_overlays(self, frame):
        # Draw box for face positioning
        if self.box_coords and not self.capture_complete:
            x, y, box_width, box_height = self.box_coords
            FrameProcessor.draw_rectangle(frame, (x, y), (x + box_width, y + box_height), color=(0, 255, 0))
            if not self.id_card_captured:
                FrameProcessor.draw_text(frame, "Position your ID card here", (x, y - 10))
            elif not self.ready_for_face_recognition:
                # Show countdown message
                if self.id_card_capture_time:
                    seconds_left = max(0, 3 - int(time.time() - self.id_card_capture_time))
                    FrameProcessor.draw_text(frame, f"Preparing face recognition... {seconds_left}s", (x, y - 10))
            else:
                FrameProcessor.draw_text(frame, "Position your face here", (x, y - 10))

        # Draw error message if any
        if self.error_message:
            FrameProcessor.draw_text(frame, self.error_message, (10, 60), color=(0, 0, 255))

        # Draw matched student info if available
        if self.matched_student:
            FrameProcessor.draw_text(frame, f"Welcome, {self.matched_student.name}!", (10, 90), color=(0, 255, 0))
            FrameProcessor.draw_text(frame, f"ID: {self.matched_student.student_id}", (10, 120), color=(0, 255, 0))

        # Draw status text
        status_text = (
            "Place your ID card in the box..." if not self.id_card_captured else
            f"Preparing face recognition..." if self.id_card_captured and not self.ready_for_face_recognition else
            "Now place your face in the box..." if not self.face_saved and self.ready_for_face_recognition else
            "Login successful!" if self.matched_student else
            "Login verification failed" if self.face_saved and not self.matched_student else
            "Processing..."
        )
        FrameProcessor.draw_text(frame, status_text, (10, 30)) 