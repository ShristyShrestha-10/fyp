import cv2
import numpy as np
import os
from datetime import datetime
import logging
import easyocr
from .utils import (
    FileManager,
    MemoryManager,
    ErrorHandler,
    FrameProcessor
)
from django.utils import timezone
from .models import Student

logger = logging.getLogger(__name__)

class IDCardProcessor:
    def __init__(self):
        # Check for CUDA availability using OpenCV
        self.device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Clear GPU memory
        MemoryManager.clear_gpu_memory()
            
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'], gpu=self.device == "cuda")
        
        self.capture_timer = 5  # seconds for each mode
        self.start_time = None
        self.captured = False
        self.mode = "id_card"  # Can be "id_card" or "face"
        self.box_coords = None  # Coordinates for face detection box
        
    def __del__(self):
        """Cleanup when object is destroyed"""
        MemoryManager.clear_gpu_memory()
        
    def switch_to_face_mode(self):
        """Switch to face detection mode"""
        self.mode = "face"
        self.start_time = None
        self.captured = False
        logger.info("Switched to face detection mode")
        
    def set_box_coords(self, x, y, width, height):
        """Set the coordinates for the face detection box"""
        self.box_coords = (x, y, width, height)
        logger.info(f"Box coordinates set to: {self.box_coords}")
        
    def start_capture_timer(self):
        """Start the capture timer"""
        if self.start_time is None:
            self.start_time = datetime.now().timestamp()
            self.captured = False
            logger.info("Capture timer started")
    
    def should_capture(self):
        """Check if it's time to capture the image"""
        if self.start_time is None or self.captured:
            return False
        
        elapsed = datetime.now().timestamp() - self.start_time
        if elapsed >= self.capture_timer:
            self.captured = True
            logger.info("Capture triggered")
            return True
        return False
    
    def save_images(self, frame):
        """Save the captured image"""
        try:
            # Generate timestamp for filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save full frame
            full_image_path = FileManager.get_media_path(
                'id_cards',
                f'id_card_{timestamp}.jpg'
            )
            cv2.imwrite(full_image_path, frame)
            
            # Create a cropped version (assuming the ID card is in the center of the frame)
            height, width = frame.shape[:2]
            # Crop the center 70% of the image
            crop_width = int(width * 0.7)
            crop_height = int(height * 0.7)
            x = (width - crop_width) // 2
            y = (height - crop_height) // 2
            
            # Set box coordinates for face detection
            self.set_box_coords(x, y, crop_width, crop_height)
            
            cropped_frame = frame[y:y+crop_height, x:x+crop_width]
            
            # Save cropped frame
            cropped_image_path = FileManager.get_media_path(
                'id_cards',
                f'id_card_cropped_{timestamp}.jpg'
            )
            cv2.imwrite(cropped_image_path, cropped_frame)
            
            logger.info(f"Images saved: {full_image_path} and {cropped_image_path}")
            return full_image_path, cropped_image_path
            
        except Exception as e:
            ErrorHandler.handle_error(e, "Saving images")
            return None, None
    
    def perform_ocr(self, image_path):
        """Perform OCR on the ID card image using EasyOCR"""
        try:
            if not os.path.exists(image_path):
                ErrorHandler.handle_error(f"Image file not found: {image_path}", "OCR")
                return None, None
                
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image")
            
            # Preprocess the image
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Perform OCR
            results = self.reader.readtext(thresh)
            
            # Extract text with confidence scores
            extracted_text = []
            for detection in results:
                text = detection[1]
                confidence = detection[2]
                if confidence > 0.4:  # Only keep high confidence detections
                    extracted_text.append(text)
            
            # Join all text with newlines
            final_text = '\n'.join(extracted_text)
            
            # Save text to file
            text_file_path = image_path.replace('.jpg', '.txt')
            with open(text_file_path, 'w') as f:
                f.write(final_text)
            
            logger.info(f"OCR completed successfully. Text saved to: {text_file_path}")
            return final_text, text_file_path
            
        except Exception as e:
            ErrorHandler.handle_error(e, "OCR processing")
            return None, None

    def process_id_card(self, image_path):
        """Process ID card image and extract student details"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image")
            
            # Extract text from ID card
            extracted_text = self.perform_ocr(image)
            if not extracted_text:
                raise ValueError("Failed to extract text from ID card")
            
            # Extract student details
            student_details = self.extract_student_details(extracted_text)
            
            # Check if both name and ID are present
            if not student_details.get('name') or not student_details.get('id'):
                raise ValueError("Name and ID are required fields")
            
            # Detect and extract face from ID card
            face_encoding = self.detect_face(image)
            if face_encoding is None:
                raise ValueError("No face detected in ID card")
            
            # Save processed image with student details
            processed_image_path = self.save_processed_image(image, student_details)
            
            # Create student record with only name and ID
            student = Student.objects.create(
                name=student_details['name'],
                student_id=student_details['id'],
                face_encoding=face_encoding,
                face_verified=False,  # Will be verified when student shows up in person
                registered_at=timezone.now()
            )
            
            # Save ID card image path
            student.id_card_image = processed_image_path
            student.save()
            
            return student
            
        except Exception as e:
            logger.error(f"Error processing ID card: {str(e)}")
            raise