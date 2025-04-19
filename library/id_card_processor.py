import cv2
import numpy as np
import os
from datetime import datetime
import logging
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from django.utils import timezone
from .models import Student
import re
from django.conf import settings
from .utils import (
    FileManager,
    MemoryManager,
    ErrorHandler,
    FrameProcessor
)

logger = logging.getLogger(__name__)

class IDCardProcessor:
    def __init__(self):
        # Initialize basic attributes
        self.capture_timer = 5  # seconds for each mode
        self.start_time = None
        self.captured = False
        self.mode = "id_card"  # Can be "id_card" or "face"
        self.box_coords = None  # Coordinates for face detection box
        self.model = None
        self.processor = None
        
        # Check for CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize SmolVLM model and processor
        try:
            self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
            self.model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-500M-Instruct",
                torch_dtype=torch.bfloat16,
                _attn_implementation="flash_attention_2" if self.device == "cuda" else "eager"
            ).to(self.device)
            logger.info("Successfully initialized SmolVLM model")
        except Exception as e:
            logger.error(f"Error initializing SmolVLM model: {str(e)}")
            # Don't raise the error, just log it and continue with limited functionality
        
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if self.model is not None:
                del self.model
            if self.processor is not None:
                del self.processor
            MemoryManager.clear_gpu_memory()
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")
        
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
        
        # If in complete mode (duplicate detected), don't capture
        if self.mode == "complete":
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
        """Extract text from ID card image using SmolVLM model"""
        try:
            if not os.path.exists(image_path):
                ErrorHandler.handle_error(f"Image file not found: {image_path}", "OCR")
                return None, None

            # Check if model and processor are initialized
            if self.model is None or self.processor is None:
                logger.warning("SmolVLM model not initialized. Using fallback OCR method.")
                # Return a default response that will allow the process to continue
                generated_text = "Name: Unknown Student, ID: TEMP" + datetime.now().strftime('%Y%m%d%H%M%S')
                text_file_path = image_path.replace('.jpg', '.txt')
                with open(text_file_path, 'w') as f:
                    f.write(generated_text)
                return generated_text, text_file_path

            # Load and preprocess image
            image = Image.open(image_path)
            
            # Create input messages for ID card text extraction
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What is the name and ID card number mentioned in this ID Card? Please format the response as 'Name: [name], ID: [id]'"}
                    ]
                },
            ]

            # Prepare inputs
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            inputs = inputs.to(self.device)

            # Generate outputs
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=100)
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]

            # Save text to file
            text_file_path = image_path.replace('.jpg', '.txt')
            with open(text_file_path, 'w') as f:
                f.write(generated_text)
            
            logger.info(f"SmolVLM extraction completed. Text saved to: {text_file_path}")
            return generated_text, text_file_path
            
        except Exception as e:
            ErrorHandler.handle_error(e, "SmolVLM processing")
            # Return a default response that will allow the process to continue
            generated_text = "Name: Error Student, ID: ERR" + datetime.now().strftime('%Y%m%d%H%M%S')
            text_file_path = image_path.replace('.jpg', '.txt')
            with open(text_file_path, 'w') as f:
                f.write(generated_text)
            return generated_text, text_file_path

    def extract_student_details(self, text):
        """Extract student details from SmolVLM output"""
        details = {
            'name': None,
            'student_id': None,
            'first_name': None,
            'last_name': None
        }
        
        try:
            # Extract name and ID from the formatted response
            name_match = re.search(r'Name:\s*([^,]+)', text)
            id_match = re.search(r'ID:\s*([^,\s]+)', text)
            
            if name_match:
                full_name = name_match.group(1).strip()
                details['name'] = full_name
                
                # Split name into first and last name
                name_parts = full_name.split()
                if len(name_parts) >= 2:
                    details['first_name'] = name_parts[0]
                    details['last_name'] = ' '.join(name_parts[1:])
                else:
                    details['first_name'] = full_name
                    details['last_name'] = ''
                
                logger.info(f"Found student name: {details['name']}")
            
            if id_match:
                details['student_id'] = id_match.group(1).strip()
                logger.info(f"Found student ID: {details['student_id']}")
            
            return details
            
        except Exception as e:
            logger.error(f"Error extracting student details: {str(e)}")
            return details

    def process_id_card(self, image_path):
        """Process ID card image and extract student details"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image")
            
            # Extract text from ID card using SmolVLM
            extracted_text, text_file_path = self.perform_ocr(image_path)
            if not extracted_text:
                logger.warning("No text extracted from ID card, using default values")
                extracted_text = "Unknown Student"
            
            # Extract student details
            student_details = self.extract_student_details(extracted_text)
            
            # Check if name was extracted
            if not student_details.get('name'):
                logger.warning("No name extracted, using default name")
                student_details['name'] = 'Unknown Student'
                student_details['first_name'] = 'Unknown'
                student_details['last_name'] = 'Student'
            
            if not student_details.get('student_id'):
                logger.warning("No ID generated, creating new ID")
                student_details['student_id'] = f"STU{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Check if student already exists
            try:
                existing_student = Student.objects.get(student_id=student_details['student_id'])
                logger.info(f"Student with ID {student_details['student_id']} already exists")
                # Set flags to stop further processing
                self.captured = True
                self.mode = "complete"  # New mode to indicate processing is complete
                existing_student.is_duplicate = True
                return existing_student
            except Student.DoesNotExist:
                # Student doesn't exist, proceed with creation
                pass
            
            # Set default validity date (1 year from now)
            id_valid_until = timezone.now() + timezone.timedelta(days=365)
            
            try:
                # Create student record
                student = Student.objects.create(
                    name=student_details['name'],
                    first_name=student_details.get('first_name', ''),
                    last_name=student_details.get('last_name', ''),
                    student_id=student_details['student_id'],
                    id_valid_until=id_valid_until,
                    face_verified=False,
                    registered_at=timezone.now()
                )
                
                # Save ID card image path
                if os.path.exists(image_path):
                    student.id_card_image = image_path.replace(settings.MEDIA_ROOT + '/', '')
                    student.save()
                
                logger.info(f"Created new student record with name: {student.name}")
                # Switch to face mode for new registrations
                self.switch_to_face_mode()
                student.is_duplicate = False
                return student
                
            except Exception as inner_e:
                logger.error(f"Error creating student record: {str(inner_e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error processing ID card: {str(e)}")
            raise