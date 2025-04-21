import cv2
import numpy as np
import os
from datetime import datetime
import logging
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
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
        self.tokenizer = None
        
        # Check for CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize Moondream2 model and tokenizer
        try:
            model_id = "vikhyatk/moondream2"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            logger.info("Successfully initialized Moondream2 model")
        except Exception as e:
            logger.error(f"Error initializing Moondream2 model: {str(e)}")
            # Don't raise the error, just log it and continue with limited functionality
        
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
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
        """Extract text from ID card image using Moondream2 model"""
        try:
            if not os.path.exists(image_path):
                ErrorHandler.handle_error(f"Image file not found: {image_path}", "OCR")
                return None, None

            # Check if model and tokenizer are initialized
            if self.model is None or self.tokenizer is None:
                logger.warning("Moondream2 model not initialized. Using fallback OCR method.")
                # Return a default response that will allow the process to continue
                generated_text = "Name: Unknown Student, ID: TEMP" + datetime.now().strftime('%Y%m%d%H%M%S')
                text_file_path = image_path.replace('.jpg', '.txt')
                with open(text_file_path, 'w') as f:
                    f.write(generated_text)
                return generated_text, text_file_path

            # Load image
            image = Image.open(image_path)
            
            # Process with Moondream2 
            try:
                # Check if the model has the method we expect
                if hasattr(self.model, "answer_question") and callable(getattr(self.model, "answer_question")):
                    # Use answer_question method if available
                    prompt = "This is a student ID card. Extract the student's full name and ID number from this ID card. Format your response as: 'Name: [full name], ID: [ID number]'. If you cannot clearly see either the name or ID, indicate with 'unknown'."
                    generated_text = self.model.answer_question(image, prompt)
                    logger.info("Using model.answer_question method")
                elif hasattr(self.model, "query") and callable(getattr(self.model, "query")):
                    # Use query method if available
                    prompt = "This is a student ID card. Extract the student's full name and ID number from this ID card. Format your response as: 'Name: [full name], ID: [ID number]'. If you cannot clearly see either the name or ID, indicate with 'unknown'."
                    response = self.model.query(
                        image=image, 
                        text=prompt  # Using 'text' parameter as per error message
                    )
                    generated_text = response.get("answer", "")
                    logger.info("Using model.query method with text parameter")
                else:
                    # Direct model call with tokenizer
                    logger.info("Using direct model call with tokenizer")
                    inputs = self.tokenizer(
                        f"<image>\nThis is a student ID card. Extract the student's full name and ID number from this ID card. Format your response as: 'Name: [full name], ID: [ID number]'. If you cannot clearly see either the name or ID, indicate with 'unknown'.",
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Preprocess the image according to the model's requirements
                    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                    # Add pixel_values to the inputs
                    inputs["pixel_values"] = pixels
                    
                    # Generate text
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=100,
                            do_sample=False
                        )
                    
                    # Decode the generated text
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # If no text was generated, use fallback
                if not generated_text:
                    logger.warning("Moondream2 returned empty response, using fallback")
                    generated_text = "Name: Unknown Student, ID: TEMP" + datetime.now().strftime('%Y%m%d%H%M%S')
                
                # Log the raw response for debugging
                logger.info(f"Moondream2 raw response: {generated_text}")
                
                # Format the response if needed - ensure it has Name and ID format
                if not ("Name:" in generated_text and "ID:" in generated_text):
                    # Extract any apparent name and ID using heuristics
                    words = generated_text.split()
                    # Format response to match expected pattern
                    if len(words) >= 4:
                        # Try to detect if there's a name (usually first parts) and ID (usually numeric or alphanumeric)
                        # This is a simple heuristic, might need adjustment
                        name_part = " ".join(words[:2])  # Assume first two words are name
                        id_part = words[-1]  # Assume last word is ID
                        
                        # Format into standard format
                        generated_text = f"Name: {name_part}, ID: {id_part}"
                    else:
                        generated_text = "Name: Unknown Student, ID: TEMP" + datetime.now().strftime('%Y%m%d%H%M%S')
            
            except Exception as model_error:
                logger.error(f"Error processing with Moondream2: {str(model_error)}")
                generated_text = "Name: Unknown Student, ID: TEMP" + datetime.now().strftime('%Y%m%d%H%M%S')

            # Save text to file
            text_file_path = image_path.replace('.jpg', '.txt')
            with open(text_file_path, 'w') as f:
                f.write(generated_text)
            
            logger.info(f"Moondream2 extraction completed. Text saved to: {text_file_path}")
            return generated_text, text_file_path
            
        except Exception as e:
            ErrorHandler.handle_error(e, "Moondream2 processing")
            # Return a default response that will allow the process to continue
            generated_text = "Name: Error Student, ID: ERR" + datetime.now().strftime('%Y%m%d%H%M%S')
            text_file_path = image_path.replace('.jpg', '.txt')
            with open(text_file_path, 'w') as f:
                f.write(generated_text)
            return generated_text, text_file_path

    def extract_student_details(self, text):
        """Extract student details from Moondream2 output"""
        details = {
            'name': None,
            'student_id': None,
            'first_name': None,
            'last_name': None
        }
        
        try:
            # Extract name and ID using regex from the text
            # The prompt asks for output in format: "Name: [name], ID: [id]"
            name_match = re.search(r'Name:\s*([^,]+)', text)
            id_match = re.search(r'ID:\s*([^,\s]+)', text)
            
            # Extract student ID first (most important for our system)
            if id_match:
                student_id = id_match.group(1).strip()
                
                # Skip obviously invalid IDs
                if "TEMP" in student_id or "ERR" in student_id or "UNKNOWN" in student_id:
                    logger.warning(f"Rejected invalid ID in ID card: {student_id}")
                else:
                    details['student_id'] = student_id
                    logger.info(f"Extracted student ID: '{details['student_id']}'")
            else:
                # Try alternative regex patterns for ID
                alt_id_match = re.search(r'ID\s*(?:number|card|#)?\s*[:\-]?\s*([A-Za-z0-9]+)', text, re.IGNORECASE)
                if alt_id_match:
                    student_id = alt_id_match.group(1).strip()
                    details['student_id'] = student_id
                    logger.info(f"Extracted student ID using alternative pattern: '{details['student_id']}'")
                else:
                    logger.warning("Student ID not found in OCR text")
            
            # Then process name (names can be duplicated, but we still need them)
            if name_match:
                full_name = name_match.group(1).strip()
                
                # Skip "Unknown" names - these are fallback values
                if "Unknown" in full_name or full_name == "":
                    logger.warning("Rejected invalid name in ID card: contains 'Unknown' or is empty")
                else:
                    details['name'] = full_name
                    
                    # Split name into first and last name
                    name_parts = full_name.split()
                    if len(name_parts) >= 2:
                        details['first_name'] = name_parts[0]
                        details['last_name'] = ' '.join(name_parts[1:])
                    else:
                        details['first_name'] = full_name
                        details['last_name'] = ''
                    
                    logger.info(f"Extracted student name: '{details['name']}'")
            else:
                # Try alternative regex patterns for name
                alt_name_match = re.search(r'(?:name|student):\s*([^,\.]+)', text, re.IGNORECASE)
                if alt_name_match:
                    full_name = alt_name_match.group(1).strip()
                    details['name'] = full_name
                    
                    # Split name into first and last name
                    name_parts = full_name.split()
                    if len(name_parts) >= 2:
                        details['first_name'] = name_parts[0]
                        details['last_name'] = ' '.join(name_parts[1:])
                    else:
                        details['first_name'] = full_name
                        details['last_name'] = ''
                    
                    logger.info(f"Extracted student name using alternative pattern: '{details['name']}'")
                else:
                    logger.warning("Name not found in OCR text")
            
            return details
            
        except Exception as e:
            logger.error(f"Error extracting student details: {str(e)}")
            return details

    def process_id_card(self, image_path):
        """Process ID card image and extract student details"""
        try:
            # Extract text from ID card using Moondream2
            extracted_text, text_file_path = self.perform_ocr(image_path)
            if not extracted_text:
                logger.warning("No text extracted from ID card, using default values")
                extracted_text = "Name: Unknown Student, ID: UNKNOWN"
            
            # Extract student details from text
            student_details = self.extract_student_details(extracted_text)
            logger.info(f"Extracted student details: {student_details}")
            
            # Check if we have at least a valid student_id (required for registration)
            if not student_details.get('student_id'):
                logger.error("Invalid or missing ID extracted from ID card")
                return None
            
            # Check if student already exists by student_id (duplicates not allowed)
            existing_student = None
            if student_details.get('student_id'):
                try:
                    existing_student = Student.objects.filter(student_id=student_details['student_id']).first()
                    if existing_student:
                        logger.info(f"Found existing student with ID {student_details['student_id']}")
                        # Mark as duplicate for caller to handle
                        existing_student.is_duplicate = True
                        return existing_student
                except Exception as e:
                    logger.error(f"Error checking existing student by ID: {str(e)}")
            
            # For name check, we won't prevent registration but just log for reference
            if student_details.get('name'):
                try:
                    # Try exact name match
                    name_matches = Student.objects.filter(name__iexact=student_details['name']).count()
                    
                    # Try first and last name match if available
                    fname_lname_matches = 0
                    if student_details.get('first_name') and student_details.get('last_name'):
                        fname_lname_matches = Student.objects.filter(
                            first_name__iexact=student_details['first_name'],
                            last_name__iexact=student_details['last_name']
                        ).count()
                        
                    if name_matches > 0 or fname_lname_matches > 0:
                        logger.info(f"Found {name_matches + fname_lname_matches} existing students with similar name {student_details['name']}, but allowing registration with new ID")
                except Exception as e:
                    logger.error(f"Error checking for name duplicates: {str(e)}")
            
            # Create new student - only if we have a valid student_id (name duplicates are allowed)
            try:
                # Get the ID card image path relative to MEDIA_ROOT for database storage
                relative_path = image_path.replace(settings.MEDIA_ROOT + '/', '')
                
                # Set ID card validity for one year from now
                id_validity = timezone.now().date() + timezone.timedelta(days=365)
                
                # Ensure name is set, use a default if not available
                if not student_details.get('name'):
                    student_details['name'] = f"Student {student_details['student_id']}"
                    student_details['first_name'] = student_details['name']
                    student_details['last_name'] = ""
                
                new_student = Student(
                    name=student_details['name'],
                    student_id=student_details['student_id'],
                    first_name=student_details.get('first_name', ''),
                    last_name=student_details.get('last_name', ''),
                    id_card_image=relative_path,
                    registered_at=timezone.now(),
                    id_valid_until=id_validity
                )
                new_student.save()
                logger.info(f"Created new student: {new_student.name}, ID: {new_student.student_id}")
                return new_student
            except Exception as e:
                logger.error(f"Error creating new student: {str(e)}")
                return None
        except Exception as e:
            ErrorHandler.handle_error(e, "Processing ID card")
            return None