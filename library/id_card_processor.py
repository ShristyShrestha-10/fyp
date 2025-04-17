import cv2
import numpy as np
import os
from datetime import datetime
import logging
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from .utils import (
    FileManager,
    MemoryManager,
    ErrorHandler,
    FrameProcessor
)

logger = logging.getLogger(__name__)

class IDCardProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Clear GPU memory
        MemoryManager.clear_gpu_memory()
            
        # Load model with optimized settings
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
        self.model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-256M-Instruct",
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
            device_map="auto" if self.device == "cuda" else None,
        ).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Optimize model for inference
        if self.device == "cuda":
            self.model = torch.compile(self.model)
            
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
        """Perform OCR on the ID card image using SmolVLM with optimized settings"""
        try:
            if not os.path.exists(image_path):
                ErrorHandler.handle_error(f"Image file not found: {image_path}", "OCR")
                return None, None
                
            # Load image directly
            pil_image = load_image(image_path)
            
            # Create input messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Extract name, ID, Course Name, Validity Date of ID Card. "}
                    ]
                },
            ]
            
            # Prepare inputs with optimized settings
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[pil_image], return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Generate outputs with optimized parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,  # Reduced from 500 to speed up inference
                    num_beams=1,  # Use greedy decoding for speed
                    do_sample=False,  # Disable sampling for faster inference
                    early_stopping=True,  # Stop when the model is confident
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
                
            extracted_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
            
            # Save text to file
            text_file_path = image_path.replace('.jpg', '.txt')
            with open(text_file_path, 'w') as f:
                f.write(extracted_text)
            
            # Clear GPU memory
            MemoryManager.clear_gpu_memory()
            
            logger.info(f"OCR completed successfully. Text saved to: {text_file_path}")
            return extracted_text, text_file_path
            
        except Exception as e:
            ErrorHandler.handle_error(e, "OCR processing")
            return None, None