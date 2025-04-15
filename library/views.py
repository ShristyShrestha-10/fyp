from django.shortcuts import render, redirect
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators import gzip
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
import os
import sys
import subprocess
from django.conf import settings
import cv2
import numpy as np
import logging
import pickle
from datetime import datetime
import face_recognition
import apriltag
import time
from django.db.models import Q
from .models import Book, Student, BorrowedBook
from django.db.models import Count
from django.utils import timezone
from .cameras import IDCardScanner, BorrowBookScanner, ReturnBookScanner
from .id_card_processor import IDCardProcessor

# Set up logging
logger = logging.getLogger(__name__)

# Dictionary to track AprilTag detection timers

def store_detected_book(tag_id, book_info):
    """Store or update a book in the database based on detected AprilTag."""
    try:
        # Extract clean ISBN if it has 'ISBN:' prefix
        isbn = book_info.get('isbn', '')
        if isbn and isbn.startswith('ISBN:'):
            isbn = isbn[5:].strip()
        
        # Check if book with same tag_id already exists
        existing_book = Book.objects.filter(tag_id=tag_id).first()
        
        if existing_book:
            # Update existing book with latest info
            existing_book.title = book_info['title']
            existing_book.author = book_info.get('author', '')
            existing_book.genre = book_info.get('genre', '')
            existing_book.isbn = isbn
            existing_book.last_detected = timezone.now()
            existing_book.save()
            logger.info(f"Updated existing book in database: {book_info['title']} (Tag ID: {tag_id}, ISBN: {isbn})")
            return existing_book
        else:
            # Create new book entry
            new_book = Book(
                title=book_info['title'],
                author=book_info.get('author', ''),
                genre=book_info.get('genre', ''),
                isbn=isbn,
                tag_id=tag_id,
                last_detected=timezone.now(),
                description=f"Book automatically added from AprilTag detection. Tag ID: {tag_id}"
            )
            new_book.save()
            logger.info(f"Added new book to database: {book_info['title']} (Tag ID: {tag_id}, ISBN: {isbn})")
            return new_book
    except Exception as e:
        logger.error(f"Error storing book in database: {str(e)}")
        return None

def load_tag_mapping():
    """Load the tag-to-book mapping from database or file."""
    tag_to_book = {}
    try:
        # Load from mapping file directly - the database doesn't have tag_id field
        mapping_file = os.path.join(settings.BASE_DIR, "book_mapping.txt")
        if os.path.exists(mapping_file):
            with open(mapping_file, "r") as f:
                for line in f:
                    try:
                        parts = line.strip().split(',')
                        if len(parts) < 2:
                            continue
                            
                        tag_id = int(parts[0])
                        
                        # Initialize with defaults
                        book_info = {
                            'title': parts[1].strip() if len(parts) > 1 else f'Book ID: {tag_id}',
                            'author': parts[2].strip() if len(parts) > 2 else '',
                            'genre': parts[3].strip() if len(parts) > 3 else '',
                            'isbn': ''
                        }
                        
                        # Extract ISBN from the last part if it exists
                        if len(parts) > 4:
                            isbn_part = parts[4].strip()
                            if isbn_part.startswith('ISBN:'):
                                book_info['isbn'] = isbn_part[5:].strip()
                            else:
                                book_info['isbn'] = isbn_part
                        
                        tag_to_book[tag_id] = book_info
                        logger.info(f"Loaded book mapping for tag {tag_id}: {book_info}")
                        
                    except ValueError as e:
                        logger.warning(f"Invalid line in mapping file: {line}, Error: {e}")
        else:
            logger.warning(f"Mapping file not found at: {mapping_file}")
    except Exception as e:
        logger.error(f"Error loading tag mapping: {str(e)}")
    
    return tag_to_book

# Load tag mapping when module is loaded
tag_to_book_mapping = load_tag_mapping()

def home(request):
    # Get statistics for the dashboard
    total_books = Book.objects.count()
    total_members = Student.objects.count()
    books_borrowed = BorrowedBook.objects.filter(is_returned=False).count()
    
    return render(request, 'library/home.html', {
        'total_books': total_books,
        'total_members': total_members,
        'books_borrowed': books_borrowed
    })

# Create your views here.
def start_opencv(request):
    try:
        # Initialize camera with multiple attempts
        camera_found = False
        cap = None
        
        # Try different camera indices
        for i in range(3):  # Try indices 0 to 3
            try:
                if cap is not None:
                    cap.release()
                # droidcam_url = f"http://192.168.18.6:4747/video"
                cap = cv2.VideoCapture(i)
                
                if cap.isOpened():
                    # Test if we can actually read from the camera
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        camera_found = True
                        logger.info(f"Camera found at index {i}")
                        break
                    else:
                        cap.release()
                        cap = None
            except Exception as e:
                logger.warning(f"Failed to open camera at index {i}: {str(e)}")
                continue

        # Clean up
        if cap is not None:
            cap.release()

        if not camera_found:
            logger.error("No working camera found")
            return JsonResponse({
                'status': 'error',
                'message': 'No working camera found. Please check your camera connection.'
            })
        
        return JsonResponse({
            'status': 'success',
            'message': 'Camera is ready'
        })
        
    except Exception as e:
        logger.error(f"Error in start_opencv: {str(e)}")
        if cap is not None:
            cap.release()
        return JsonResponse({
            'status': 'error',
            'message': f'An unexpected error occurred: {str(e)}'
        })

def camera_feed(request):
    """View for the camera feed page"""
    try:
        # Get the type of scanner from query parameters
        scanner_type = request.GET.get('type', 'id_card')
        
        # Initialize camera based on scanner type
        if scanner_type == 'id_card':
            camera = IDCardScanner()
        elif scanner_type == 'borrow':
            camera = BorrowBookScanner()
        elif scanner_type == 'return':
            camera = ReturnBookScanner()
        else:
            return JsonResponse({'error': 'Invalid scanner type'}, status=400)
            
        # Check if camera was initialized successfully
        if not camera.is_running or camera.video is None:
            return render(request, 'library/camera_feed.html', {
                'error': 'Camera initialization failed. Please check your camera connection.',
                'show_retry': True
            })
            
        return render(request, 'library/camera_feed.html', {
            'camera_ready': True,
            'scanner_type': scanner_type
        })
        
    except Exception as e:
        logger.error(f"Error in camera_feed view: {str(e)}")
        return render(request, 'library/camera_feed.html', {
            'error': f'An error occurred while initializing the camera: {str(e)}',
            'show_retry': True
        })

def stop_scanner(request):
    if request.method == 'POST':
        try:
            return JsonResponse({
                'status': 'success',
                'message': 'Scanner stopped successfully'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    }, status=405)

class BaseCamera:
    def __init__(self):
        try:
            self.video = None
            self.is_running = False
            
            # Try camera indices in priority order (2, 1, 0)
            camera_indices = [2,0,1]
            
            for i in camera_indices:
                try:
                    logger.info(f"Attempting to open camera at index {i}")
                    self.video = cv2.VideoCapture(i)
                    
                    if self.video.isOpened():
                        # Test if we can read from the camera
                        ret, frame = self.video.read()
                        if ret and frame is not None:
                            logger.info(f"Successfully opened camera at index {i}")
                            self.is_running = True
                            return
                    else:
                        logger.warning(f"Could not open camera {i}")
                        
                except Exception as e:
                    logger.warning(f"Failed to open camera at index {i}: {str(e)}")
                    if self.video is not None:
                        self.video.release()
                        self.video = None
            
            if self.video is None or not self.video.isOpened():
                raise Exception("Could not open any camera")
            
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
                time.sleep(0.1)
                
            if not success or frame is None:
                logger.error("Failed to read frame from camera after multiple attempts")
                return None
            
            # Use higher quality JPEG encoding
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            ret, jpeg = cv2.imencode('.jpg', frame, encode_params)
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
        self.face_saved = False  # New flag to track if face has been saved
        
    def detect_and_save_face(self, frame):
        """Detect face in the frame and save it if found"""
        try:
            # Get box coordinates from processor after drawing
            self.box_coords = self.processor.box_coords
            if self.box_coords is None:
                logger.error("Box coordinates not available")
                return False
                
            x, y, box_width, box_height = self.box_coords
            
            # Extract the region of interest for face detection
            face_roi = frame[y:y + box_height, x:x + box_width]
            
            # Convert ROI to RGB (face_recognition uses RGB)
            rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            # Find all face locations in the ROI
            face_locations = face_recognition.face_locations(rgb_roi)
            
            if face_locations:
                # Get the first face found
                top, right, bottom, left = face_locations[0]
                
                # Only save face if it hasn't been saved yet and we're in capture phase
                if not self.face_saved and self.processor.should_capture():
                    # Extract face image from ROI
                    face_image = face_roi[top:bottom, left:right]
                    
                    # Create output directory if it doesn't exist
                    output_dir = os.path.join('media', 'faces')
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Generate timestamp for filename
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    face_path = os.path.join(output_dir, f'face_{timestamp}.jpg')
                    
                    # Save face image
                    cv2.imwrite(face_path, face_image)
                    
                    self.face_detected = True
                    self.face_image = face_path
                    self.face_saved = True
                    logger.info(f"Face detected and saved to: {face_path}")
                
                # Draw rectangle around face (in the ROI coordinates)
                cv2.rectangle(frame, 
                            (x + left, y + top),
                            (x + right, y + bottom),
                            (0, 255, 0), 2)
                
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error detecting face: {str(e)}")
            return False
        
    def get_frame(self):
        """Get a frame from the camera with ID card scanning overlay"""
        success, frame = self.video.read()
        if not success:
            logger.error("Failed to read frame from camera")
            return None
            
        try:
          
            
            if not self.capture_complete:
                # Start capture timer if not already started
                self.processor.start_capture_timer()
                
                # Check if it's time to capture
                if self.processor.should_capture():
                    if not self.id_card_captured:
                        # First phase: Capture ID card
                        logger.info("Attempting to capture and process ID card")
                        
                        # Save the images
                        full_path, cropped_path = self.processor.save_images(frame)
                        if cropped_path:
                            # Perform OCR on the cropped image
                            self.ocr_results, text_file = self.processor.perform_ocr(cropped_path)
                            if self.ocr_results:
                                logger.info("OCR completed successfully")
                                self.id_card_captured = True
                                # Switch to face detection mode
                                self.processor.switch_to_face_mode()
                                # Reset face saved flag for new capture phase
                                self.face_saved = False
                            else:
                                self.error_message = "Failed to extract text from ID card"
                                logger.error(self.error_message)
                                # Reset timer to try again
                                self.processor.start_time = None
                        else:
                            self.error_message = "Failed to save ID card images"
                            logger.error(self.error_message)
                            # Reset timer to try again
                            self.processor.start_time = None
                    else:
                        # Second phase: Capture face
                        if self.detect_and_save_face(frame):
                            logger.info("Face detection completed successfully")
                            self.capture_complete = True
                        else:
                            self.error_message = "No face detected. Please position your face in the green box."
                            logger.error(self.error_message)
                            # Reset timer to try again
                            self.processor.start_time = None
                
                # Try to detect face continuously in face detection mode
                elif self.id_card_captured:
                    # Only detect face without saving
                    self.detect_and_save_face(frame)
            
            # Display error message on frame if any
            if self.error_message:
                cv2.putText(frame, self.error_message, (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Encode the frame
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return None

class BorrowCamera(BaseCamera):
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
            
            # Check for 'Q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
                return None
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes() if ret else None
            
        except Exception as e:
            logger.error(f"Error in borrow camera frame: {str(e)}")
            return None

class ReturnCamera(BaseCamera):
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
            
            # Check for 'Q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
                return None
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes() if ret else None
            
        except Exception as e:
            logger.error(f"Error in return camera frame: {str(e)}")
            return None

def gen_frames(camera):
    try:
        while camera.is_running:
            frame = camera.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                break
    except Exception as e:
        logger.error(f"Error in gen_frames: {str(e)}")
    finally:
        camera.stop()

@gzip.gzip_page
def video_feed(request, camera_type='id_card'):
    """Stream the camera feed"""
    try:
        # Create the appropriate camera scanner based on type
        if camera_type == 'id_card':
            camera = IDCardScanner()
        elif camera_type == 'borrow':
            camera = BorrowBookScanner()
        elif camera_type == 'return':
            camera = ReturnBookScanner()
        else:
            return JsonResponse({'error': 'Invalid camera type'}, status=400)
            
        # Check if camera was initialized successfully
        if not camera.is_running or camera.video is None:
            logger.error("Camera failed to initialize")
            return JsonResponse({
                'error': 'Camera initialization failed. Please check your camera connection.'
            }, status=500)
            
        return StreamingHttpResponse(
            gen_frames(camera),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )

    except Exception as e:
        logger.error(f"Error in video_feed: {str(e)}")
        return JsonResponse({
            'error': f'Camera error: {str(e)}'
        }, status=500)

def capture_frame(request, camera_type='id_card'):
    """Capture a single frame from the camera"""
    try:
        # Create the appropriate camera scanner based on type
        if camera_type == 'id_card':
            camera = IDCardScanner()
        elif camera_type == 'borrow':
            camera = BorrowBookScanner()
        elif camera_type == 'return':
            camera = ReturnBookScanner()
        else:
            return JsonResponse({'error': 'Invalid camera type'}, status=400)
            
        # Check if camera was initialized successfully
        if not camera.is_running or camera.video is None:
            return JsonResponse({
                'error': 'Camera initialization failed'
            }, status=500)
            
        # Get a single frame
        frame = camera.get_frame()
        if frame is None:
            return JsonResponse({
                'error': 'Failed to capture frame'
            }, status=500)
            
        # For ID card scanner, process the captured frame
        if camera_type == 'id_card':
            # Save the captured frame
            image_path, cropped_path = camera.processor.save_images(frame)
            if cropped_path:
                # Perform OCR on the cropped image
                ocr_results, text_file = camera.processor.perform_ocr(cropped_path)
                if ocr_results:
                    return JsonResponse({
                        'status': 'success',
                        'text': ocr_results,
                        'text_file': text_file,
                        'image_path': image_path
                    })
                    
        return JsonResponse({
            'status': 'success',
            'message': 'Frame captured successfully'
        })
            
    except Exception as e:
        logger.error(f"Error capturing frame: {str(e)}")
        return JsonResponse({
            'error': f'Capture error: {str(e)}'
        }, status=500)
    finally:
        if 'camera' in locals():
            camera.stop()

def process_id_card(request):
    # Handle ID card processing logic
    if request.method == 'POST':
        # Process uploaded ID card
        uploaded_file = request.FILES.get('id_card')
        if uploaded_file:
            # Save file to media directory
            file_path = os.path.join(settings.MEDIA_ROOT, 'id_cards', uploaded_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            # Additional processing logic
    return render(request, 'library/id_card_processing.html')

def main_page(request):
    """View for the main page with borrow and return buttons"""
    return render(request, 'library/main_page.html')

def recommendations(request):
    """View for the book recommendations page"""
    # Get all books and sort by rating
    books = Book.objects.all().order_by('-rating')[:6]  # Get top 6 books
    return render(request, 'library/recommendations.html', {'recommended_books': books})

def about(request):
    """View for the about page"""
    return render(request, 'library/about.html')

def admin_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None and user.is_staff:
            login(request, user)
            return redirect('admin_dashboard')
        else:
            return render(request, 'library/admin_login.html', {
                'error': 'Invalid username or password'
            })
    
    return render(request, 'library/admin_login.html')

def admin_logout(request):
    logout(request)
    return redirect('home')

@login_required(login_url='admin_login')
def admin_dashboard(request):
    if not request.user.is_staff:
        return redirect('admin_login')
        
    # Get total number of students
    total_students = Student.objects.count()
    
    # Get total number of books
    total_books = Book.objects.count()
    
    # Get number of borrowed books
    books_borrowed = BorrowedBook.objects.filter(is_returned=False).count()
    
    # Get number of overdue books
    overdue_books = BorrowedBook.objects.filter(
        is_returned=False,
        due_date__lt=timezone.now()
    ).count()
    
    # Get recent activities
    recent_activities = BorrowedBook.objects.all().order_by('-borrowed_date')[:5]
    
    # Get most borrowed books
    most_borrowed = Book.objects.annotate(
        borrow_count=Count('borrowedbook')
    ).order_by('-borrow_count')[:5]
    
    # Get recently detected books (those with last_detected not null)
    recently_detected = Book.objects.filter(
        last_detected__isnull=False
    ).order_by('-last_detected')[:5]
    
    context = {
        'total_students': total_students,
        'total_books': total_books,
        'books_borrowed': books_borrowed,
        'overdue_books': overdue_books,
        'recent_activities': recent_activities,
        'most_borrowed': most_borrowed,
        'recently_detected': recently_detected,
    }
    
    return render(request, 'library/admin_dashboard.html', context)