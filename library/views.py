import logging
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
from datetime import datetime, timedelta
import face_recognition
import apriltag
import time
from django.db.models import Q
from .models import Book, Student, BorrowedBook, StudentLogin
from django.db.models import Count
from django.utils import timezone
from .camera_factory import CameraFactory
from .utils import ErrorHandler
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from .recommendation_service import BookRecommender
from django.views.decorators.csrf import csrf_exempt
import torch
import json
from django.db.utils import IntegrityError
from django.urls import reverse
from django.http import HttpRequest

# Initialize logger
logger = logging.getLogger(__name__)

def store_detected_book(tag_id, book_info):
    """Store or update a book in the database based on detected AprilTag."""
    try:
        # Ensure tag_id is an integer
        tag_id = int(tag_id)
        
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
        ErrorHandler.handle_error(e, "Storing detected book")
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
        ErrorHandler.handle_error(e, "Loading tag mapping")
    
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

def start_opencv(request):
    try:
        # Initialize camera using factory
        camera = CameraFactory.create_camera('id_card')
        if camera is None or not camera.is_running:
            return JsonResponse({
                'status': 'error',
                'message': 'No working camera found. Please check your camera connection.'
            })
        
        return JsonResponse({
            'status': 'success',
            'message': 'Camera is ready'
        })
        
    except Exception as e:
        ErrorHandler.handle_error(e, "Starting OpenCV")
        return JsonResponse({
            'status': 'error',
            'message': f'An unexpected error occurred: {str(e)}'
        })

def camera_feed(request):
    """View for the camera feed page"""
    try:
        # Get the type of scanner from query parameters
        scanner_type = request.GET.get('type', 'id_card')
        
        # Check if this is a refresh and user is already logged in
        is_student_login = scanner_type == 'student_login'
        student_session_exists = 'student_id' in request.session
        
        # If this is a student login camera and user is already logged in,
        # redirect them to the main page instead
        if is_student_login and student_session_exists:
            logger.info(f"User with session {request.session.get('student_id')} already logged in, redirecting to main page")
            return redirect('main_page')
            
        # Initialize camera using factory
        camera = CameraFactory.create_camera(scanner_type)
        if camera is None or not camera.is_running:
            return render(request, 'library/camera_feed.html', {
                'error': 'Camera initialization failed. Please check your camera connection.',
                'show_retry': True,
                'scanner_type': scanner_type,  # Add scanner_type to context
                'is_registration': request.GET.get('is_registration') == 'true'
            })
        
        # If this is a borrow/return camera and we have a student session, set it
        if scanner_type in ['borrow', 'return'] and hasattr(camera, 'set_student_session'):
            student_id = request.session.get('student_id')
            if student_id:
                camera.set_student_session(student_id)
                logger.info(f"Set student session {student_id} for {scanner_type} camera")
            else:
                logger.warning(f"No student session found for {scanner_type} camera, redirecting to auth page")
                # Stop the camera since we won't be using it
                camera.stop()
                CameraFactory.remove_camera(scanner_type)
                return redirect('auth_page')
            
        # Include session info in template context to maintain state
        student_name = request.session.get('student_name', '')
        student_id = request.session.get('student_id', '')
            
        return render(request, 'library/camera_feed.html', {
            'camera_ready': True,
            'scanner_type': scanner_type,
            'is_registration': request.GET.get('is_registration') == 'true',
            'page_title': request.GET.get('page_title', 'Camera Feed'),
            'student_name': student_name,
            'student_id': student_id
        })
        
    except Exception as e:
        logger.error(f"Error in camera feed: {str(e)}")
        return render(request, 'library/camera_feed.html', {
            'error': f'An error occurred while initializing the camera: {str(e)}',
            'show_retry': True,
            'scanner_type': request.GET.get('type', 'id_card'),  # Add scanner_type to context
            'is_registration': request.GET.get('is_registration') == 'true'
        })

@csrf_exempt
def stop_scanner(request):
    """Stop the camera scanner and clean up resources"""
    if request.method == 'POST':
        try:
            # Get camera type from request body
            import json
            try:
                data = json.loads(request.body)
                camera_type = data.get('camera_type', 'id_card')
            except json.JSONDecodeError:
                camera_type = request.POST.get('camera_type', 'id_card')
            
            logger.info(f"Stopping scanner of type: {camera_type}")
            
            # Get the active camera
            active_camera = CameraFactory.get_active_camera(camera_type)
            if active_camera:
                logger.info(f"Found active camera of type {camera_type}, stopping it...")
                
                # Stop the camera and wait for it to fully stop
                active_camera.stop()
                time.sleep(0.5)  # Give a small delay for cleanup
                
                # Release OpenCV video capture
                if hasattr(active_camera, 'video') and active_camera.video is not None:
                    active_camera.video.release()
                    active_camera.video = None
                    cv2.destroyAllWindows()  # Cleanup any OpenCV windows
                
                # Remove from factory's active cameras
                CameraFactory.remove_camera(camera_type)
                
                # Clear GPU memory if needed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Clear any session data related to camera
                if 'camera_active' in request.session:
                    del request.session['camera_active']
                if camera_type == 'borrow' and 'student_id' in request.session:
                    del request.session['student_id']
                request.session.modified = True
                
                logger.info(f"Successfully stopped camera of type {camera_type}")
                return JsonResponse({
                    'status': 'success',
                    'message': 'Scanner stopped successfully'
                })
            else:
                logger.warning(f"No active camera of type {camera_type} found")
                return JsonResponse({
                    'status': 'success',
                    'message': 'No active scanner to stop'
                })
                
        except Exception as e:
            logger.error(f"Error stopping scanner: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f'Error stopping scanner: {str(e)}'
            })
            
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    }, status=405)

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
        ErrorHandler.handle_error(e, "Generating frames")
    finally:
        # Don't stop the camera here - let the factory manage it
        pass

@gzip.gzip_page
def video_feed(request, camera_type='id_card'):
    """Stream the camera feed"""
    try:
        # Check if there's already an active camera of this type
        active_camera = CameraFactory.get_active_camera(camera_type)
        if active_camera and active_camera.is_running:
            logger.info(f"Using existing active camera of type {camera_type}")
            camera = active_camera
        else:
            # Create a new camera using the factory
            logger.info(f"Creating new camera of type {camera_type}")
            camera = CameraFactory.create_camera(camera_type)
            
        if camera is None or not camera.is_running:
            return JsonResponse({
                'error': 'Camera initialization failed. Please check your camera connection.'
            }, status=500)
            
        return StreamingHttpResponse(
            gen_frames(camera),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )

    except Exception as e:
        ErrorHandler.handle_error(e, "Video feed")
        return JsonResponse({
            'error': f'Camera error: {str(e)}'
        }, status=500)

def capture_frame(request, camera_type='id_card'):
    """Capture a single frame from the camera"""
    try:
        # Create camera using factory
        camera = CameraFactory.create_camera(camera_type)
        if camera is None or not camera.is_running:
            return JsonResponse({
                'error': 'Camera initialization failed'
            }, status=500)
            
        # Get a single frame
        frame = camera.get_frame()
        if frame is None:
            # Check if this is due to a duplicate student
            if hasattr(camera, 'is_duplicate') and camera.is_duplicate:
                return JsonResponse({
                    'status': 'warning',
                    'message': f'Student {camera.student_created.name} (ID: {camera.student_created.student_id}) is already registered.',
                    'stop_camera': True,
                    'student_details': {
                        'name': camera.student_created.name,
                        'student_id': camera.student_created.student_id,
                        'registered_at': camera.student_created.registered_at.strftime('%Y-%m-%d %H:%M:%S')
                    }
                })
            return JsonResponse({
                'error': 'Failed to capture frame'
            }, status=500)
            
        # Process the frame based on camera type
        if camera_type == 'id_card':
            # For ID card scanner, use OCR processing
            full_path, cropped_path = camera.processor.save_images(frame)
            if cropped_path:
                # Perform OCR on the cropped image
                ocr_results, text_file = camera.processor.perform_ocr(cropped_path)
                if ocr_results:
                    # Check if this is a duplicate registration
                    if hasattr(camera, 'is_duplicate') and camera.is_duplicate:
                        return JsonResponse({
                            'status': 'warning',
                            'message': f'Student {camera.student_created.name} (ID: {camera.student_created.student_id}) is already registered.',
                            'stop_camera': True,
                            'student_details': {
                                'name': camera.student_created.name,
                                'student_id': camera.student_created.student_id,
                                'registered_at': camera.student_created.registered_at.strftime('%Y-%m-%d %H:%M:%S')
                            }
                        })
                    
                    # For new registrations, try to save face image
                    if camera.detect_and_save_face(frame):
                        logger.info("Face detection completed successfully")
                        face_image = camera.face_image
                    else:
                        face_image = None
                        
                    return JsonResponse({
                        'status': 'success',
                        'text': ocr_results,
                        'text_file': text_file,
                        'image_path': full_path,
                        'face_image': face_image,
                        'student_details': {
                            'name': camera.student_created.name if camera.student_created else 'Unknown',
                            'student_id': camera.student_created.student_id if camera.student_created else 'Unknown',
                            'registered_at': camera.student_created.registered_at.strftime('%Y-%m-%d %H:%M:%S') if camera.student_created else None
                        }
                    })
                else:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Failed to extract text from ID card'
                    })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Failed to save ID card images'
                })
        else:
            # For borrow and return cameras, use AprilTag detection
            detections = camera.book_detector.detect_books(frame)
            
            if detections:
                # Process detected tags
                detected_books = []
                for detection in detections:
                    tag_id = detection['tag_id']
                    
                    # Check if tag is in our mapping
                    if tag_id in tag_to_book_mapping:
                        book_info = tag_to_book_mapping[tag_id]
                        # Store the book in the database
                        book = store_detected_book(tag_id, book_info)
                        if book:
                            detected_books.append({
                                'tag_id': tag_id,
                                'title': book_info['title'],
                                'author': book_info.get('author', ''),
                                'isbn': book_info.get('isbn', '')
                            })
                
                # Save the frame with detections drawn on it
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir = os.path.join(settings.MEDIA_ROOT, 'book_detections')
                os.makedirs(output_dir, exist_ok=True)
                image_path = os.path.join(output_dir, f'book_detection_{timestamp}.jpg')
                cv2.imwrite(image_path, frame)
                
                return JsonResponse({
                    'status': 'success',
                    'detected_books': detected_books,
                    'image_path': image_path
                })
            else:
                return JsonResponse({
                    'status': 'warning',
                    'message': 'No AprilTags detected in the frame'
                })
                    
        return JsonResponse({
            'status': 'success',
            'message': 'Frame captured successfully'
        })
            
    except Exception as e:
        ErrorHandler.handle_error(e, "Capturing frame")
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

def auth_page(request):
    """Landing page with options for login or registration"""
    # Clear any existing student session
    if 'student_id' in request.session:
        return redirect('main_page')
        
    # Clear specified session variables
    for key in ['student_id', 'student_name', 'is_student']:
        if key in request.session:
            del request.session[key]
    
    return render(request, 'library/auth_page.html')

def register_student(request):
    """Register a new student with ID card and face"""
    # If already logged in, redirect to main page
    if 'student_id' in request.session:
        return redirect('main_page')
        
    return render(request, 'library/camera_feed.html', {
        'camera_ready': False,
        'scanner_type': 'id_card',
        'is_registration': True,
        'page_title': 'Student Registration'
    })

def student_login(request):
    """Student login with ID card and face verification"""
    # If already logged in, redirect to main page
    if 'student_id' in request.session:
        return redirect('main_page')
    
    if request.method == 'POST':
        try:
            # Extract data from the POST request
            data = request.POST
            student_id = data.get('student_id')
            is_verified = data.get('is_verified') == 'true'
            similarity_score = float(data.get('similarity_score', 0))
            
            if not student_id:
                logger.error("Student ID not provided in login attempt")
                return JsonResponse({
                    'status': 'error',
                    'message': 'Student ID not provided'
                })
            
            # Get the student from the database
            try:
                student = Student.objects.get(student_id=student_id)
            except Student.DoesNotExist:
                logger.error(f"Student with ID {student_id} not found during login")
                return JsonResponse({
                    'status': 'error',
                    'message': 'Student not found'
                })
            
            # If the face is verified, set session data and return success
            if is_verified:
                # Create a login record before setting session - as a backup if session fails
                login_record = StudentLogin.objects.create(
                    student=student,
                    login_time=timezone.now(),
                    status="Verified",
                    similarity_score=similarity_score,
                    is_active=True
                )
                
                # Try to set session variables with error handling
                session_success = True
                try:
                    # Set session variables
                    request.session['student_id'] = student.student_id
                    request.session['student_name'] = student.name
                    request.session['is_student'] = True
                    request.session['last_activity'] = timezone.now().timestamp()
                    request.session['login_time'] = timezone.now().timestamp()
                    
                    # Generate a new session key for security (prevent session fixation)
                    request.session.cycle_key()
                    
                    # Set a longer session timeout (1 day by default)
                    request.session.set_expiry(86400)
                except Exception as e:
                    # Log session failure but don't block login - we have the login record as backup
                    session_success = False
                    logger.error(f"Session storage failed for student {student.name}: {str(e)}")
                    ErrorHandler.handle_error(e, "Student login session storage")
                
                logger.info(f"Student {student.name} (ID: {student.student_id}) logged in successfully. Session status: {'success' if session_success else 'failed'}")
                
                # Return success even if session failed (we have the login record)
                return JsonResponse({
                    'status': 'success',
                    'message': 'Login successful' + ('' if session_success else ' (session storage issue - limited functionality)'),
                    'student_name': student.name,
                    'student_id': student.student_id,
                    'redirect_url': '/main-page/'  # Use absolute URL path to ensure proper redirection
                })
            else:
                # Create a failed login record to track verification failures
                StudentLogin.objects.create(
                    student=student,
                    login_time=timezone.now(),
                    status="Not Verified",
                    similarity_score=similarity_score,
                    is_active=False
                )
                
                logger.warning(f"Face verification failed for student {student.name} (ID: {student.student_id}) with similarity score: {similarity_score}")
                return JsonResponse({
                    'status': 'error',
                    'message': f'Face verification failed. Similarity score: {similarity_score:.2f} (required: >= 0.3)',
                    'similarity_score': similarity_score
                })
                
        except Exception as e:
            ErrorHandler.handle_error(e, "Student login")
            return JsonResponse({
                'status': 'error',
                'message': f'An unexpected error occurred: {str(e)}'
            })
    
    # If not a POST request, render the login template
    return render(request, 'library/camera_feed.html', {
        'camera_ready': False,
        'scanner_type': 'student_login',
        'is_registration': False,
        'page_title': 'Student Login'
    })

def student_logout(request):
    """Log out student and clear session data"""
    try:
        # Stop any active cameras associated with this session
        camera_type = request.POST.get('camera_type', 'student_login')
        active_camera = CameraFactory.get_active_camera(camera_type)
        if active_camera:
            active_camera.stop()
            CameraFactory.remove_camera(camera_type)
            logger.info(f"Stopped camera for student logout: {camera_type}")
            
        # Safely attempt to clear GPU memory if needed
        student_id = request.session.get('student_id')
        if student_id:
            # Update login record to mark as inactive
            try:
                # Get the student object
                student = Student.objects.get(student_id=student_id)
                
                # Mark all active logins as inactive
                active_logins = StudentLogin.objects.filter(
                    student=student,
                    is_active=True
                )
                
                for login in active_logins:
                    login.is_active = False
                    login.logout_time = timezone.now()
                    login.save()
                    logger.info(f"Marked login record {login.id} as inactive for student {student_id}")
            except Exception as e:
                logger.error(f"Error updating login records during logout: {str(e)}")
                
        # Clear the session
        request.session.flush()
        logger.info("Student logged out and session cleared")
        
        # Redirect to auth page
        return redirect('auth_page')
    except Exception as e:
        ErrorHandler.handle_error(e, "Student logout")
        return redirect('auth_page')  # Redirect even if there's an error

def get_face_detection_status(request):
    """Get the status of face detection for student login"""
    try:
        # Get the current camera session
        camera_type = request.GET.get('camera_type', 'id_card')
        camera = CameraFactory.get_active_camera(camera_type)
        
        if not camera:
            return JsonResponse({
                'status': 'No active camera session',
                'face_detected': False,
                'face_saved': False,
                'capture_complete': False,
                'id_card_captured': False,
                'error_message': None
            })
        
        # Return the status - using getattr with default values for safety
        response = {
            'status': 'Capturing...',
            'face_detected': bool(getattr(camera, 'face_detected', False)),
            'face_saved': bool(getattr(camera, 'face_saved', False)),
            'capture_complete': bool(getattr(camera, 'capture_complete', False)),
            'error_message': getattr(camera, 'error_message', None),
            'phase': getattr(camera, 'phase', 'detection'),
        }
        
        # Add student details if available - create a simplified structure
        student_details = None
        
        # For ID card registration
        if hasattr(camera, 'student_created') and camera.student_created:
            student_details = {
                'name': camera.student_created.name,
                'student_id': camera.student_created.student_id
            }
            response['student_details'] = student_details
        
        # For student login
        if hasattr(camera, 'matched_student') and camera.matched_student:
            student_details = {
                'name': camera.matched_student.name,
                'student_id': camera.matched_student.student_id
            }
            response['student_details'] = student_details
            
            # Only set is_verified to true if the similarity score meets the threshold
            has_sufficient_score = bool(hasattr(camera, 'best_similarity_score') and 
                                   camera.best_similarity_score >= 0.3)
            response['is_verified'] = has_sufficient_score
            
            # Add similarity score if available (calculated during face matching)
            if hasattr(camera, 'best_similarity_score'):
                response['similarity_score'] = float(round(camera.best_similarity_score, 2))
        
        # Add specific status for borrow/return camera
        if camera_type in ['borrow', 'return'] and hasattr(camera, 'detected_books'):
            response['detected_books'] = int(len(camera.detected_books))
            if hasattr(camera, 'student_session') and camera.student_session:
                response['student_session'] = str(camera.student_session)
                # Look up student details if we have a session but no details yet
                if not student_details and camera.student_session:
                    try:
                        student = Student.objects.filter(student_id=camera.student_session).first()
                        if student:
                            student_details = {
                                'name': student.name, 
                                'student_id': student.student_id
                            }
                            response['student_details'] = student_details
                    except Exception as e:
                        logger.error(f"Error looking up student details: {str(e)}")
            
        # For student login, also check ID card status
        if camera_type == 'student_login':
            response['id_card_captured'] = bool(getattr(camera, 'id_card_captured', False))
            response['ready_for_face_recognition'] = bool(getattr(camera, 'ready_for_face_recognition', False))
        
        return JsonResponse(response)
    except Exception as e:
        ErrorHandler.handle_error(e, "Getting face detection status")
        return JsonResponse({
            'status': 'Error',
            'face_detected': False,
            'face_saved': False,
            'error_message': str(e)
        })

def get_student_details(request):
    """API view to get student details for the modal"""
    if request.method == 'GET':
        try:
            student_id = request.GET.get('student_id')
            if not student_id:
                return JsonResponse({'error': 'Student ID is required'}, status=400)
                
            student = Student.objects.get(student_id=student_id)
            
            # Prepare student data for JSON response
            student_data = {
                'id': student.id,
                'student_id': student.student_id,
                'name': student.name,
                'first_name': student.first_name or '',
                'last_name': student.last_name or '',
                'course': student.course or 'Unknown',
                'registered_at': student.registered_at.strftime('%Y-%m-%d %H:%M'),
                'face_verified': student.face_verified,
                'id_card_image': student.id_card_image.url if student.id_card_image else None,
                'face_image': student.face_image.url if hasattr(student, 'face_image') and student.face_image else None,
            }
            
            # Include borrowed books if any
            borrowed_books = BorrowedBook.objects.filter(student=student, returned_date__isnull=True)
            student_data['borrowed_books'] = []
            
            for borrowed in borrowed_books:
                student_data['borrowed_books'].append({
                    'title': borrowed.book.title,
                    'borrowed_date': borrowed.borrowed_date.strftime('%Y-%m-%d'),
                    'due_date': borrowed.due_date.strftime('%Y-%m-%d'),
                    'is_overdue': borrowed.due_date < timezone.now().date()
                })
            
            return JsonResponse({'student': student_data})
            
        except Student.DoesNotExist:
            return JsonResponse({'error': 'Student not found'}, status=404)
        except Exception as e:
            ErrorHandler.handle_error(e, "Getting student details")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@login_required
@require_http_methods(["POST"])
def delete_book(request, book_id):
    try:
        book = Book.objects.get(id=book_id)
        book.delete()
        return JsonResponse({'success': True})
    except Book.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Book not found'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@login_required
@require_http_methods(["POST"])
def delete_activity(request, activity_id):
    try:
        activity = BorrowedBook.objects.get(id=activity_id)
        activity.delete()
        return JsonResponse({'success': True})
    except BorrowedBook.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Activity not found'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@login_required
@require_http_methods(["POST"])
def delete_student(request, student_id):
    try:
        student = Student.objects.get(id=student_id)
        
        # Delete associated files
        if student.face_image:
            try:
                # Get the full path of the image
                image_path = os.path.join(settings.MEDIA_ROOT, str(student.face_image))
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                logger.error(f"Error deleting face image for student {student.name}: {str(e)}")
        
        if student.id_card_image:
            try:
                # Get the full path of the image
                image_path = os.path.join(settings.MEDIA_ROOT, str(student.id_card_image))
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                logger.error(f"Error deleting ID card image for student {student.name}: {str(e)}")
        
        # Delete student record
        student.delete()
        return JsonResponse({'success': True})
    except Student.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Student not found'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
def borrow_detected_book(request):
    """Endpoint for borrowing a detected book"""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed'}, status=405)
    
    # Custom CSRF validation for exempt view
    csrf_token = request.META.get('HTTP_X_CSRFTOKEN', '')
    if not csrf_token and not request.session.get('is_student'):
        return JsonResponse({'status': 'error', 'message': 'CSRF validation failed'}, status=403)

    try:
        # Get tag ID from request
        try:
            data = json.loads(request.body)
            tag_id = data.get('tag_id')
        except json.JSONDecodeError:
            tag_id = request.POST.get('tag_id')
            
        # Validate tag_id
        if not tag_id:
            return JsonResponse({'status': 'error', 'message': 'No tag ID provided'}, status=400)
            
        # Ensure tag_id is an integer
        try:
            tag_id = int(tag_id)
        except ValueError:
            return JsonResponse({'status': 'error', 'message': 'Invalid tag ID format'}, status=400)
            
        # Get student ID from session
        student_id = request.session.get('student_id')
        if not student_id:
            return JsonResponse({'status': 'error', 'message': 'No student session found'}, status=401)
            
        # Get the student and book objects
        try:
            student = Student.objects.get(student_id=student_id)
            book = Book.objects.get(tag_id=tag_id)
        except Student.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Student not found'}, status=404)
        except Book.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Book not found'}, status=404)
            
        # Check if book is already borrowed by this student
        already_borrowed = BorrowedBook.objects.filter(
            book=book,
            student=student,
            is_returned=False
        ).exists()
        
        if already_borrowed:
            return JsonResponse({
                'status': 'error',
                'message': f'You have already borrowed "{book.title}"'
            })
            
        # Check if book is available (not borrowed by someone else)
        is_available = not BorrowedBook.objects.filter(
            book=book,
            is_returned=False
        ).exists()
        
        if not is_available:
            return JsonResponse({
                'status': 'error',
                'message': f'"{book.title}" is currently borrowed by another student'
            })
            
        # Create new borrowing record
        due_date = timezone.now() + timezone.timedelta(days=14)  # 2 weeks loan period
        
        borrow_record = BorrowedBook.objects.create(
            book=book,
            student=student,
            borrowed_date=timezone.now(),
            due_date=due_date,
            is_returned=False
        )
        
        # Update book status
        book.is_available = False
        book.last_borrowed = timezone.now()
        book.borrow_count = book.borrow_count + 1 if hasattr(book, 'borrow_count') else 1
        book.save()
        
        # Log the transaction
        logger.info(f"Book borrowed: {book.title} (Tag ID: {tag_id}) by {student.name} (ID: {student_id})")
        
        return JsonResponse({
            'status': 'success',
            'message': f'Successfully borrowed "{book.title}"',
            'due_date': due_date.strftime('%Y-%m-%d'),
            'student_name': student.name,
            'book_title': book.title
        })
        
    except Exception as e:
        ErrorHandler.handle_error(e, "Borrowing book")
        return JsonResponse({
            'status': 'error',
            'message': f'Error borrowing book: {str(e)}'
        }, status=500)

def process_detected_book(request):
    """Process a detected book from the camera and perform borrowing if student is logged in"""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed'}, status=405)
        
    try:
        # Get tag ID from request
        tag_id = request.POST.get('tag_id')
        if not tag_id:
            return JsonResponse({'status': 'error', 'message': 'No tag ID provided'}, status=400)
            
        # Ensure tag_id is an integer
        try:
            tag_id = int(tag_id)
        except ValueError:
            return JsonResponse({'status': 'error', 'message': 'Invalid tag ID format'}, status=400)
        
        # First check if student is logged in
        student_id = request.session.get('student_id')
        if student_id:
            # Student is logged in, call borrow_detected_book with tag_id
            # Create a new request with the tag_id
            data = json.dumps({'tag_id': tag_id})
            
            # Create mock request for borrow_detected_book
            borrow_request = HttpRequest()
            borrow_request.method = 'POST'
            borrow_request.META = request.META.copy()
            borrow_request._body = data.encode('utf-8')
            borrow_request.session = request.session  # Share the session
            
            # Call borrow_detected_book and return its response
            response = borrow_detected_book(borrow_request)
            return response
        else:
            # Student not logged in, just acknowledge the book detection
            if tag_id in tag_to_book_mapping:
                book_info = tag_to_book_mapping[tag_id]
                book = store_detected_book(tag_id, book_info)
                
                if book:
                    return JsonResponse({
                        'status': 'success',
                        'message': f'Book detected: {book.title}',
                        'book_title': book.title,
                        'author': book.author,
                        'tag_id': tag_id,
                        'needs_login': True
                    })
                else:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Failed to store book information'
                    }, status=500)
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Unknown tag ID: {tag_id}'
                }, status=404)
                
    except Exception as e:
        ErrorHandler.handle_error(e, "Processing detected book")
        return JsonResponse({
            'status': 'error',
            'message': f'Error processing detected book: {str(e)}'
        }, status=500)

@csrf_exempt
def return_book(request):
    """Handle book return process from either scanner or manually by ID"""
    if request.method != 'POST':
        return JsonResponse({
            'status': 'error',
            'message': 'Only POST requests are allowed'
        }, status=405)
        
    # Custom CSRF validation - check header or session auth
    csrf_token = request.META.get('HTTP_X_CSRFTOKEN')
    is_authenticated = ('student_id' in request.session or request.user.is_authenticated)
    
    if not csrf_token and not is_authenticated:
        logger.warning("CSRF token missing in return_book request")
        return JsonResponse({
            'status': 'error',
            'message': 'Missing CSRF token'
        }, status=403)
        
    try:
        # Extract data from the request
        data = json.loads(request.body) if request.body else {}
        borrowed_book_id = data.get('borrowed_book_id') or request.POST.get('borrowed_book_id')
        tag_id = data.get('tag_id') or request.POST.get('tag_id')
        
        if not borrowed_book_id and not tag_id:
            return JsonResponse({
                'status': 'error',
                'message': 'Missing borrowed_book_id or tag_id parameter'
            }, status=400)
            
        # Look up by ID if provided
        if borrowed_book_id:
            try:
                borrowed_book = BorrowedBook.objects.get(id=borrowed_book_id, is_returned=False)
            except BorrowedBook.DoesNotExist:
                return JsonResponse({
                    'status': 'error',
                    'message': f'No active borrowing found with ID: {borrowed_book_id}'
                }, status=404)
                
        # Look up by tag ID if provided
        elif tag_id:
            try:
                # Get the book by tag_id
                book = Book.objects.get(tag_id=tag_id)
                
                # Check if it's currently borrowed
                borrowed_book = BorrowedBook.objects.filter(
                    book=book,
                    is_returned=False
                ).first()
                
                if not borrowed_book:
                    return JsonResponse({
                        'status': 'error',
                        'message': f'Book with tag ID {tag_id} is not currently borrowed'
                    }, status=404)
            except Book.DoesNotExist:
                return JsonResponse({
                    'status': 'error',
                    'message': f'No book found with tag ID: {tag_id}'
                }, status=404)
                
        # Mark as returned
        borrowed_book.is_returned = True
        borrowed_book.returned_date = timezone.now()
        borrowed_book.save()
        
        # Get the book and update status
        book = borrowed_book.book
        book.is_available = True  # Set book as available again
        book.save()  # Save the book to update its status
        logger.info(f"Book returned successfully - Title: {book.title}, Student: {borrowed_book.student.name}")
        
        return JsonResponse({
            'status': 'success',
            'message': f'Book "{book.title}" returned successfully',
            'data': {
                'book_title': book.title,
                'student_name': borrowed_book.student.name,
                'returned_date': borrowed_book.returned_date.strftime('%Y-%m-%d %H:%M:%S')
            }
        })
            
    except json.JSONDecodeError:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        ErrorHandler.handle_error(e, "Returning book")
        return JsonResponse({
            'status': 'error',
            'message': f'Error returning book: {str(e)}'
        }, status=500)

def main_page(request):
    """Main page after login with options to borrow, return, and recommendations"""
    # Check if user is logged in
    if not request.session.get('is_student') or 'student_id' not in request.session:
        logger.warning("User attempted to access main_page without valid session")
        return redirect('auth_page')
        
    student_id = request.session.get('student_id')
    student_name = request.session.get('student_name', 'Student')
    
    try:
        # Verify student exists in database
        student = Student.objects.filter(student_id=student_id).first()
        if not student:
            logger.warning(f"Student with ID {student_id} not found in database during main_page access")
            # Clear invalid session and redirect to auth page
            request.session.flush()
            return redirect('auth_page')
            
        # Update session activity timestamp
        request.session['last_activity'] = timezone.now().timestamp()
        
        # Get student's borrowed books for display
        borrowed_books = BorrowedBook.objects.filter(
            student=student,
            is_returned=False
        ).select_related('book').order_by('-borrowed_date')
        
        return render(request, 'library/main_page.html', {
            'student_name': student_name,
            'student_id': student_id,
            'student': student,
            'borrowed_books': borrowed_books
        })
        
    except Exception as e:
        logger.error(f"Error in main_page: {str(e)}")
        # In case of error, still try to show the page if we have basic session data
        return render(request, 'library/main_page.html', {
            'student_name': student_name,
            'student_id': student_id,
            'error': str(e)
        })

def recommendations(request):
    try:
        # Initialize the recommender
        recommender = BookRecommender()
        
        # Get all books from the dataset for the dropdown
        all_books = recommender.df['Book-Title'].unique().tolist() if recommender.df is not None else []
        
        # Get the selected book from the query parameters
        selected_book = request.GET.get('book_title')
        recommended_books = []
        
        if selected_book:
            # Get recommendations
            recommendations = recommender.get_recommendations(selected_book, top_n=4)
            
            # Format recommendations for the template
            for book in recommendations:
                recommended_books.append({
                    'title': book['title'],
                    'author': book['author'],
                    'cover_image': book['image_url'],
                    'genre': 'Not specified',  # Dataset doesn't include genre
                    'rating': 4  # Default rating since dataset doesn't include ratings
                })
        
        return render(request, 'library/recommendations.html', {
            'books': all_books,
            'recommended_books': recommended_books,
            'selected_book': selected_book
        })
        
    except Exception as e:
        logger.error(f"Error in recommendations view: {str(e)}")
        messages.error(request, "An error occurred while getting book recommendations.")
        return render(request, 'library/recommendations.html', {
            'books': [],
            'recommended_books': [],
            'error': str(e)
        })

def about(request):
    """View for the about page"""
    return render(request, 'library/about.html')

def admin_login(request):
    """Admin login view"""
    # Check if user is already authenticated
    if request.user.is_authenticated and request.user.is_staff:
        return redirect('admin_dashboard')
        
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
    """Admin dashboard view - focused on recent registrations with face images"""
    # Extra security check to ensure only staff can access this view
    if not request.user.is_staff:
        messages.error(request, "You don't have permission to access the admin dashboard")
        return redirect('home')
    
    try:
        # Count total students and books
        total_students = Student.objects.count()
        total_books = Book.objects.count()
        
        # Get currently borrowed books
        books_borrowed = BorrowedBook.objects.filter(is_returned=False).count()
        
        # Get overdue books (more than 14 days)
        fourteen_days_ago = timezone.now() - timezone.timedelta(days=14)
        overdue_books = BorrowedBook.objects.filter(
            is_returned=False,
            borrowed_date__lt=fourteen_days_ago
        ).count()
        
        # Get all recent registrations (priority to those with face images)
        face_verified_students = Student.objects.filter(
            face_verified=True
        ).order_by('-registered_at')[:10]
        
        # Get recent registrations (regardless of verification)
        recent_registrations = Student.objects.all().order_by('-registered_at')[:15]
        
        # Get currently logged-in students
        current_logins = StudentLogin.objects.filter(
            is_active=True,
            logout_time__isnull=True
        ).order_by('-login_time')[:20]

        # Get recently detected books (last 24 hours)
        recently_detected = Book.objects.filter(
            last_detected__gte=timezone.now() - timezone.timedelta(hours=24)
        ).order_by('-last_detected')[:10]

        # Get recent borrowing activities - show both borrowed and returned books
        recent_activities = BorrowedBook.objects.select_related(
            'book', 'student'
        ).order_by(
            '-borrowed_date' if 'borrowed' in request.GET.get('sort', 'borrowed').lower() else '-returned_date'
        )[:20]  # Increased from 10 to 20 to show more activities
        
        # Log the activities for debugging
        logger.info(f"Found {recent_activities.count()} recent activities for admin dashboard")
        for activity in recent_activities:
            status = "RETURNED" if activity.is_returned else "BORROWED"
            return_info = f", returned on {activity.returned_date.strftime('%Y-%m-%d %H:%M:%S')}" if activity.is_returned and activity.returned_date else ""
            logger.info(
                f"Activity: Book '{activity.book.title}' {status} by {activity.student.name} "
                f"on {activity.borrowed_date.strftime('%Y-%m-%d %H:%M:%S')}{return_info}"
            )

        # Get most borrowed books - updated to include more details
        most_borrowed = BorrowedBook.objects.values(
            'book__id',
            'book__title', 
            'book__author', 
            'book__cover_image'
        ).annotate(
            borrow_count=Count('book__id')
        ).order_by('-borrow_count')[:10]
        
        # Format most borrowed books data
        most_borrowed_formatted = []
        for item in most_borrowed:
            most_borrowed_formatted.append({
                'book': {
                    'id': item['book__id'],
                    'title': item['book__title'],
                    'author': item['book__author'],
                    'cover_image': item['book__cover_image']
                },
                'borrow_count': item['borrow_count']
            })
        
        # Get available books for add borrowing form
        available_books = Book.objects.filter(is_available=True).order_by('title')
        
        # Get all books for the book management section
        books = Book.objects.all().order_by('title')
        
        # Log the number of available books
        logger.info(f"Found {available_books.count()} available books for borrowing form")
        
        context = {
            'total_books': total_books,
            'total_students': total_students,
            'books_borrowed': books_borrowed,
            'overdue_books': overdue_books,
            'verified_students': face_verified_students,
            'recent_registrations': recent_registrations,
            'current_logins': current_logins,
            'now': timezone.now(),
            'today': timezone.now().date(),
            'recently_detected': recently_detected,
            'recent_activities': recent_activities,
            'most_borrowed': most_borrowed_formatted,
            'available_books': available_books,
            'books': books
        }
        
        # Add meta refresh header to auto-refresh the dashboard
        response = render(request, 'library/admin_dashboard.html', context)
        response['Refresh'] = '10'  # Refresh every 10 seconds
        return response
        
    except Exception as e:
        logger.error(f"Error in admin dashboard: {str(e)}")
        messages.error(request, "An error occurred while loading the dashboard")
        return redirect('home')

@login_required(login_url='admin_login')
def add_user(request):
    """Add a new user/student from the admin dashboard"""
    if not request.user.is_staff:
        messages.error(request, "You don't have permission to add users")
        return redirect('admin_dashboard')
    
    if request.method == 'POST':
        try:
            name = request.POST.get('name')
            student_id = request.POST.get('student_id')
            face_verified = 'face_verified' in request.POST
            
            # Check if student already exists
            if Student.objects.filter(student_id=student_id).exists():
                messages.error(request, f"Student with ID {student_id} already exists")
                return redirect('admin_dashboard')
            
            # Create new student
            student = Student(
                name=name,
                student_id=student_id,
                face_verified=face_verified,
                registered_at=timezone.now()
            )
            
            # Handle face image if provided
            if 'face_image' in request.FILES:
                student.face_image = request.FILES['face_image']
            
            student.save()
            messages.success(request, f"Student {name} added successfully")
            
        except Exception as e:
            ErrorHandler.handle_error(e, "Adding user")
            messages.error(request, f"Error adding student: {str(e)}")
            
    return redirect('admin_dashboard')

@login_required(login_url='admin_login')
def add_book(request):
    """Add a new book from the admin dashboard"""
    if not request.user.is_staff:
        messages.error(request, "You don't have permission to add books")
        return redirect('admin_dashboard')
    
    if request.method == 'POST':
        try:
            title = request.POST.get('title')
            author = request.POST.get('author')
            tag_id = request.POST.get('tag_id')
            
            # Create new book
            book = Book(
                title=title,
                author=author,
                is_available=True,
                last_detected=timezone.now()
            )
            
            # Add tag ID if provided
            if tag_id:
                try:
                    book.tag_id = int(tag_id)
                except ValueError:
                    messages.warning(request, "Invalid tag ID format, saving book without tag ID")
            
            # Handle cover image if provided
            if 'cover_image' in request.FILES:
                book.cover_image = request.FILES['cover_image']
            
            book.save()
            messages.success(request, f"Book '{title}' added successfully")
            
        except Exception as e:
            ErrorHandler.handle_error(e, "Adding book")
            messages.error(request, f"Error adding book: {str(e)}")
            
    return redirect('admin_dashboard')

@login_required(login_url='admin_login')
def add_borrowing(request):
    """Add a new borrowing record from the admin dashboard"""
    if not request.user.is_staff:
        messages.error(request, "You don't have permission to add borrowing records")
        return redirect('admin_dashboard')
    
    if request.method == 'POST':
        try:
            book_id = request.POST.get('book_id')
            student_id = request.POST.get('student_id')
            borrowed_date_str = request.POST.get('borrowed_date')
            
            # Get book and student objects
            book = Book.objects.get(id=book_id)
            student = Student.objects.get(id=student_id)
            
            # Parse borrowed date or use today's date
            try:
                borrowed_date = timezone.datetime.strptime(borrowed_date_str, '%Y-%m-%d').date()
            except (ValueError, TypeError):
                borrowed_date = timezone.now().date()
            
            # Set due date to 14 days after borrow date
            due_date = borrowed_date + timezone.timedelta(days=14)
            
            # Create borrowing record
            borrow_record = BorrowedBook.objects.create(
                book=book,
                student=student,
                borrowed_date=borrowed_date,
                due_date=due_date,
                is_returned=False
            )
            
            # Update book status
            book.is_available = False
            book.last_borrowed = timezone.now()
            book.save()
            
            messages.success(request, f"Borrowing record created: {book.title} borrowed by {student.name}")
            
        except Book.DoesNotExist:
            messages.error(request, "Selected book not found")
        except Student.DoesNotExist:
            messages.error(request, "Selected student not found")
        except Exception as e:
            ErrorHandler.handle_error(e, "Adding borrowing record")
            messages.error(request, f"Error adding borrowing record: {str(e)}")
            
    return redirect('admin_dashboard')

@login_required(login_url='admin_login')
def edit_student(request, student_id):
    """Edit a student record from the admin dashboard"""
    if not request.user.is_staff:
        messages.error(request, "You don't have permission to edit student records")
        return redirect('admin_dashboard')
    
    try:
        student = Student.objects.get(id=student_id)
        
        if request.method == 'POST':
            # Update student data
            student.name = request.POST.get('name', student.name)
            student.student_id = request.POST.get('student_id', student.student_id)
            
            # Optional fields
            if 'first_name' in request.POST:
                student.first_name = request.POST.get('first_name')
            if 'last_name' in request.POST:
                student.last_name = request.POST.get('last_name')
            if 'course' in request.POST:
                student.course = request.POST.get('course')
                
            # Handle face verification checkbox
            student.face_verified = 'face_verified' in request.POST
            
            # Process face image if provided
            if 'face_image' in request.FILES:
                # Delete old image if it exists
                if student.face_image:
                    try:
                        old_image_path = os.path.join(settings.MEDIA_ROOT, str(student.face_image))
                        if os.path.exists(old_image_path):
                            os.remove(old_image_path)
                    except Exception as e:
                        logger.error(f"Error deleting old face image: {str(e)}")
                
                # Save new image
                student.face_image = request.FILES['face_image']
            
            # Save changes
            student.save()
            messages.success(request, f"Student {student.name} updated successfully")
            return redirect('admin_dashboard')
            
        # For GET requests, just return JSON data for the modal
        student_data = {
            'id': student.id,
            'name': student.name,
            'student_id': student.student_id,
            'first_name': student.first_name or '',
            'last_name': student.last_name or '',
            'course': student.course or '',
            'face_verified': student.face_verified,
            'has_face_image': bool(student.face_image),
        }
        
        return JsonResponse({'student': student_data})
        
    except Student.DoesNotExist:
        messages.error(request, "Student not found")
        return redirect('admin_dashboard')
    except Exception as e:
        ErrorHandler.handle_error(e, "Editing student")
        messages.error(request, f"Error editing student: {str(e)}")
        return redirect('admin_dashboard')

@login_required(login_url='admin_login')
def edit_book(request, book_id):
    """Edit a book record from the admin dashboard"""
    if not request.user.is_staff:
        messages.error(request, "You don't have permission to edit book records")
        return redirect('admin_dashboard')
    
    try:
        book = Book.objects.get(id=book_id)
        
        if request.method == 'POST':
            # Update book data
            book.title = request.POST.get('title', book.title)
            book.author = request.POST.get('author', book.author)
            
            # Optional fields
            if 'tag_id' in request.POST and request.POST.get('tag_id'):
                try:
                    book.tag_id = int(request.POST.get('tag_id'))
                except ValueError:
                    messages.warning(request, "Invalid tag ID format, not updating tag ID")
                    
            if 'isbn' in request.POST:
                book.isbn = request.POST.get('isbn')
            if 'genre' in request.POST:
                book.genre = request.POST.get('genre')
            if 'description' in request.POST:
                book.description = request.POST.get('description')
                
            # Process cover image if provided
            if 'cover_image' in request.FILES:
                # Delete old image if it exists
                if book.cover_image:
                    try:
                        old_image_path = os.path.join(settings.MEDIA_ROOT, str(book.cover_image))
                        if os.path.exists(old_image_path):
                            os.remove(old_image_path)
                    except Exception as e:
                        logger.error(f"Error deleting old cover image: {str(e)}")
                
                # Save new image
                book.cover_image = request.FILES['cover_image']
            
            # Save changes
            book.save()
            messages.success(request, f"Book '{book.title}' updated successfully")
            return redirect('admin_dashboard')
            
        # For GET requests, just return JSON data for the modal
        book_data = {
            'id': book.id,
            'title': book.title,
            'author': book.author,
            'tag_id': book.tag_id,
            'isbn': book.isbn or '',
            'genre': book.genre or '',
            'description': book.description or '',
            'has_cover_image': bool(book.cover_image),
        }
        
        return JsonResponse({'book': book_data})
        
    except Book.DoesNotExist:
        messages.error(request, "Book not found")
        return redirect('admin_dashboard')
    except Exception as e:
        ErrorHandler.handle_error(e, "Editing book")
        messages.error(request, f"Error editing book: {str(e)}")
        return redirect('admin_dashboard')