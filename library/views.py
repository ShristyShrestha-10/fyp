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

# Initialize logger
logger = logging.getLogger(__name__)

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
        
        # Initialize camera using factory
        camera = CameraFactory.create_camera(scanner_type)
        if camera is None or not camera.is_running:
            return render(request, 'library/camera_feed.html', {
                'error': 'Camera initialization failed. Please check your camera connection.',
                'show_retry': True,
                'scanner_type': scanner_type  # Add scanner_type to context
            })
        
        # If this is a borrow camera and we have a student session, set it
        if scanner_type == 'borrow' and hasattr(camera, 'set_student_session'):
            student_id = request.session.get('student_id')
            if student_id:
                camera.set_student_session(student_id)
                logger.info(f"Set student session {student_id} for borrow camera")
            else:
                logger.warning("No student session found for borrow camera")
            
        return render(request, 'library/camera_feed.html', {
            'camera_ready': True,
            'scanner_type': scanner_type
        })
        
    except Exception as e:
        logger.error(f"Error in camera feed: {str(e)}")
        return render(request, 'library/camera_feed.html', {
            'error': f'An error occurred while initializing the camera: {str(e)}',
            'show_retry': True,
            'scanner_type': request.GET.get('type', 'id_card')  # Add scanner_type to context
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
                    'message': 'Scanner stopped successfully',
                    'refresh_session': True,
                    'stay_on_page': True
                })
            else:
                logger.warning(f"No active camera of type {camera_type} found")
                return JsonResponse({
                    'status': 'success',
                    'message': 'No active scanner to stop',
                    'refresh_session': True,
                    'stay_on_page': True
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
                # Stop and reset camera for next registration
                camera.stop()
                CameraFactory.remove_camera(camera_type)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return JsonResponse({
                    'status': 'warning',
                    'message': f'Student {camera.student_created.name} (ID: {camera.student_created.student_id}) is already registered.',
                    'stop_camera': True,
                    'refresh_session': True,
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
                        # Stop and reset camera for next registration
                        camera.stop()
                        CameraFactory.remove_camera(camera_type)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return JsonResponse({
                            'status': 'warning',
                            'message': f'Student {camera.student_created.name} (ID: {camera.student_created.student_id}) is already registered.',
                            'stop_camera': True,
                            'refresh_session': True,
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
                        
                    # Stop and reset camera for next registration after successful processing
                    camera.stop()
                    CameraFactory.remove_camera(camera_type)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    return JsonResponse({
                        'status': 'success',
                        'text': ocr_results,
                        'text_file': text_file,
                        'image_path': full_path,
                        'face_image': face_image,
                        'refresh_session': True,
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

def main_page(request):
    """View for the main page with borrow and return buttons"""
    return render(request, 'library/main_page.html')

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

@login_required
def admin_dashboard(request):
    """Admin dashboard view - focused on recent registrations with face images"""
    try:
        # Get basic statistics
        total_books = Book.objects.count()
        total_students = Student.objects.count()
        books_borrowed = BorrowedBook.objects.filter(is_returned=False).count()
        overdue_books = BorrowedBook.objects.filter(
            is_returned=False,
            due_date__lt=timezone.now()
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

        # Get recent borrowing activities - updated to show latest first and include all necessary fields
        recent_activities = BorrowedBook.objects.select_related(
            'book', 'student'
        ).order_by(
            '-borrowed_date'
        )[:10]
        
        # Log the activities for debugging
        logger.info(f"Found {recent_activities.count()} recent activities")
        for activity in recent_activities:
            logger.info(
                f"Activity: Book '{activity.book.title}' borrowed by {activity.student.name} "
                f"on {activity.borrowed_date.strftime('%Y-%m-%d %H:%M:%S')}"
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
        
        context = {
            'total_books': total_books,
            'total_students': total_students,
            'books_borrowed': books_borrowed,
            'overdue_books': overdue_books,
            'verified_students': face_verified_students,
            'recent_registrations': recent_registrations,
            'current_logins': current_logins,
            'now': timezone.now(),
            'recently_detected': recently_detected,
            'recent_activities': recent_activities,
            'most_borrowed': most_borrowed_formatted
        }
        
        # Add meta refresh header to auto-refresh the dashboard
        response = render(request, 'library/admin_dashboard.html', context)
        response['Refresh'] = '10'  # Refresh every 10 seconds
        return response
        
    except Exception as e:
        logger.error(f"Error in admin dashboard: {str(e)}")
        messages.error(request, "An error occurred while loading the dashboard")
        return redirect('home')

def student_login(request):
    """View for student login with face verification"""
    if request.method == 'POST':
        # Get student ID from the form data
        student_id = request.POST.get('student_id')
        
        if not student_id:
            messages.error(request, "Student ID is required")
            return render(request, 'library/student_login.html', {
                'camera_type': 'student_login',
                'error': 'Student ID is required'
            })
            
        try:
            # Find the student by ID
            student = Student.objects.get(student_id=student_id)
            
            # Check if the student's face has been verified
            if not student.face_verified:
                messages.error(request, "Your face has not been verified yet. Please visit the library desk.")
                return render(request, 'library/student_login.html', {
                    'camera_type': 'student_login',
                    'error': 'Face not verified'
                })
                
            # Create a session for the student
            request.session['student_id'] = student.student_id
            request.session['student_name'] = student.name
            request.session['is_student'] = True
            
            # Record login activity
            try:
                # Create a default login record (without similarity score, which is added during face verification)
                StudentLogin.objects.create(
                    student=student,
                    login_time=timezone.now(),
                    status="Manual Login",  # This indicates a manual login without face verification
                    similarity_score=0.0,   # Default to 0 for manual logins
                    is_active=True
                )
                logger.info(f"Student manually logged in: {student.name} (ID: {student.student_id})")
            except Exception as activity_error:
                logger.error(f"Error recording login activity: {str(activity_error)}")
            
            messages.success(request, f"Welcome, {student.name}!")
            return redirect('home')  # Redirect to home page after successful login
            
        except Student.DoesNotExist:
            messages.error(request, "Student not found. Please check your ID.")
            return render(request, 'library/student_login.html', {
                'camera_type': 'student_login',
                'error': 'Student not found'
            })
        except Exception as e:
            ErrorHandler.handle_error(e, "Student login")
            messages.error(request, "An error occurred during login. Please try again.")
            return render(request, 'library/student_login.html', {
                'camera_type': 'student_login',
                'error': 'Login error'
            })
    
    # GET request - show login page with camera feed        
    return render(request, 'library/student_login.html', {
        'camera_type': 'student_login'
    })

def student_logout(request):
    """Handle student logout and update the StudentLogin record"""
    try:
        if 'student_id' in request.session:
            student_id = request.session.get('student_id')
            # Get the student
            try:
                student = Student.objects.get(student_id=student_id)
                
                # Update any active login records
                active_logins = StudentLogin.objects.filter(
                    student=student,
                    is_active=True,
                    logout_time__isnull=True
                )
                
                for login in active_logins:
                    login.logout_time = timezone.now()
                    login.is_active = False
                    login.save()
                    
                logger.info(f"Student logged out: {student.name} (ID: {student.student_id})")
            except Exception as e:
                logger.error(f"Error updating login record on logout: {str(e)}")
            
            # Clear session
            request.session.pop('student_id', None)
            request.session.pop('student_name', None)
            request.session.pop('is_student', None)
            messages.success(request, "You have been logged out successfully.")
            
        return redirect('home')
    except Exception as e:
        ErrorHandler.handle_error(e, "Student logout")
        messages.error(request, "An error occurred during logout.")
        return redirect('home')

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
                'id_card_captured': False,
                'error': None
            })
        
        # Return the status
        response = {
            'status': 'Capturing...',
            'face_detected': camera.face_detected,
            'face_saved': camera.face_saved,
            'error': camera.error_message,
        }
        
        # For student login, also check ID card status
        if camera_type == 'student_login':
            response['id_card_captured'] = getattr(camera, 'id_card_captured', False)
        
        # Add student info if available from ID card scanning during registration
        if hasattr(camera, 'student_created') and camera.student_created:
            response['student_name'] = camera.student_created.name
            response['student_id'] = camera.student_created.student_id
        
        # Add student info if available from face matching (for login)
        if hasattr(camera, 'matched_student') and camera.matched_student:
            response['student_name'] = camera.matched_student.name
            response['student_id'] = camera.matched_student.student_id
            response['is_verified'] = camera.matched_student.face_verified
            
            # Add similarity score if available (calculated during face matching)
            if hasattr(camera, 'best_similarity_score'):
                response['similarity_score'] = round(camera.best_similarity_score, 2)
            
            # If login is complete (matched and verified), add auto-login info
            if camera.capture_complete and camera.matched_student.face_verified:
                # Set login_success based on similarity score
                if hasattr(camera, 'best_similarity_score') and camera.best_similarity_score >= 0.5:
                    response['login_success'] = True
                    response['status'] = 'Login successful! Face verified.'
                else:
                    response['login_success'] = False
                    response['status'] = 'Login failed! Face not verified.'
                
                # Create a login session through AJAX only if verification passes
                if not hasattr(camera, 'best_similarity_score') or camera.best_similarity_score >= 0.5:
                    request.session['student_id'] = camera.matched_student.student_id
                    request.session['student_name'] = camera.matched_student.name
                    request.session['is_student'] = True
                    
                    # Log the successful login
                    logger.info(f"Student auto-logged in via face recognition: {camera.matched_student.name}")
            
        return JsonResponse(response)
        
    except Exception as e:
        ErrorHandler.handle_error(e, "Getting face detection status")
        return JsonResponse({
            'status': 'Error',
            'error': str(e),
            'face_detected': False,
            'face_saved': False,
            'id_card_captured': False
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
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed'}, status=405)
    
    try:
        # First check if there's a logged in student
        student_id = request.session.get('student_id')
        if not student_id:
            return JsonResponse({
                'status': 'error',
                'message': 'No student is logged in. Please log in first.'
            }, status=401)

        # Get the student from the session
        try:
            student = Student.objects.get(student_id=student_id)
        except Student.DoesNotExist:
            return JsonResponse({
                'status': 'error',
                'message': 'Student session is invalid. Please log in again.'
            }, status=401)

        # Get the tag_id from the request
        tag_id = request.POST.get('tag_id')
        if not tag_id:
            return JsonResponse({'status': 'error', 'message': 'No tag_id provided'}, status=400)
        
        # Get the book by tag_id
        try:
            book = Book.objects.get(tag_id=tag_id)
        except Book.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': f'No book found with tag_id {tag_id}'}, status=404)
        
        # Check if book is available
        if not book.available:
            return JsonResponse({'status': 'error', 'message': 'Book is already borrowed'}, status=400)
        
        # Check if student has any overdue books
        overdue_books = BorrowedBook.objects.filter(
            student=student,
            is_returned=False,
            due_date__lt=timezone.now()
        )
        if overdue_books.exists():
            return JsonResponse({
                'status': 'error',
                'message': 'Cannot borrow book. You have overdue books to return.'
            }, status=400)
        
        # Create new borrowed book record
        due_date = timezone.now() + timedelta(days=14)  # 2 weeks borrowing period
        borrowed_book = BorrowedBook.objects.create(
            book=book,
            student=student,
            borrowed_date=timezone.now(),
            due_date=due_date,
            is_returned=False
        )
        
        # Update book availability
        book.available = False
        book.save()

        # Log the successful borrowing
        logger.info(f"Book borrowed successfully - Student: {student.name} ({student.student_id}), Book: {book.title} (Tag: {tag_id})")
        
        return JsonResponse({
            'status': 'success',
            'message': 'Book borrowed successfully',
            'data': {
                'book_title': book.title,
                'student_name': student.name,
                'due_date': due_date.strftime('%Y-%m-%d %H:%M:%S'),
                'borrowed_book_id': borrowed_book.id
            }
        })
        
    except Exception as e:
        logger.error(f"Error in borrow_detected_book: {str(e)}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def process_detected_book(request):
    """Process a detected book and initiate borrowing if conditions are met"""
    try:
        # Check if student is logged in
        student_id = request.session.get('student_id')
        if not student_id:
            return JsonResponse({
                'status': 'error',
                'message': 'No student is logged in. Please log in first.'
            }, status=401)

        # Get tag_id from the request
        tag_id = request.POST.get('tag_id')
        if not tag_id:
            return JsonResponse({'status': 'error', 'message': 'No tag_id provided'})
        
        # Check if book exists
        try:
            book = Book.objects.get(tag_id=tag_id)
        except Book.DoesNotExist:
            # If book doesn't exist but we have mapping info, create it
            if tag_id in tag_to_book_mapping:
                book_info = tag_to_book_mapping[tag_id]
                book = store_detected_book(tag_id, book_info)
                if not book:
                    return JsonResponse({
                        'status': 'error',
                        'message': f'Failed to create book for tag {tag_id}'
                    })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': f'No book found with tag {tag_id}'
                })
        
        # Update last detection time
        book.last_detected = timezone.now()
        book.save()
        
        # Get the student from session
        try:
            student = Student.objects.get(student_id=student_id)
        except Student.DoesNotExist:
            return JsonResponse({
                'status': 'error',
                'message': 'Student not found. Please log in again.'
            }, status=401)
        
        # If book is available and student is logged in, initiate borrowing
        if book.available:
            # Directly call borrow_detected_book with the same request
            request.POST = request.POST.copy()  # Make POST mutable
            request.POST['tag_id'] = tag_id  # Ensure tag_id is in POST
            borrow_response = borrow_detected_book(request)
            
            # Parse the response
            import json
            response_data = json.loads(borrow_response.content)
            
            if borrow_response.status_code == 200:
                return JsonResponse({
                    'status': 'success',
                    'message': 'Book detected and borrowed successfully',
                    'book_info': {
                        'title': book.title,
                        'author': book.author,
                        'student': student.name,
                        'borrowed_book_id': response_data.get('data', {}).get('borrowed_book_id')
                    }
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Book detected but borrowing failed',
                    'error': response_data.get('message', 'Unknown error')
                })
        else:
            # Check if this book is borrowed by the current student
            current_borrow = BorrowedBook.objects.filter(
                book=book,
                student=student,
                is_returned=False
            ).first()
            
            if current_borrow:
                return JsonResponse({
                    'status': 'info',
                    'message': 'You have already borrowed this book',
                    'book_info': {
                        'title': book.title,
                        'author': book.author,
                        'due_date': current_borrow.due_date.strftime('%Y-%m-%d %H:%M:%S')
                    }
                })
            else:
                return JsonResponse({
                    'status': 'warning',
                    'message': 'Book detected but is already borrowed by another student',
                    'book_info': {
                        'title': book.title,
                        'author': book.author,
                        'available': False
                    }
                })
            
    except Exception as e:
        logger.error(f"Error processing detected book: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'Error processing book: {str(e)}'
        })

@csrf_exempt
def return_book(request):
    """API view to handle book returns"""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed'}, status=405)
    
    try:
        # Get the borrowed book activity ID from the request
        activity_id = request.POST.get('activity_id')
        if not activity_id:
            return JsonResponse({'status': 'error', 'message': 'No activity_id provided'}, status=400)
        
        # Get the borrowed book record
        try:
            borrowed_book = BorrowedBook.objects.get(id=activity_id)
        except BorrowedBook.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Borrowed book record not found'}, status=404)
        
        # Check if book is already returned
        if borrowed_book.is_returned:
            return JsonResponse({'status': 'error', 'message': 'Book is already returned'}, status=400)
        
        # Update the borrowed book record
        borrowed_book.is_returned = True
        borrowed_book.returned_date = timezone.now()
        borrowed_book.save()
        
        # Update book availability
        book = borrowed_book.book
        book.available = True
        book.save()
        
        return JsonResponse({
            'status': 'success',
            'message': 'Book returned successfully',
            'data': {
                'book_title': book.title,
                'student_name': borrowed_book.student.name,
                'returned_date': borrowed_book.returned_date.strftime('%Y-%m-%d %H:%M:%S')
            }
        })
        
    except Exception as e:
        logger.error(f"Error in return_book: {str(e)}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)