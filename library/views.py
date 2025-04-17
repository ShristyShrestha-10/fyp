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
from .camera_factory import CameraFactory
from .utils import ErrorHandler
from django.contrib import messages

# Set up logging
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
                'show_retry': True
            })
            
        return render(request, 'library/camera_feed.html', {
            'camera_ready': True,
            'scanner_type': scanner_type
        })
        
    except Exception as e:
        ErrorHandler.handle_error(e, "Camera feed")
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
            ErrorHandler.handle_error(e, "Stopping scanner")
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
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
        camera.stop()

@gzip.gzip_page
def video_feed(request, camera_type='id_card'):
    """Stream the camera feed"""
    try:
        # Create camera using factory
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
                    # Save the face image if face is detected
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
                        'face_image': face_image
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

@login_required
def admin_dashboard(request):
    """Admin dashboard view"""
    try:
        # Get basic statistics
        total_books = Book.objects.count()
        total_students = Student.objects.filter(student_id__isnull=False).count()
        books_borrowed = BorrowedBook.objects.filter(is_returned=False).count()
        overdue_books = BorrowedBook.objects.filter(
            is_returned=False,
            due_date__lt=timezone.now()
        ).count()
        
        # Get verified students (those with both name and ID)
        verified_students = Student.objects.filter(
            student_id__isnull=False,
            name__isnull=False
        ).order_by('-registered_at')
        
        # Get recent registrations (only name and ID)
        recent_registrations = Student.objects.filter(
            student_id__isnull=False,
            name__isnull=False
        ).order_by('-registered_at')[:5]
        
        context = {
            'total_books': total_books,
            'total_students': total_students,
            'books_borrowed': books_borrowed,
            'overdue_books': overdue_books,
            'verified_students': verified_students,
            'recent_registrations': recent_registrations,
        }
        
        return render(request, 'library/admin_dashboard.html', context)
        
    except Exception as e:
        logger.error(f"Error in admin dashboard: {str(e)}")
        messages.error(request, "An error occurred while loading the dashboard")
        return redirect('home')

def student_login(request):
    """View for student login with face verification"""
    if request.method == 'POST':
        # Handle face verification and login
        pass
    return render(request, 'library/student_login.html', {
        'camera_type': 'student_login'
    })