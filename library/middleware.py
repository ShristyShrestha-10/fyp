import logging
import time
from django.utils import timezone
from django.shortcuts import redirect
from django.conf import settings
from django.urls import reverse, resolve
from django.contrib import messages

logger = logging.getLogger(__name__)

class SessionSecurityMiddleware:
    """
    Middleware to enforce session security and validation
    
    This middleware:
    1. Validates student sessions consistently across all views
    2. Checks for session timeouts and forces logout if needed
    3. Tracks session activity and updates last activity times
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.session_timeout = getattr(settings, 'SESSION_TIMEOUT', 1800)  # Default 30 minutes
        
        # URLs that don't require authentication
        self.public_urls = [
            'auth_page', 
            'student_login', 
            'register_student', 
            'admin_login', 
            'about', 
            'home',
            'stop_scanner',
            'video_feed',
            'capture_frame'
        ]
        
        # AJAX endpoints that should return JSON instead of redirecting
        self.api_urls = [
            'get_face_detection_status',
            'process_detected_book',
            'borrow_book',
            'return_book'
        ]
        
        logger.info(f"Session security middleware initialized with timeout: {self.session_timeout} seconds")
        
    def __call__(self, request):
        # Check if path is exempt from session checking
        path = request.path_info.lstrip('/')
        view_name = ''
        
        try:
            resolver_match = resolve(request.path)
            view_name = resolver_match.url_name
        except:
            pass
            
        # Skip middleware for public URLs
        if view_name in self.public_urls:
            return self.get_response(request)
            
        # Check if this is a student session
        is_student = request.session.get('is_student', False)
        
        if is_student:
            # Get last activity time from session
            last_activity = request.session.get('last_activity')
            
            # Check if session has timed out
            current_time = timezone.now().timestamp()
            
            if last_activity and (current_time - last_activity > self.session_timeout):
                logger.info(f"Session timeout for student, last activity: {last_activity}, current: {current_time}")
                
                # Clear session
                request.session.flush()
                
                # If this is an API call, return JSON instead of redirecting
                if view_name in self.api_urls:
                    from django.http import JsonResponse
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Session expired',
                        'redirect': reverse('auth_page')
                    }, status=401)
                    
                # Otherwise redirect to login
                return redirect('auth_page')
                
            # Update last activity timestamp
            request.session['last_activity'] = current_time
            
        # Process the request
        response = self.get_response(request)
        return response
    
    def _validate_session(self, request):
        """Validate the session and handle expired sessions"""
        # Skip validation for authentication urls
        auth_urls = [reverse('student_login'), reverse('admin_login'), reverse('admin_logout'), reverse('student_logout')]
        if request.path in auth_urls:
            return
            
        # Check for student session validity
        if 'is_student' in request.session and request.session['is_student']:
            # Check if student_id is present
            if 'student_id' not in request.session:
                logger.warning("Student session without student_id detected, clearing session")
                self._clear_student_session(request)
                messages.error(request, "Your session is invalid. Please login again.")
                return redirect('student_login')
                
            # Check for session timeout
            last_activity = request.session.get('last_activity')
            if last_activity:
                # Convert from timestamp to datetime
                last_activity_time = timezone.datetime.fromtimestamp(last_activity, tz=timezone.get_current_timezone())
                elapsed = timezone.now() - last_activity_time
                
                # Force logout after 30 minutes of inactivity
                if elapsed.total_seconds() > 30 * 60:
                    logger.info(f"Session timeout for student {request.session.get('student_id')}")
                    self._clear_student_session(request)
                    messages.info(request, "Your session has expired due to inactivity. Please login again.")
                    return redirect('student_login')
    
    def _update_session_activity(self, request):
        """Update the session's last activity timestamp"""
        if 'is_student' in request.session and request.session['is_student']:
            request.session['last_activity'] = time.time()
            request.session.modified = True
    
    def _clear_student_session(self, request):
        """Clear all student-related session data"""
        for key in ['student_id', 'student_name', 'is_student', 'last_activity', 'camera_active']:
            if key in request.session:
                del request.session[key]
        request.session.modified = True

class SessionMaintenanceMiddleware:
    """
    Middleware to maintain session activity for students and prevent timeouts
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        
    def __call__(self, request):
        # Update session activity timestamp for student sessions
        if 'student_id' in request.session:
            # Get session data for logging
            student_id = request.session.get('student_id')
            student_name = request.session.get('student_name', '')
            last_activity = request.session.get('last_activity', 0)
            login_time = request.session.get('login_time', 0)
            current_time = timezone.now().timestamp()
            
            # Calculate time since last activity and login
            time_since_activity = current_time - last_activity if last_activity else 0
            time_since_login = current_time - login_time if login_time else 0
            
            # Log session information for every request (debug level)
            logger.debug(
                f"Session active: {student_name} (ID: {student_id}), "
                f"Path: {request.path}, "
                f"Last activity: {time_since_activity:.1f}s ago, "
                f"Login: {time_since_login:.1f}s ago"
            )
            
            # If it's been more than 5 minutes since last update
            if time_since_activity > 300:  # 300 seconds = 5 minutes
                request.session['last_activity'] = current_time
                # Extend session timeout to 24 hours from now
                request.session.set_expiry(86400)  # 24 hours in seconds
                logger.info(
                    f"Updated session activity for {student_name} (ID: {student_id}), "
                    f"Session extended for 24 hours"
                )
        
        response = self.get_response(request)
        return response 