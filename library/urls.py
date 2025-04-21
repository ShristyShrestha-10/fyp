from django.urls import path
from . import views

urlpatterns = [
    path('', views.auth_page, name='auth_page'),  # Default landing page is now auth page
    path('home/', views.home, name='home'),  # Move home to its own URL
    path('camera-feed/', views.camera_feed, name='camera_feed'),
    path('video-feed/<str:camera_type>/', views.video_feed, name='video_feed'),
    path('capture-frame/<str:camera_type>/', views.capture_frame, name='capture_frame'),
    path('stop-scanner/', views.stop_scanner, name='stop_scanner'),
    path('process-id-card/', views.process_id_card, name='process_id_card'),
    path('main-page/', views.main_page, name='main_page'),
    path('recommendations/', views.recommendations, name='recommendations'),
    path('about/', views.about, name='about'),  # About page URL
    path('admin/login/', views.admin_login, name='admin_login'),
    path('admin/logout/', views.admin_logout, name='admin_logout'),
    path('admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('student-login/', views.student_login, name='student_login'),
    path('student-logout/', views.student_logout, name='student_logout'),
    path('register-student/', views.register_student, name='register_student'),
    path('get-face-detection-status/', views.get_face_detection_status, name='get_face_detection_status'),
    path('delete-book/<int:book_id>/', views.delete_book, name='delete_book'),
    path('delete-activity/<int:activity_id>/', views.delete_activity, name='delete_activity'),
    path('delete-student/<int:student_id>/', views.delete_student, name='delete_student'),
    path('api/borrow-book/', views.borrow_detected_book, name='borrow_book'),
    path('api/process-detected-book/', views.process_detected_book, name='process_detected_book'),
    path('api/return-book/', views.return_book, name='return_book'),
]