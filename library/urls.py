from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Home page as root URL
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
    path('student_login/', views.student_login, name='student_login'),
]