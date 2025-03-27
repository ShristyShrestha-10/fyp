from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('start_opencv/', views.start_opencv, name='start_opencv'),
]