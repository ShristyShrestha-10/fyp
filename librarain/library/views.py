from django.shortcuts import render, redirect
from django.http import JsonResponse
import os
import sys
import subprocess
from django.conf import settings

def home(request):
    return render(request, 'library/home.html')

# Create your views here.
def start_opencv(request):
    try:
        # Get the correct path to the OpenCV script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, 'opencv_registration.py')
        
        if not os.path.exists(script_path):
            raise FileNotFoundError("OpenCV script not found")

        # Start the OpenCV process
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        # Redirect back to home page
        return redirect('home')

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

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