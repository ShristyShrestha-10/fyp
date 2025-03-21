from django.shortcuts import render


def home(request):
    return render(request, 'library/home.html')

# Create your views here.
