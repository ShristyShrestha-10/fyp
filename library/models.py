from django.db import models
import numpy as np
import pickle
from django.utils import timezone

# Create your models here.
class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=100, blank=True, default='')
    isbn = models.CharField(max_length=20, blank=True, default='')
    genre = models.CharField(max_length=50, blank=True, default='')
    description = models.TextField(default='')
    cover_image = models.ImageField(upload_to='book_covers/', null=True, blank=True)
    available = models.BooleanField(default=True)
    tag_id = models.IntegerField(unique=True, null=True, blank=True)
    last_detected = models.DateTimeField(null=True, blank=True)
    rating = models.FloatField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

class Student(models.Model):
    name = models.CharField(max_length=100)
    student_id = models.CharField(max_length=20, unique=True)
    id_valid_until = models.DateField()
    face_encoding = models.BinaryField(null=True, blank=True)
    registered_at = models.DateTimeField(auto_now_add=True)
    id_card_image = models.ImageField(
    upload_to='id_cards/',
    null=True,
    blank=True,
    default='default_id.jpg'  # Add this default value
)

    def get_face_encoding(self):
        """Convert binary face encoding back to numpy array"""
        return pickle.loads(self.face_encoding)
        
    def __str__(self):
        return f"{self.name} ({self.student_id})"

class BorrowedBook(models.Model):
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    borrowed_date = models.DateTimeField(default=timezone.now)
    due_date = models.DateTimeField()
    returned_date = models.DateTimeField(null=True, blank=True)
    is_returned = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.book.title} borrowed by {self.student.name}"

    class Meta:
        ordering = ['-borrowed_date']

class IDCardDetection(models.Model):
    student = models.OneToOneField(Student, on_delete=models.CASCADE)
    detection_timestamp = models.DateTimeField(auto_now_add=True)
    is_verified = models.BooleanField(default=False)

class IDCardScan(models.Model):
    student_id = models.CharField(max_length=50)
    face_image = models.ImageField(upload_to='face_scans/')
    id_card_image = models.ImageField(upload_to='id_scans/')
    scan_date = models.DateTimeField(default=timezone.now)
    confidence_score = models.FloatField()
    match_status = models.BooleanField(default=False)

    def __str__(self):
        return f"Scan for {self.student_id} on {self.scan_date}"
    