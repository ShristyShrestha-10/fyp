from django.db import models
from django.utils import timezone

# Create your models here.
class Student(models.Model):
    name = models.CharField(max_length=100)
    id_card_image = models.ImageField(upload_to='id_cards/')
    registration_date = models.DateTimeField(auto_now_add=True)

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
    