from django.contrib import admin
from .models import Book, Student, BorrowedBook, StudentLogin

@admin.register(Book)
class BookAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'isbn', 'genre', 'tag_id', 'available', 'last_detected')
    list_filter = ('genre', 'available')
    search_fields = ('title', 'author', 'isbn', 'tag_id')
    readonly_fields = ('last_detected',)
    fieldsets = (
        ('Book Information', {
            'fields': ('title', 'author', 'isbn', 'genre', 'description', 'rating')
        }),
        ('Tag Information', {
            'fields': ('tag_id', 'last_detected')
        }),
        ('Status', {
            'fields': ('available', 'cover_image')
        }),
    )

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ('name', 'student_id', 'registered_at')
    search_fields = ('name', 'student_id')
    list_filter = ('registered_at',)
    readonly_fields = ('face_encoding',)

@admin.register(BorrowedBook)
class BorrowedBookAdmin(admin.ModelAdmin):
    list_display = ('book', 'student', 'borrowed_date', 'due_date', 'is_returned')
    search_fields = ('book__title', 'student__name', 'student__student_id')
    list_filter = ('is_returned', 'borrowed_date')

@admin.register(StudentLogin)
class StudentLoginAdmin(admin.ModelAdmin):
    list_display = ('student', 'login_time', 'status', 'similarity_score', 'is_active')
    search_fields = ('student__name', 'student__student_id')
    list_filter = ('status', 'is_active', 'login_time')
    readonly_fields = ('login_time', 'similarity_score')