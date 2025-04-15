from django.contrib import admin
from .models import Book, Student, BorrowedBook

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

class StudentAdmin(admin.ModelAdmin):
    list_display = ('name', 'student_id', 'email')
    search_fields = ('name', 'student_id', 'email')

class BorrowedBookAdmin(admin.ModelAdmin):
    list_display = ('book', 'student', 'borrowed_date', 'due_date', 'is_returned', 'returned_date')
    list_filter = ('is_returned',)
    search_fields = ('book__title', 'student__name')
    date_hierarchy = 'borrowed_date'

admin.site.register(Book, BookAdmin)
admin.site.register(Student, StudentAdmin)
admin.site.register(BorrowedBook, BorrowedBookAdmin)
