from django.core.management.base import BaseCommand
from library.models import Book

class Command(BaseCommand):
    help = 'Loads sample books into the database'

    def handle(self, *args, **kwargs):
        sample_books = [
            {
                'title': 'The Great Gatsby',
                'author': 'F. Scott Fitzgerald',
                'genre': 'Classic',
                'description': 'A story of decadence and excess...',
                'rating': 4.5
            },
            {
                'title': '1984',
                'author': 'George Orwell',
                'genre': 'Science Fiction',
                'description': 'A dystopian social science fiction novel...',
                'rating': 4.8
            },
            {
                'title': 'To Kill a Mockingbird',
                'author': 'Harper Lee',
                'genre': 'Classic',
                'description': 'A story of racial injustice...',
                'rating': 4.7
            },
            {
                'title': 'Pride and Prejudice',
                'author': 'Jane Austen',
                'genre': 'Romance',
                'description': 'A classic romance novel...',
                'rating': 4.6
            },
            {
                'title': 'The Hobbit',
                'author': 'J.R.R. Tolkien',
                'genre': 'Fantasy',
                'description': 'A fantasy novel about Bilbo Baggins...',
                'rating': 4.9
            },
            {
                'title': 'Dune',
                'author': 'Frank Herbert',
                'genre': 'Science Fiction',
                'description': 'A science fiction masterpiece...',
                'rating': 4.7
            }
        ]

        for book_data in sample_books:
            Book.objects.create(**book_data)

        self.stdout.write(self.style.SUCCESS('Successfully loaded sample books')) 