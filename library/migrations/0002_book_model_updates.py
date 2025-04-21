# Generated manually
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('library', '0001_initial'),
    ]

    operations = [
        # Add borrow_count field if it doesn't exist
        migrations.AddField(
            model_name='book',
            name='borrow_count',
            field=models.IntegerField(default=0),
        ),
        # Add last_borrowed field if it doesn't exist
        migrations.AddField(
            model_name='book',
            name='last_borrowed',
            field=models.DateTimeField(blank=True, null=True),
        ),
        # Add is_available field
        migrations.AddField(
            model_name='book',
            name='is_available',
            field=models.BooleanField(default=True),
        ),
        # Add added_date field
        migrations.AddField(
            model_name='book',
            name='added_date',
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
        # Rename available to is_available (preserve data)
        migrations.RunSQL(
            sql="""
            UPDATE library_book
            SET is_available = available
            WHERE available IS NOT NULL;
            """,
            reverse_sql="""
            UPDATE library_book
            SET available = is_available
            WHERE is_available IS NOT NULL;
            """
        ),
    ] 