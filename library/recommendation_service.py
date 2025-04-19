import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import gc
import os
from django.conf import settings

class BookRecommender:
    def __init__(self):
        self.df = None
        self.load_books_data()
    
    def load_books_data(self):
        try:
            file_path = os.path.join(settings.BASE_DIR, 'dataset_books.csv')
            # Define column names based on dataset format
            columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 
                      'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
            
            # Read CSV file
            self.df = pd.read_csv(file_path, names=columns, encoding='utf-8')
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_recommendations(self, book_title, sample_size=5000, top_n=3):
        if self.df is None:
            return []
            
        # Check if book exists in dataset
        if book_title not in self.df['Book-Title'].values:
            return []
        
        try:
            # Get the book's index
            target_idx = self.df[self.df['Book-Title'] == book_title].index[0]
            
            # Take a random sample if dataset is too large
            # Always include the target book in the sample
            if len(self.df) > sample_size:
                other_indices = self.df.index.drop(target_idx).to_list()
                sample_indices = np.random.choice(other_indices, size=min(sample_size-1, len(other_indices)), replace=False)
                sample_indices = np.append(sample_indices, target_idx)
                df_sample = self.df.loc[sample_indices].copy()
            else:
                df_sample = self.df.copy()
            
            # Fill missing values with empty strings
            df_sample['Book-Author'] = df_sample['Book-Author'].fillna('')
            df_sample['Publisher'] = df_sample['Publisher'].fillna('')
            df_sample['Year-Of-Publication'] = df_sample['Year-Of-Publication'].fillna('').astype(str)
            
            # Create a feature representation for each book
            df_sample['features'] = df_sample['Book-Author'] + ' ' + df_sample['Publisher'] + ' ' + df_sample['Year-Of-Publication']
            
            # Create TF-IDF matrix
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df_sample['features'])
            
            # Get new index of the target book in the sample
            sample_target_idx = df_sample.index.get_indexer([target_idx])[0]
            
            # Calculate cosine similarity only for the target book (memory efficient)
            cosine_similarities = linear_kernel(tfidf_matrix[sample_target_idx:sample_target_idx+1], tfidf_matrix).flatten()
            
            # Get indices of top similar books (excluding the book itself)
            similar_indices = cosine_similarities.argsort()[::-1][1:top_n+1]
            
            # Get the recommended books
            recommendations = []
            for idx in similar_indices:
                book = df_sample.iloc[idx]
                recommendations.append({
                    'title': book['Book-Title'],
                    'author': book['Book-Author'],
                    'year': book['Year-Of-Publication'],
                    'publisher': book['Publisher'],
                    'isbn': book['ISBN'],
                    'image_url': book['Image-URL-M']  # Using medium size image
                })
            
            # Free up memory
            gc.collect()
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return [] 