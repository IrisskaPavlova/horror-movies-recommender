import os
import pandas as pd
from recommender import MovieRecommender


class MovieDatabase:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.movies_file = os.path.join(self.data_dir, 'movies.csv')
        self.ratings_file = os.path.join(self.data_dir, 'ratings.csv')


        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)


        self.movies = self._load_or_create(self.movies_file, ['movieId', 'title', 'genres', 'year'])
        self.ratings = self._load_or_create(self.ratings_file, ['userId', 'movieId', 'rating', 'timestamp'])


        # Инициализация MovieRecommender
        self.recommender = MovieRecommender(self)


    def _load_or_create(self, filepath, columns):
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            df = pd.DataFrame(columns=columns)
            df.to_csv(filepath, index=False)
            return df

    def add_rating(self, user_id, movie_id, rating):
        if movie_id not in self.movies['movieId'].values:
            raise ValueError(f"Фильм с ID {movie_id} не найден.")
        new_rating = {
            'userId': user_id,
            'movieId': movie_id,
            'rating': rating,
            'timestamp': pd.Timestamp.now().timestamp()
        }
        self.ratings = pd.concat([self.ratings, pd.DataFrame([new_rating])], ignore_index=True)
        self.ratings.to_csv(self.ratings_file, index=False)


        # Обновляем матрицу оценок в MovieRecommender
        self.recommender.update_movie_features()


    
    def get_user_ratings(self, user_id):
        return self.ratings[self.ratings['userId'] == user_id]
    
    def get_movie(self, movie_id):
        movie = self.movies[self.movies['movieId'] == movie_id]
        if movie.empty:
            raise ValueError(f"Фильм с ID {movie_id} не найден.")
        return movie.iloc[0]
    
    def search_movies(self, query):
        result = self.movies[self.movies['title'].str.contains(query, case=False, na=False)]
        return result if not result.empty else None

    def get_horror_movies(self):
        return self.movies[self.movies['genres'].str.contains('Ужасы', case=False, na=False)]
