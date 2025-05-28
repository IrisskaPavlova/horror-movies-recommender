import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

class MovieRecommender:
    def __init__(self, db, n_neighbors=5, user_similarity_metric='cosine'):
        self.db = db
        self.user_similarity_metric = user_similarity_metric.lower()
        self.movie_features = self.prepare_movie_features()
        self.user_features = self.prepare_user_features()
        self.movie_similarity = cosine_similarity(self.movie_features)

        if self.user_similarity_metric == 'cosine':
            self.user_similarity = cosine_similarity(self.user_features)
        elif self.user_similarity_metric == 'jaccard':
            self.user_similarity = self.compute_jaccard_user_similarity()
        else:
            raise ValueError(f"Unsupported user similarity metric: {user_similarity_metric}")

        self.neighbours = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(self.movie_features)
        self.last_recommendations = []
        self.last_index = 0

    def update_movie_features(self):
        """Обновление матрицы оценок после добавления новой оценки"""
        self.movie_features = self.prepare_movie_features()
        self.user_features = self.prepare_user_features()
        self.movie_similarity = cosine_similarity(self.movie_features)

        if self.user_similarity_metric == 'cosine':
            self.user_similarity = cosine_similarity(self.user_features)
        elif self.user_similarity_metric == 'jaccard':
            self.user_similarity = self.compute_jaccard_user_similarity()

    def prepare_movie_features(self):
        """Создание матрицы фильм x пользователь с рейтингами для фильмов ужасов"""
        self.db.ratings = pd.read_csv(self.db.ratings_file)  # Обновляем данные из файла
        horror_movies = self.db.get_horror_movies()

        if horror_movies.empty:
            raise ValueError("No horror movies found in the database.")

        horror_movie_ids = horror_movies['movieId'].unique()
        unique_user_ids = self.db.ratings['userId'].unique()

        if len(unique_user_ids) == 0:
            raise ValueError("No users found in the ratings database.")

        movie_id_to_index = {movie_id: i for i, movie_id in enumerate(horror_movie_ids)}
        user_id_to_index = {user_id: i for i, user_id in enumerate(unique_user_ids)}
        matrix = np.zeros((len(horror_movie_ids), len(unique_user_ids)))

        for row in self.db.ratings.itertuples():
            if row.movieId in movie_id_to_index:
                movie_index = movie_id_to_index[row.movieId]
                user_index = user_id_to_index[row.userId]
                matrix[movie_index, user_index] = row.rating

        self.horror_movie_ids = horror_movie_ids
        self.movie_id_to_index = movie_id_to_index
        self.unique_user_ids = unique_user_ids
        self.user_id_to_index = user_id_to_index

        return matrix

    def prepare_user_features(self):
        """Создание матрицы пользователь x фильм"""
        return self.movie_features.T

    def compute_jaccard_user_similarity(self):
        """Вычисление Jaccard similarity матрицы пользователей."""
        n_users = len(self.unique_user_ids)
        user_movie_binary = (self.user_features > 0).astype(int)

        jaccard_sim = np.zeros((n_users, n_users))

        for i in range(n_users):
            set_i = set(np.where(user_movie_binary[i] == 1)[0])
            for j in range(i, n_users):
                set_j = set(np.where(user_movie_binary[j] == 1)[0])
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                sim = intersection / union if union > 0 else 0.0
                jaccard_sim[i, j] = sim
                jaccard_sim[j, i] = sim

        return jaccard_sim

    def get_similar_users(self, user_id, top=5):
        """Возвращает индексы наиболее похожих пользователей для заданного user_id"""
        if user_id not in self.user_id_to_index:
            return []

        user_index = self.user_id_to_index[user_id]
        similarity_scores = self.user_similarity[user_index]

        # Исключаем самого пользователя
        similarity_scores[user_index] = -1
        similar_users_indices = np.argsort(similarity_scores)[::-1][:top]
        return similar_users_indices

    def get_recommendations(self, user_id, watched_movie_ids, top=5):
        """Гибридный метод рекомендаций с использованием схожести фильмов и пользователей"""
        watched_movies_set = set(watched_movie_ids)
        recommendations_scores = np.zeros(len(self.horror_movie_ids))

        if user_id not in self.user_id_to_index:
            popular_indices = np.argsort(np.sum(self.movie_features, axis=1))[::-1]
            recs = [self.horror_movie_ids[i] for i in popular_indices if self.horror_movie_ids[i] not in watched_movies_set]
            return recs[:top]

        user_index = self.user_id_to_index[user_id]
        user_ratings = self.user_features[user_index]

        similar_users_indices = self.get_similar_users(user_id, top=5)

        if len(similar_users_indices) > 0:
            sim_scores = self.user_similarity[user_index][similar_users_indices]
            sim_users_ratings = self.user_features[similar_users_indices]
            weighted_ratings = np.dot(sim_scores, sim_users_ratings) / (np.sum(np.abs(sim_scores)) + 1e-9)
        else:
            weighted_ratings = np.zeros_like(user_ratings)

        combined_profile = np.where(user_ratings > 0, user_ratings, weighted_ratings)

        scores = self.movie_similarity.dot(combined_profile)

        ranked_indices = np.argsort(scores)[::-1]
        all_recs = [self.horror_movie_ids[idx] for idx in ranked_indices if self.horror_movie_ids[idx] not in watched_movies_set]

        if not all_recs:
            return []

        start = self.last_index
        end = start + top
        recommendations = all_recs[start:end]

        if end >= len(all_recs):
            self.last_index = 0
        else:
            self.last_index = end

        mae = None
        rmse = None
        r2 = None
        try:
            true_ratings = combined_profile[combined_profile > 0]
            pred_scores = scores[combined_profile > 0]
            if len(true_ratings) > 0 and len(pred_scores) > 0:
                mae = mean_absolute_error(true_ratings, pred_scores)
                rmse = mean_squared_error(true_ratings, pred_scores)
                r2 = r2_score(true_ratings, pred_scores)
        except Exception as e:
            print(f"Error calculating metrics: {e}")

        if mae is not None and rmse is not None and r2 is not None:
            print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        else:
            print("MAE, RMSE и R2 не могут быть вычислены из-за отсутствия данных")

        return recommendations
