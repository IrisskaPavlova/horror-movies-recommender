import os
import requests
from flask import Flask, render_template, request, redirect, url_for, flash
from database import MovieDatabase
from recommender import MovieRecommender
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Для использования flash сообщений
db = MovieDatabase()
recommender = MovieRecommender(db)
KINOPISK_API_KEY = 'FXF5JJQ-G4DMJDC-PHSBSA0-K68GBET'  # Получаем ключ из переменных окружения
KINOPISK_BASE_URL = 'https://api.kinopoisk.dev/v1.3'  # базовый URL для Кинопоиска

def fetch_movie_details(movie_id):
    if KINOPISK_API_KEY is None:
        print("Ошибка: API ключ не установлен.")
        return None


    url = f"{KINOPISK_BASE_URL}/movie/{movie_id}"  # Используем базовый URL
    headers = {
        'X-API-KEY': KINOPISK_API_KEY
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()


        # Извлечение необходимых данных с проверкой на None
        year = data.get('year', 'N/A')
        countries = ', '.join(country['name'] for country in data.get('countries', [])) or 'N/A'
        genres = ', '.join(genre['name'] for genre in data.get('genres', [])) or 'N/A'
        slogan = data.get('slogan', 'N/A')


        directors = [p['name'] for p in data.get('persons', []) if p['profession'] == 'режиссеры']
        producers = [p['name'] for p in data.get('persons', []) if p['profession'] == 'продюсеры']
        operators = [p['name'] for p in data.get('persons', []) if p['profession'] == 'операторы']
        composers = [p['name'] for p in data.get('persons', []) if p['profession'] == 'композиторы']


        budget = data.get('budget', {'value': 0, 'currency': ''})
        budget_value = budget.get('value', 0)  # Используйте get для безопасного доступа
        budget_str = f"{budget_value} {budget.get('currency', '')}" if budget_value > 0 else 'N/A'

        premiere_date= data.get('premiere', {}).get('russia', 'N/A')
        if premiere_date != 'N/A' and premiere_date != None:
            premiere_date=(premiere_date).split('T')[0]
            date_obj = datetime.strptime(premiere_date, '%Y-%m-%d')
            premiere_date = date_obj.strftime('%d.%m.%Y')
        else:
            if premiere_date == None:
                premiere_date = "Неизвестно"    
            else:
                premiere_date = premiere_date

        age_rating = data.get('ageRating', 'N/A')
        rating = data.get('rating', {}).get('kp', 'N/A')
        movie_length = data.get('movieLength', 'N/A')


        poster_path = data.get('poster', {}).get('url', None)
        overview = data.get('description', '')
        title = data.get('name', '')


        return {
            'id': movie_id,
            'poster_path': poster_path,
            'overview': overview,
            'title': title,
            'year': year,
            'countries': countries,
            'genres': genres,
            'slogan': slogan,
            'director': ', '.join(directors) if directors else 'N/A',
            'producer': ', '.join(producers) if producers else 'N/A',
            'operator': ', '.join(operators) if operators else 'N/A',
            'composer': ', '.join(composers) if composers else 'N/A',
            'budget': budget_str,
            'premiere_date': premiere_date,
            'age_rating': age_rating,
            'rating': rating,
            'movie_length': movie_length
        }


    except requests.exceptions.RequestException as err:
        print(f"Ошибка при запросе к Кинопоиск API: {err}")
        return None

@app.route('/')
def home():
    return render_template('index.html', found_movies=None, recommendations=None)

@app.route('/random_movie')
def random_movie():
    random_movie = db.movies.sample(n=1)  # Выбираем случайный фильм из базы данных
    if not random_movie.empty:
        movie_id = random_movie.iloc[0]['movieId']
        return redirect(url_for('movie_details', movie_id=movie_id))
    else:
        flash('Не удалось найти случайный фильм.', 'error')
        return redirect(url_for('home'))

@app.route('/rate', methods=['POST'])
def rate_movie():
    movie_id = request.form['movie_id']
    rating = request.form['rating']
    user_id = 612  # Замените на реальный ID пользователя

    try:
        db.add_rating(user_id, int(movie_id), float(rating))
        flash('Оценка добавлена!', 'success')
    except Exception as e:
        flash(f'Ошибка: {e}', 'error')

    return redirect(url_for('home'))

@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    details = fetch_movie_details(movie_id)
    if details is None:
        flash('Фильм не найден.', 'error')
        return redirect(url_for('home'))

    return render_template('movie_details.html', movie=details)

@app.route('/search', methods=['POST'])
def search_movie():
    query = request.form['search_query']
    found_movies = db.search_movies(query)

    # Создаем копию найденных фильмов
    if found_movies is not None and not found_movies.empty:
        found_movies = found_movies.copy()  # Создаем копию DataFrame
        for idx, movie in found_movies.iterrows():
            details = fetch_movie_details(movie['movieId'])
            found_movies.loc[idx, 'poster'] = details['poster_path'] if details['poster_path'] else None
            found_movies.loc[idx, 'description'] = details['overview'] if details['overview'] else ""
    return render_template('index.html', found_movies=found_movies, db=db, recommendations=None)

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    user_id = 612  # Замените на реальный ID пользователя
    watched_movies = []  # Замените на список просмотренных фильмов

    recommendations = recommender.get_recommendations(user_id, watched_movies)
    
    # Для рекомендаций загрузим постеры и описания из Кинопоиска
    detailed_recs = []
    for movie_id in recommendations:
        try:
            movie = db.get_movie(movie_id)
            details = fetch_movie_details(movie_id)
            detailed_recs.append({
                'movieId': movie_id,
                'title': movie.title,
                'genres': movie.genres,
                'poster': details['poster_path'] if details['poster_path'] else None,
                'description': details['overview'] if details['overview'] else ""
            })
        except Exception as e:
            print(f"Ошибка при обработке рекомендации фильма {movie_id}: {e}")
    return render_template('index.html', recommendations=detailed_recs, db=db, found_movies=None)

if __name__ == '__main__':
    app.run(debug=True)
