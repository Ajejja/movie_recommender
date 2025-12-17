import pickle
import pandas as pd

MOVIE_PATH = "data/movie.csv"
MODEL_PATH = "backend/model/recommender.pkl"

# Load model
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# Load movies
def load_movies():
    return pd.read_csv(MOVIE_PATH)

# Recommend movies
def recommend_movies(user_id, n=12):
    model = load_model()
    movies = load_movies()

    # Predict ratings for all movies
    movie_ids = movies["movieId"].unique()
    preds = [(mid, model.predict(user_id, mid).est) for mid in movie_ids]

    # Sort by predicted rating
    top_movies = sorted(preds, key=lambda x: x[1], reverse=True)[:n]
    top_movie_ids = [mid for mid, _ in top_movies]

    recommendations = movies[movies["movieId"].isin(top_movie_ids)][["title", "poster_url", "genres"]]
    return recommendations.to_dict(orient="records")
