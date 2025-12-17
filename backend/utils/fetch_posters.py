import pandas as pd
import requests
import os
from dotenv import load_dotenv
from time import sleep

# Load your TMDB API key from .env file
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

# Paths
MOVIE_PATH = "backend/data/movie.csv"
LINK_PATH = "backend/data/link.csv"

# Load datasets
movies = pd.read_csv(MOVIE_PATH)
links = pd.read_csv(LINK_PATH)

# Merge with link.csv to get tmdbId
movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")

# Function to get poster URL using TMDB API
def get_poster_url(tmdb_id):
    if pd.isna(tmdb_id):
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "poster_path" in data and data["poster_path"]:
                return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
        else:
            print(f"⚠️ Skipping {tmdb_id}, status {response.status_code}")
    except Exception as e:
        print(f"Error fetching {tmdb_id}: {e}")
    return None

# Limit (optional, for testing)
# movies = movies.head(100)

# Add poster URLs
posters = []
for i, row in movies.iterrows():
    posters.append(get_poster_url(row["tmdbId"]))
    if i % 20 == 0:  # every 20 requests
        print(f"Fetched {i} posters...")
        sleep(1)  # small delay to avoid hitting API limits

movies["poster_url"] = posters

# Save directly back to the same file
movies.to_csv(MOVIE_PATH, index=False)
print("✅ Done! Posters added inside movie.csv 🎬")
