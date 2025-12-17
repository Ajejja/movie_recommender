import React, { useEffect, useState } from "react";
import MovieCard from "../components/MovieCard";

export default function Watchlist() {
  const [movies, setMovies] = useState([]);

  const load = () => {
    const uid = localStorage.getItem("userId") || "guest";
    const key = `watchlist_${uid}`;
    try {
      const arr = JSON.parse(localStorage.getItem(key) || "[]");
      setMovies(arr);
    } catch (e) {
      setMovies([]);
    }
  };

  useEffect(() => {
    load();

    const handler = () => load();
    window.addEventListener("watchlistUpdated", handler);
    return () => window.removeEventListener("watchlistUpdated", handler);
  }, []);

  const remove = (movieId) => {
    const uid = localStorage.getItem("userId") || "guest";
    const key = `watchlist_${uid}`;
    const existing = JSON.parse(localStorage.getItem(key) || "[]");
    const updated = existing.filter((m) => String(m.movieId) !== String(movieId));
    localStorage.setItem(key, JSON.stringify(updated));
    setMovies(updated);
  };

  return (
    <div className="container watchlist-container">
      <h1 style={{ marginTop: 8 }}>🎬 Your Watchlist</h1>

      {movies.length === 0 ? (
        <p>You haven't added any movies yet. Click "Add to List" on a movie to save it here.</p>
      ) : (
        <div className="movie-grid watchlist-grid">
          {movies.map((m) => (
            <div key={m.movieId} className="watchlist-item">
              <div className="card-wrapper">
                <MovieCard movie={m} onRecommendationsUpdate={() => {}} showActions={false} />
                <button
                  className="remove-btn"
                  onClick={() => remove(m.movieId)}
                >
                  Remove from Watchlist
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
