import React, { useEffect, useState } from "react";
import axios from "axios";
import { getRecommendations } from "../api/api";
import MovieCard from "../components/MovieCard";
import MovieAgent from "../components/MovieAgent";


export default function Home() {
  const [recommended, setRecommended] = useState([]);
  const [allMovies, setAllMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  const userId = localStorage.getItem("userId");

  // --- Fetch all movies ---
  const fetchAllMovies = async () => {
    try {
      const res = await axios.get("http://127.0.0.1:5001/movies");
      setAllMovies(res.data);
    } catch (err) {
      console.error("Error loading movies:", err);
      setError("Failed to load all movies.");
    }
  };

  // --- Fetch recommendations ---
  const fetchRecommendations = async () => {
    if (!userId) return;
    try {
      const res = await getRecommendations(userId);
      setRecommended(res.data);
    } catch (err) {
      console.error("Error loading recommendations:", err);
      setError("Failed to load recommendations.");
    }
  };

  // --- Initial load ---
  useEffect(() => {
    if (userId) {
      (async () => {
        setLoading(true);
        await fetchAllMovies();
        await fetchRecommendations();
        setLoading(false);
      })();
    }
  }, [userId]);

  // --- Update recommendations after liking a movie ---
  const handleRecommendationsUpdate = async (newRecs) => {
    setRefreshing(true);
    try {
      if (newRecs && newRecs.length > 0) {
        // Directly use updated recommendations from MovieCard
        setRecommended(newRecs);
      } else {
        // Otherwise re-fetch to ensure fresh model output
        await fetchRecommendations();
      }
    } catch (err) {
      console.error("Error updating recommendations:", err);
    } finally {
      setRefreshing(false);
    }
  };

  // --- UI ---
  return (
  <div className="home-container" style={{ padding: "20px", position: "relative" }}>
    <h2 style={{ marginBottom: "10px" }}>🎬 Recommended Movies</h2>

    {error && <p style={{ color: "red" }}>{error}</p>}

    {loading ? (
      <p>Loading movies...</p>
    ) : (
      <>
        {refreshing && (
          <p style={{ color: "#666", marginBottom: "10px" }}>
            🔄 Updating recommendations...
          </p>
        )}

        <div
          className="movie-grid"
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "20px",
            marginBottom: "40px",
          }}
        >
          {recommended.length > 0 ? (
            recommended.map((movie) => (
              <MovieCard
                key={movie.movieId}
                movie={movie}
                onRecommendationsUpdate={handleRecommendationsUpdate}
              />
            ))
          ) : (
            <p>No recommendations yet. Try liking some movies below 👇</p>
          )}
        </div>

        <h2>🎥 All Movies</h2>
        <div
          className="movie-grid"
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "20px",
            marginTop: "10px",
          }}
        >
          {allMovies.map((movie) => (
            <MovieCard
              key={movie.movieId}
              movie={movie}
              onRecommendationsUpdate={handleRecommendationsUpdate}
            />
          ))}
        </div>
      </>
    )}

    {/* 🌟 Floating chatbot */}
    <div style={{ position: "fixed", bottom: "20px", right: "20px", zIndex: 9999 }}>
      <MovieAgent />
    </div>
  </div>
);
}