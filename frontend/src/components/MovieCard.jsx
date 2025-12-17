import React, { useState } from "react";
import { recordAction, getUpdatedRecommendations } from "../api/api";

export default function MovieCard({ movie, onRecommendationsUpdate, showActions = true }) {
  const [loading, setLoading] = useState(false);

  const handleAction = async (actionType) => {
    const userId = localStorage.getItem("userId");
    if (!userId) {
      alert("Please log in first!");
      return;
    }

    try {
      setLoading(true);

      // ✅ Record the user action
      const res = await recordAction({
        userId: parseInt(userId),
        movieId: movie.movieId,
        action: actionType,
      });

      console.log(`✅ Action: ${actionType} → ${movie.title}`);

      // If user added to list, persist locally so Watchlist page can show it instantly.
      if (actionType === "added_to_list") {
        try {
          const uid = localStorage.getItem("userId") || "guest";
          const key = `watchlist_${uid}`;
          const existing = JSON.parse(localStorage.getItem(key) || "[]");

          // store minimal movie object but keep poster and genres
          const entry = {
            movieId: movie.movieId,
            title: movie.title,
            poster_url: movie.poster_url,
            genres: movie.genres,
          };

          const merged = [entry, ...existing.filter((m) => m.movieId != entry.movieId)];
          localStorage.setItem(key, JSON.stringify(merged));

          // let other pages know
          window.dispatchEvent(new CustomEvent("watchlistUpdated", { detail: { movie: entry } }));
        } catch (e) {
          console.error("Failed to save to local watchlist:", e);
        }
      }

      // ✅ Use updated recommendations if backend returned them
      if (
        res.data.updated_recommendations &&
        res.data.updated_recommendations.length
      ) {
        onRecommendationsUpdate(res.data.updated_recommendations);
      } else {
        // fallback: fetch refreshed recommendations
        const refresh = await getUpdatedRecommendations(userId);
        onRecommendationsUpdate(refresh.data);
      }
    } catch (err) {
      console.error("❌ Error recording action:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="movie-card">
      <img src={movie.poster_url} alt={movie.title} />
      <h3 className="movie-title">{movie.title}</h3>

      {/* Genres rendered as pill chips (up to 3) */}
      <div className="genre-row">
        {String(movie.genres || "")
          .split(/[|,]/)
          .map((g) => g.trim())
          .filter(Boolean)
          .slice(0, 3)
          .map((g) => (
            <span key={g} className="genre-chip">{g}</span>
          ))}
      </div>
      {showActions && (
        <div className="card-actions">
          <button className="action-btn" onClick={() => handleAction("clicked")} disabled={loading} style={buttonStyle("#9aa3b2")}>
            👆 Clicked
          </button>
          <button className="action-btn" onClick={() => handleAction("added_to_list")} disabled={loading} style={buttonStyle("#6c5ce7")}>
            📝 Add to List
          </button>
          <button className="action-btn" onClick={() => handleAction("watched")} disabled={loading} style={buttonStyle("#00b894")}>
            🎥 Watched
          </button>
          <button className="action-btn" onClick={() => handleAction("liked")} disabled={loading} style={buttonStyle("#e17055")}>
            ❤️ Liked
          </button>
        </div>
      )}
    </div>
  );
}

// 🎨 Button styling helper
function buttonStyle(color) {
  return {
    backgroundColor: color,
    color: "#fff",
    border: "none",
    borderRadius: "5px",
    padding: "6px 8px",
    cursor: "pointer",
    fontSize: "12px",
    transition: "background 0.2s",
  };
}
