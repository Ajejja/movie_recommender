import React, { useEffect, useState } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { searchMovie } from "../api/api";
import MovieCard from "../components/MovieCard";

export default function SearchResults() {
  const [params] = useSearchParams();
  const q = params.get("query") || "";
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    if (!q) return;
    setLoading(true);
    searchMovie(q)
      .then((res) => {
        const data = res.data || [];
        setResults(Array.isArray(data) ? data : [data]);
      })
      .catch(() => setResults([]))
      .finally(() => setLoading(false));
  }, [q]);

  return (
    <div className="container">
      <h1 style={{ marginTop: 8 }}>Search results for "{q}"</h1>

      {loading ? (
        <p>Loading...</p>
      ) : results.length === 0 ? (
        <p>No movies found.</p>
      ) : (
        <div className="movie-grid" style={{ justifyContent: "center" }}>
          {results.map((m) => (
            <div key={m.movieId} style={{ cursor: "pointer" }} onClick={() => navigate(`/movie/${m.movieId}`)}>
              <MovieCard movie={m} onRecommendationsUpdate={() => {}} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
