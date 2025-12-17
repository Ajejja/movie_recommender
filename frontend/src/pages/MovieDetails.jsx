import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import MovieCard from "../components/MovieCard";
import { getMovieById } from "../api/api";

export default function MovieDetails() {
  const { id } = useParams();
  const [movie, setMovie] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!id) return;
    getMovieById(id)
      .then((res) => setMovie(res.data))
      .catch(() => setError("Movie not found"));
  }, [id]);

  if (error) {
    return (
      <div className="container">
        <h2>{error}</h2>
      </div>
    );
  }

  if (!movie) {
    return (
      <div className="container">
        <p>Loading...</p>
      </div>
    );
  }

  return (
    <div className="container">
      <MovieCard movie={movie} onRecommendationsUpdate={() => {}} />
    </div>
  );
}
