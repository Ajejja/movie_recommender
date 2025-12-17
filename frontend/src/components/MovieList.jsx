import React from "react";
import MovieCard from "./MovieCard";

export default function MovieList({ movies }) {
  return (
    <div className="movie-grid">
      {movies.length > 0 ? (
        movies.map((movie) => <MovieCard key={movie.movieId} movie={movie} />)
      ) : (
        <p>No movies yet. Try interacting with some!</p>
      )}
    </div>
  );
}
