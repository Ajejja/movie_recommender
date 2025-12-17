import React, { useState } from "react";
import { signup } from "../api/api"; // ✅ use API helper instead of raw axios
import { useNavigate } from "react-router-dom";

const genresList = [
  "Action",
  "Comedy",
  "Drama",
  "Thriller",
  "Animation",
  "Romance",
  "Horror",
  "Sci-Fi",
];

export default function Signup() {
  const [form, setForm] = useState({ username: "", password: "" });
  const [selectedGenres, setSelectedGenres] = useState([]);
  const navigate = useNavigate();

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleGenreChange = (genre) => {
    setSelectedGenres((prev) =>
      prev.includes(genre) ? prev.filter((g) => g !== genre) : [...prev, genre]
    );
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    try {
      const res = await signup({
        username: form.username,
        password: form.password,
        favorite_genres: selectedGenres,
      });

      localStorage.setItem("userId", res.data.userId);
      navigate("/home");
    } catch (err) {
      console.error(err);
      alert("Signup failed. Please try again.");
    }
  };

  return (
    <div className="auth-shell">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-logo">🎬</div>
          <div>
            <h1>Create an Account</h1>
            <p className="muted">Join and get personalized recommendations</p>
          </div>
        </div>

        <form className="auth-form" onSubmit={handleSignup}>
          <input
            name="username"
            placeholder="Username"
            value={form.username}
            onChange={handleChange}
            required
          />

          <input
            type="password"
            name="password"
            placeholder="Password"
            value={form.password}
            onChange={handleChange}
            required
          />

          <h3 style={{ marginTop: "8px" }}>Select Your Favorite Genres</h3>
          <div className="genres">
            {genresList.map((genre) => (
              <label key={genre} className="genre-item">
                <input
                  type="checkbox"
                  checked={selectedGenres.includes(genre)}
                  onChange={() => handleGenreChange(genre)}
                />
                <span>{genre}</span>
              </label>
            ))}
          </div>

          <button className="primary-btn" type="submit">Sign Up</button>
        </form>
      </div>
    </div>
  );
}
