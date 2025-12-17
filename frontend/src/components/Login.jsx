import React, { useState } from "react";
import { login } from "../api/api";
import { useNavigate, Link } from "react-router-dom";

export default function Login() {
  const [form, setForm] = useState({ username: "", password: "" });
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await login(form);
      localStorage.setItem("userId", res.data.userId);
      // Notify the app that a user has logged in so routes/navbar update immediately
      window.dispatchEvent(new Event("userLogin"));
      navigate("/home");
    } catch (err) {
      setError(err.response?.data?.error || "Login failed");
    }
  };

  return (
    <div className="auth-shell">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-logo">🎬</div>
          <div>
            <h1>Welcome back</h1>
            <p className="muted">Sign in to continue to Movie Recommender</p>
          </div>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label className="sr-only" htmlFor="username">Username</label>
          <input
            id="username"
            name="username"
            placeholder="Username"
            onChange={handleChange}
            required
            autoComplete="username"
          />

          <label className="sr-only" htmlFor="password">Password</label>
          <input
            id="password"
            name="password"
            type="password"
            placeholder="Password"
            onChange={handleChange}
            required
            autoComplete="current-password"
          />

          <button className="primary-btn" type="submit">Login</button>
        </form>

        {error && <p className="error">{error}</p>}

        <div className="auth-footer">
          <p>
            Don’t have an account? <Link to="/signup">Signup</Link>
          </p>
        </div>
      </div>
    </div>
  );
}
