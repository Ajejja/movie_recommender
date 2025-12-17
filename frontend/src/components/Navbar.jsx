import React, { useState, useEffect, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import { searchMovie } from "../api/api";

export default function Navbar() {
  const [query, setQuery] = useState("");
  const navigate = useNavigate();
  const navRef = useRef(null);

  useEffect(() => {
    const el = navRef.current;
    if (!el) return;
    const h = el.offsetHeight;
    document.body.style.paddingTop = h + "px";
    return () => {
      document.body.style.paddingTop = "";
    };
  }, []);

  const submit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    // Navigate to a search results page which will handle listing matches
    navigate(`/search?query=${encodeURIComponent(query.trim())}`);
    setQuery("");
  };

  return (
    <nav
      ref={navRef}
      className="navbar"
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 1000,
        backdropFilter: "blur(6px)",
        background: "rgba(20,20,20,0.85)",
        borderBottom: "1px solid rgba(255,255,255,0.1)",
      }}
    >
      <div className="brand">
        <Link to="/home" className="brand-link">
          <span className="logo">🎬</span>
          <span className="title">Movie Recommender</span>
        </Link>
      </div>

      <form onSubmit={submit} style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <input
          aria-label="Search movies"
          placeholder="Search movies..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ padding: "8px 10px", borderRadius: 8, border: "none", width: 260 }}
        />
        <button className="primary-btn" type="submit" style={{ padding: "8px 12px" }}>
          Search
        </button>
      </form>

      <div className="nav-links">
        <Link to="/home">Home</Link>
        <Link to="/watchlist">Watchlist</Link>
        <Link to="/logout" className="logout-btn">Logout</Link>
      </div>
    </nav>
  );
}
