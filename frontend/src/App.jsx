import React, { useEffect, useState } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Login from "./components/Login";
import Signup from "./components/Signup";
import Home from "./pages/Home";
import Watchlist from "./pages/Watchlist";
import MovieDetails from "./pages/MovieDetails";
import SearchResults from "./pages/SearchResults";
import Navbar from "./components/Navbar";
import Logout from "./components/Logout";
import MovieAgent from "./components/MovieAgent";




export default function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(!!localStorage.getItem("userId"));

  useEffect(() => {
    const onLogin = () => setIsLoggedIn(true);
    const onLogout = () => setIsLoggedIn(false);

    window.addEventListener("userLogin", onLogin);
    window.addEventListener("userLogout", onLogout);

    return () => {
      window.removeEventListener("userLogin", onLogin);
      window.removeEventListener("userLogout", onLogout);
    };
  }, []);

  return (
    <Router>
      {isLoggedIn && <Navbar />}
      <Routes>
        <Route path="/" element={isLoggedIn ? <Navigate to="/home" /> : <Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/login" element={<Login />} />
        <Route
          path="/home"
          element={isLoggedIn ? <Home /> : <Navigate to="/login" />}
        />
        <Route
          path="/watchlist"
          element={isLoggedIn ? <Watchlist /> : <Navigate to="/login" />}
        />
        <Route
          path="/movie/:id"
          element={isLoggedIn ? <MovieDetails /> : <Navigate to="/login" />}
        />
        <Route
          path="/search"
          element={isLoggedIn ? <SearchResults /> : <Navigate to="/login" />}
        />
        <Route path="/logout" element={<Logout />} />
        <Route path="/agent" element={isLoggedIn ? <MovieAgent /> : <Navigate to="/login" />} />

        {/* Catch-all route */}
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </Router>
  );
}
