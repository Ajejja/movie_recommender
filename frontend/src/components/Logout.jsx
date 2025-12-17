import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";

export default function Logout() {
  const navigate = useNavigate();

  useEffect(() => {
    // Clear session immediately
    localStorage.removeItem("userId");

    // Notify app that user logged out so Navbar/routes update instantly
    window.dispatchEvent(new Event("userLogout"));

    // Redirect to login after a short pause so user can see message
    const t = setTimeout(() => {
      navigate("/login");
    }, 800);

    return () => clearTimeout(t);
  }, [navigate]);

  return (
    <div className="container" style={{ paddingTop: "40px" }}>
      <h2 style={{ color: "#ffd166" }}>You are logged out</h2>
      <p style={{ color: "#cbd5e1" }}>Redirecting to the login page...</p>
    </div>
  );
}
