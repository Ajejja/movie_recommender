import React, { useState } from "react";
import "./agent.css";

export default function MovieAgent() {
  const [open, setOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [minimized, setMinimized] = useState(false);

  const API_BASE =
    (typeof import.meta !== "undefined" &&
      import.meta.env &&
      // Prefer Vite env (configure VITE_API_URL at build/deploy)
      (import.meta.env.VITE_API_URL || import.meta.env.PUBLIC_API_URL)) ||
    // Optional: allow overriding via a global for static hosting
    (typeof window !== "undefined" && window.__API_BASE__) ||
    // Fallback to local dev
    "http://127.0.0.1:5001";

  async function sendMessage() {
    if (!message.trim()) return;
    setLoading(true);
    setResponse(null);

    try {
      const res = await fetch(`${API_BASE}/agent/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: message,
          top_k: 5,
        }),
      });

      const data = await res.json();
      setResponse(data);
    } catch (err) {
      console.error(err);
      setResponse({ error: "Something went wrong" });
    }

    setLoading(false);
  }

  return (
    <>
      {/* Inline UI styling overrides */}
      <style>{`
        .agent-window {
          width: 360px;
          height: 520px;
          position: fixed;
          bottom: 90px;
          right: 20px;
          z-index: 9999;
          background: #ffffff !important;
          color: #1a1a1a !important;
          border-radius: 12px;
          box-shadow: 0 4px 24px rgba(0,0,0,0.12);
          display: flex;
          flex-direction: column;
        }
        .agent-header {
          padding: 12px;
          font-size: 17px;
          background:#222 !important;
          color:#fff !important;
          border-radius: 12px 12px 0 0;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .agent-body {
          padding: 14px;
          overflow-y: auto;
        }
        .agent-input {
          width: 100%;
          height: 70px;
          border-radius: 8px;
          padding: 10px;
          margin-bottom: 10px;
          background:#fdfdfd !important;
          border:1px solid #cfd3d8;
        }
        .agent-send-btn {
          width: 100%;
          padding: 10px;
          border-radius: 8px;
          background:#6246ea !important;
          color:#fff !important;
          font-weight: bold;
          cursor: pointer;
        }
        
        .result-box {
          background:#f5f7fa !important;
          margin-top: 15px;
          padding: 12px;
          border-radius: 10px;
          border:1px solid #e0e4e8;
        }
        .similar-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 10px;
          margin-top: 10px;
        }
        .similar-card img {
          width: 100%;
          height: 110px;
          object-fit: cover;
          border-radius: 6px;
        }
          .mac-btn {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  cursor: pointer;
}

.mac-red { background: #ff5f56; }
.mac-yellow { background: #ffbd2e; }


      `}</style>
      {/* MINIMIZED BUBBLE */}
{open && minimized && (
  <button
    className="agent-btn"
    style={{
      color: "#fff",
      boxShadow: "0 4px 12px rgba(98,70,234,0.4)"
    }}
    onClick={() => setMinimized(false)}
  >
    🤖
  </button>
)}



      {/* Floating open button */}
      {!open && (
        <button className="agent-btn" onClick={() => setOpen(true)}>
          🤖
        </button>
      )}

      {/* CHAT WINDOW */}
      {open && !minimized && (
        <div className="agent-window">
          <div className="agent-header">
  <span className="agent-title">Movie AI Agent</span>

  <div style={{ marginLeft: "auto", display: "flex", gap: "6px" }}>
    <div
      className="mac-btn mac-yellow"
      onClick={() => setMinimized(true)}
    ></div>

    <div
      className="mac-btn mac-red"
      onClick={() => setOpen(false)}
    ></div>
  </div>
</div>


          <div className="agent-body">
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Describe the movie you want..."
              className="agent-input"
            />

            <button onClick={sendMessage} disabled={loading} className="agent-send-btn">
              {loading ? "Searching..." : "Ask Agent"}
            </button>

            {/* RESULTS */}
            {response && (
              <div className="result-box">

                {response.error && <p className="error">{response.error}</p>}

                {response.identified_movie && (
                  <div style={{ textAlign: "center", marginBottom: "15px" }}>
                    <h3 style={{ fontSize: "1.3rem", fontWeight: "bold", marginBottom: "10px" }}>
                      🎯 {response.identified_movie}
                    </h3>

                    {/* POSTER IMAGE — CENTERED */}
                    {response.dataset_match?.poster_url && (
                      <img
                        src={response.dataset_match.poster_url}
                        alt={response.identified_movie}
                        style={{
                          width: "160px",
                          borderRadius: "10px",
                          margin: "0 auto 15px auto",
                          display: "block",
                          boxShadow: "0 4px 12px rgba(0,0,0,0.18)",
                        }}
                      />
                    )}
                  </div>
                )}

                {/* SUMMARY */}
                {response.summary && <p>📝 {response.summary}</p>}

                {/* TRAILER */}
                {response.trailer_link && (
                  <a className="trailer-link" href={response.trailer_link} target="_blank">
                    ▶ Watch Trailer
                  </a>
                )}

                {/* SIMILAR MOVIES */}
                {response.similar_from_dataset?.length > 0 && (
                  <>
                    <h4 style={{ marginTop: "10px" }}>🔥 Similar movies:</h4>
                    <div className="similar-grid">
                      {response.similar_from_dataset.map((m) => (
                        <div key={m.movieId} className="similar-card">
                          <img src={m.poster_url} alt="" />
                          <p>{m.title}</p>
                        </div>
                      ))}
                    </div>
                  </>
                )}

              </div>
            )}

          </div>
        </div>
      )}
    </>
  );
}
