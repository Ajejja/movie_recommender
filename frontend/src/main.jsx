import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

// NEW: set a global API base URL (works in Vite and static hosting)
(() => {
  const viteEnv = (typeof import.meta !== "undefined" && import.meta.env) || {};
  const initial =
    viteEnv.VITE_API_URL ||
    viteEnv.PUBLIC_API_URL ||
    (typeof window !== "undefined" && window.__API_BASE__) ||
    "http://127.0.0.1:5000";
  if (typeof window !== "undefined") window.__API_BASE__ = initial;

  // NEW: detect a reachable backend (5000/5001/5002) and update __API_BASE__
  const candidates = Array.from(
    new Set([
      initial,
      "http://127.0.0.1:5000",
      "http://127.0.0.1:5001",
      "http://127.0.0.1:5002",
      "http://localhost:5000",
      "http://localhost:5001",
      "http://localhost:5002",
    ])
  );

  const testBase = async (base) => {
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 900);
      const r = await fetch(`${base}/`, { signal: ctrl.signal, mode: "cors" });
      clearTimeout(t);
      return r.ok;
    } catch {
      return false;
    }
  };

  (async () => {
    for (const base of candidates) {
      if (await testBase(base)) {
        window.__API_BASE__ = base;
        break;
      }
    }
  })();

  // NEW: rewrite hardcoded 5000 calls (Axios uses XHR)
  const origOpen = XMLHttpRequest.prototype.open;
  const rewrite = (url) => {
    try {
      const api = window.__API_BASE__ || initial;
      return url
        .replace(/^http:\/\/127\.0\.0\.1:5000/,'$&' === url ? api : url.startsWith('http://127.0.0.1:5000') ? url.replace('http://127.0.0.1:5000', api) : url)
        .replace(/^http:\/\/localhost:5000/,'$&' === url ? api : url.startsWith('http://localhost:5000') ? url.replace('http://localhost:5000', api) : url);
    } catch {
      return url;
    }
  };
  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    const patched = typeof url === "string" ? rewrite(url) : url;
    return origOpen.call(this, method, patched, ...rest);
  };
})();

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
