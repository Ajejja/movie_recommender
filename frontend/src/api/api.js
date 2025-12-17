import axios from "axios";

// ✅ Backend runs on 5001
const API_BASE = "http://127.0.0.1:5001";

// AUTH
export const signup = (data) => axios.post(`${API_BASE}/signup`, data);
export const login = (data) => axios.post(`${API_BASE}/login`, data);

// RECOMMENDATIONS
export const getRecommendations = (userId) =>
  axios.get(`${API_BASE}/recommend/${userId}`);

export const recordAction = (data) =>
  axios.post(`${API_BASE}/action`, data);

export const getUpdatedRecommendations = (userId) =>
  axios.get(`${API_BASE}/recommend/update/${userId}`);

// MOVIES
export const searchMovie = (query) =>
  axios.get(`${API_BASE}/movie/search`, { params: { query } });

export const getMovieById = (id) =>
  axios.get(`${API_BASE}/movie/${id}`);
