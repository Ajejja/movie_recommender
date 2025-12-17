from flask import Flask, request, jsonify
import pandas as pd
import os
from datetime import datetime
import pickle
# REPLACED: guarded import so app doesn't crash if surprise isn't installed
try:
    from surprise import SVD, Dataset, Reader, accuracy
    SURPRISE_AVAILABLE = True
except Exception:
    SVD = Dataset = Reader = accuracy = None
    SURPRISE_AVAILABLE = False
from threading import Thread
import hashlib
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
import socket
# Load env from project root and backend dir (no-op if files absent)
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# NEW: optional OpenAI client (safe if not installed or no key)
try:
    from openai import OpenAI  # OpenAI Python SDK v1
    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    OPENAI_AVAILABLE = False

app = Flask(__name__)
# CHANGED: wide-open CORS for dev to avoid preflight failures
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
})

# NEW/CHANGED: global preflight handler with explicit headers
@app.before_request
def _cors_preflight():
    if request.method == "OPTIONS":
        resp = app.make_response(("", 204))
        origin = request.headers.get("Origin") or "*"
        req_headers = request.headers.get("Access-Control-Request-Headers") or "Content-Type, Authorization"
        req_method = request.headers.get("Access-Control-Request-Method") or "GET, POST, PUT, DELETE, OPTIONS"
        resp.headers["Access-Control-Allow-Origin"] = origin if origin != "null" else "*"
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Headers"] = req_headers
        resp.headers["Access-Control-Allow-Methods"] = req_method
        resp.headers["Access-Control-Max-Age"] = "86400"
        return resp

# CHANGED: always append CORS headers on normal responses
@app.after_request
def _cors_headers(resp):
    origin = request.headers.get("Origin")
    resp.headers["Access-Control-Allow-Origin"] = origin if origin else "*"
    resp.headers.setdefault("Vary", "Origin")
    resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
    return resp

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "user_interactions.csv")
MOVIE_PATH = os.path.join(BASE_DIR, "data", "movie.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "recommender.pkl")
USER_PATH = os.path.join(BASE_DIR, "data", "users.csv")
MAX_MOVIE_LIMIT = 100  # upper cap for /movies responses

# NEW: OpenAI config from env (safe defaults)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_AVAILABLE and OPENAI_API_KEY) else None

# --- Utility ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def safe_read_csv(path, columns=None):
    """Read a CSV file safely and return an empty DataFrame if missing."""
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"⚠️ Error reading {path}: {e}")
    return pd.DataFrame(columns=columns or [])

# NEW: normalize and parse genres reliably (handles |, comma, slash, spaces)
def parse_genres(genres_str):
    if pd.isna(genres_str):
        return set()
    s = str(genres_str).lower()
    for sep in ['|', ',', '/', ';']:
        s = s.replace(sep, ' ')
    parts = [p.strip() for p in s.split() if p.strip() and p.strip() != '(no' and p.strip() != 'genres)']
    return set(parts)

# NEW: recency weighting (half-life in days)
def recency_weight(ts, now=None, half_life_days=14):
    now = now or pd.Timestamp.utcnow()
    try:
        dt = pd.to_datetime(ts, errors="coerce")
        if pd.isna(dt):
            return 1.0
        days = max((now - dt).total_seconds() / 86400.0, 0.0)
        return 0.5 ** (days / half_life_days)
    except Exception:
        return 1.0


# Helper: recent vs overall genre trend (insert after recency_weight)
def genre_trend(organic_hist_df, movies_df, recent_window=25):
    if organic_hist_df.empty:
        return {}, {}
    sub = organic_hist_df.copy()
    sub["ts"] = pd.to_datetime(sub["timestamp"], errors="coerce")
    sub = sub.sort_values("ts", ascending=False)
    recent = sub.head(recent_window)
    def count(gen_df):
        counts = {}
        merged = gen_df.merge(movies_df[["movieId", "genre_set"]], on="movieId", how="left")
        for _, r in merged.iterrows():
            for g in r.get("genre_set", set()):
                counts[g] = counts.get(g, 0) + 1
        total = sum(counts.values()) or 1
        return {g: c / total for g, c in counts.items()}
    return count(recent), count(sub)

# ===== AI Agent utilities (NEW) =====

def _norm(s):
    return str(s or "").strip().lower()

def _tokenize(text):
    t = _norm(text)
    for sep in ["|", "/", ",", ";", ":", ".", "(", ")", "-", "_"]:
        t = t.replace(sep, " ")
    return [w for w in t.split() if w]

# Lightweight synonyms and phrase hints
GENRE_ALIASES = {
    "scifi": "science fiction",
    "sci-fi": "science fiction",
    "sf": "science fiction",
    "ww2": "world war ii",
    "wwii": "world war ii",
    "u.s.": "usa",
    "us": "usa",
    "u.s": "usa",
    "american": "usa",
}
KNOWN_GENRES = {
    "action","adventure","animation","children","comedy","crime","documentary","drama",
    "fantasy","film-noir","horror","musical","mystery","romance","sci-fi","science fiction",
    "thriller","war","western","history","biography","family","sport"
}

def _apply_aliases(tokens):
    out = []
    for t in tokens:
        out.append(GENRE_ALIASES.get(t, t))
    return out
def fetch_movie_extra_info(title):
    """
    Ask OpenAI to fetch:
      - short summary
      - YouTube trailer link
    """
    if not openai_client:
        return {"summary": None, "trailer_url": None}

    prompt = f"""
    Fetch me REAL information about the movie titled "{title}".
    Return a JSON object with exactly these keys:
      - summary: A short 3-4 sentence summary of the movie.
      - trailer_url: A YouTube trailer link.
    If unsure, pick the most popular official trailer.
    """

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a movie data assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        import json
        data = json.loads(response.choices[0].message.content)
        return {
            "summary": data.get("summary"),
            "trailer_url": data.get("trailer_url")
        }

    except Exception as e:
        print(f"❌ Error fetching extra info: {e}")
        return {"summary": None, "trailer_url": None}

def fallback_extract_tags(message):
    """
    Heuristic extractor used when OpenAI is unavailable.
    Returns dict with genres/themes/topics/countries/eras.
    """
    text = _norm(message)
    words = set(_apply_aliases(_tokenize(text)))

    # simple phrase detection
    phrases = []
    if "world" in words and "war" in words and ("ii" in words or "2" in words or "ii" in text or "wwii" in text or "ww2" in text):
        phrases.append("world war ii")

    genres = set()
    for g in KNOWN_GENRES:
        if g in words:
            genres.add("science fiction" if g in {"sci-fi", "science fiction"} else g)
    # map mood words to probable genres
    if "realistic" in words:
        genres.add("drama")

    countries = set()
    if any(w in words for w in ["usa", "america", "american", "u.s.", "us", "u.s"]):
        countries.add("usa")

    topics = set()
    if "war" in words:
        topics.add("war")
    for ph in phrases:
        topics.add(ph)
    if "biopic" in words or "biography" in words:
        topics.add("biography")

    themes = set()
    if "history" in words or "historical" in words:
        themes.add("history")

    eras = set()
    if "world war ii" in topics:
        eras.add("1939-1945")

    return {
        "genres": sorted(genres),
        "themes": sorted(themes),
        "topics": sorted(topics),
        "countries": sorted(countries),
        "eras": sorted(eras),
        "keywords": sorted(list(words))[:20]
    }

def extract_tags_with_llm(message):
    """
    Use OpenAI to extract structured search tags. Falls back to heuristics on error.
    """
    if not openai_client:
        return fallback_extract_tags(message)
    try:
        sys = (
            "You extract movie-search tags in JSON. "
            "Return keys: genres (list), themes (list), topics (list), countries (list), eras (list), keywords (list). "
            "Prefer standard genres like Action, Drama, War, Horror, Romance, Thriller, Science Fiction."
        )
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            temperature=0.2,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": message}
            ],
        )
        content = resp.choices[0].message.content
        data = {}
        try:
            import json
            data = json.loads(content or "{}")
        except Exception:
            data = {}
        # normalize lists
        def _get(key):
            v = data.get(key, [])
            if not isinstance(v, list): return []
            return sorted({_norm(x) for x in v if str(x).strip()})
        result = {
            "genres": _get("genres"),
            "themes": _get("themes"),
            "topics": _get("topics"),
            "countries": _get("countries"),
            "eras": _get("eras"),
            "keywords": _get("keywords"),
        }
        # small cleanup: map sci-fi variants
        result["genres"] = sorted({"science fiction" if g in {"sci-fi","science fiction"} else g for g in result["genres"]})
        return result
    except Exception:
        return fallback_extract_tags(message)

def _ensure_index_columns(movies_df):
    """
    Ensure helper columns exist for searching.
    - genre_set (already used elsewhere)
    - text_blob: lowercased concatenation of title/genres/overview if present
    """
    if "genre_set" not in movies_df.columns:
        movies_df["genres"] = movies_df.get("genres", "").fillna("")
        movies_df["genre_set"] = movies_df["genres"].apply(parse_genres)
    # build a text blob (title + genres + overview/summary if present)
    overview_col = "overview" if "overview" in movies_df.columns else ("summary" if "summary" in movies_df.columns else None)
    def mk_blob(row):
        parts = [str(row.get("title", "")), " ".join(sorted(row.get("genre_set", set())))]
        if overview_col:
            parts.append(str(row.get(overview_col, "")))
        return _norm(" ".join(parts))
    if "text_blob" not in movies_df.columns:
        movies_df["text_blob"] = movies_df.apply(mk_blob, axis=1)
    return movies_df

def _popularity_scores():
    """
    Simple popularity prior from interactions. Returns dict movieId->score.
    """
    inter = safe_read_csv(DATA_PATH, ["userId", "movieId", "action", "timestamp"])
    if inter.empty:
        return {}
    aw = {"seed_like": 1.0, "clicked": 2.0, "added_to_list": 3.0, "watched": 4.0, "liked": 5.0}
    inter["w"] = inter["action"].map(aw).fillna(0)
    try:
        inter["movieId"] = inter["movieId"].astype(int)
    except Exception:
        pass
    return inter.groupby("movieId")["w"].sum().to_dict()

def search_movies_by_tags(movies_df, tags, top_k=5):
    """
    Score movies by:
    - genre overlap
    - keyword/phrase hits in title/overview blob
    - small popularity prior
    """
    movies_df = _ensure_index_columns(movies_df.copy())
    pop = _popularity_scores()

    want_genres = set(tags.get("genres", []))
    want_topics = set(tags.get("topics", []))
    want_keywords = set(tags.get("keywords", []))
    want_countries = set(tags.get("countries", []))
    want_phrases = set()
    if "world war ii" in want_topics:
        want_phrases.add("world war ii")
        want_phrases.add("wwii")
        want_phrases.add("ww2")

    scored = []
    for _, r in movies_df.iterrows():
        mid = int(r.get("movieId"))
        gs = r.get("genre_set", set())
        blob = r.get("text_blob", "")

        # genre score
        genre_overlap = len(gs & want_genres)
        score = 2.5 * genre_overlap

        # phrase/topic hits
        for ph in want_phrases:
            if ph in blob:
                score += 2.0

        # keyword hits (cap to avoid overcount)
        kw_hits = sum(1 for kw in list(want_keywords)[:8] if kw in blob)
        score += min(kw_hits, 4) * 0.6

        # country hints
        if "usa" in want_countries and any(tok in blob for tok in ["usa", "america", "american", "united states"]):
            score += 1.2

        # small popularity prior
        score += 0.002 * pop.get(mid, 0.0)

        scored.append((mid, float(score)))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [mid for mid, s in scored[:max(1, top_k)] if s > 0] or [mid for mid, _ in scored[:max(1, top_k)]]

    # Prepare payload fields
    cols = ["movieId", "title", "genres", "poster_url"]
    if "overview" in movies_df.columns:
        cols.append("overview")
    elif "summary" in movies_df.columns:
        cols.append("summary")
    if "trailer_url" in movies_df.columns:
        cols.append("trailer_url")

    out = movies_df[movies_df["movieId"].isin(top_ids)][cols].copy()
    # stable order by score
    score_map = dict(scored)
    out["__score"] = out["movieId"].apply(lambda x: score_map.get(int(x), 0.0))
    out = out.sort_values("__score", ascending=False).drop(columns="__score", errors="ignore")
    return out.to_dict(orient="records")

def build_agent_reply(tags, results, user_first_name=None):
    name = user_first_name or ""
    prefix = f"Got it{', ' + name if name else ''}! Based on your mood, here are {min(len(results), 3)} movie(s) you may enjoy:"
    return prefix

# --- SIGNUP ---
# CHANGED: allow OPTIONS explicitly
@app.route("/signup", methods=["POST", "OPTIONS"])
@cross_origin(origins=["*"], methods=["POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])
def signup():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.json
    username = data.get("username")
    password = data.get("password")
    favorite_genres = data.get("favorite_genres", [])

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    os.makedirs(os.path.dirname(USER_PATH), exist_ok=True)
    users = safe_read_csv(USER_PATH, ["userId", "username", "password", "favorite_genres"])

    # Prevent duplicate usernames
    if username in users["username"].values:
        return jsonify({"error": "Username already exists"}), 400

    user_id = len(users) + 1
    fav_str = ",".join(favorite_genres)
    new_user = pd.DataFrame(
        [[user_id, username, hash_password(password), fav_str]],
        columns=["userId", "username", "password", "favorite_genres"]
    )

    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_PATH, index=False)

    # --- Add initial liked movies from favorite genres ---
    if os.path.exists(MOVIE_PATH):
        movies = pd.read_csv(MOVIE_PATH)
        liked_rows = []
        for genre in favorite_genres:
            genre_movies = movies[movies["genres"].str.contains(genre, case=False, na=False)].head(5)
            for mid in genre_movies["movieId"]:
                # CHANGED: use 'seed_like' instead of 'liked'
                liked_rows.append([user_id, mid, "seed_like", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

        if liked_rows:
            df = safe_read_csv(DATA_PATH, ["userId", "movieId", "action", "timestamp"])
            new_df = pd.DataFrame(liked_rows, columns=["userId", "movieId", "action", "timestamp"])
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(DATA_PATH, index=False)

            # Retrain asynchronously
            Thread(target=retrain_model).start()

    return jsonify({"message": "User created", "userId": user_id}), 201


# --- LOGIN ---
# CHANGED: allow OPTIONS explicitly
@app.route("/login", methods=["POST", "OPTIONS"])
@cross_origin(origins=["*"], methods=["POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])
def login():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    users = safe_read_csv(USER_PATH, ["userId", "username", "password", "favorite_genres"])
    hashed = hash_password(password)
    user = users[(users["username"] == username) & (users["password"] == hashed)]

    if user.empty:
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify({"message": "Login successful", "userId": int(user.iloc[0]['userId'])}), 200


# --- RETRAINING LOGIC ---
def retrain_model():
    print("🧠 Retraining model...")

    # --- Load and clean interaction data ---
    df = safe_read_csv(DATA_PATH, ["userId", "movieId", "action", "timestamp"])
    if df.empty:
        print("⚠️ No user interaction data found.")
        return

    df = df.drop_duplicates(subset=["userId", "movieId", "action"])

    action_weights = {
        "seed_like": 1.0,
        "clicked": 2.0,
        "added_to_list": 3.0,
        "watched": 4.0,
        "liked": 5.0
    }
    df["rating"] = df["action"].map(action_weights)
    df = df.dropna(subset=["rating"])

    # If Surprise not available, save a simple popularity model
    if not SURPRISE_AVAILABLE:
        print("ℹ️ surprise not installed. Using popularity fallback model.")
        pop = df.groupby("movieId")["rating"].sum().to_dict()
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"type": "popularity", "scores": pop}, f)
        log_path = os.path.join(BASE_DIR, "training_log.txt")
        with open(log_path, "a") as f:
            f.write(f"{datetime.now()} - RMSE: n/a (popularity) - Interactions: {len(df)}\n")
        print("✅ Popularity model saved.")
        return

    # --- Train-test split for evaluation (Surprise path) ---
    reader = Reader(rating_scale=(1, 5))
    try:
        data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
    except Exception as e:
        print(f"⚠️ Error preparing dataset: {e}")
        return

    from surprise.model_selection import train_test_split
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD()
    model.fit(trainset)

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    print(f"✅ Model retrained successfully. RMSE: {rmse:.4f} (lower = better)")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        # Save raw SVD model when available
        pickle.dump(model, f)

    log_path = os.path.join(BASE_DIR, "training_log.txt")
    with open(log_path, "a") as f:
        f.write(f"{datetime.now()} - RMSE: {rmse:.4f} - Interactions: {len(df)}\n")
    print(f"📈 Logged training result → {log_path}")

    


# --- RECORD USER ACTION ---
@app.route("/action", methods=["POST"])
def record_action():
    data = request.json
    user_id = data.get("userId")
    movie_id = data.get("movieId")
    action = data.get("action", "liked")

    # NEW: ensure ids are ints so recommend() sees the user correctly
    try:
        user_id = int(user_id)
        movie_id = int(movie_id)
    except (TypeError, ValueError):
        return jsonify({"error": "userId and movieId must be integers"}), 400

    if user_id is None or movie_id is None:
        return jsonify({"error": "Missing fields (userId, movieId)"}), 400

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # keep stable schema even if file is missing/empty
    df = safe_read_csv(DATA_PATH, ["userId", "movieId", "action", "timestamp"])
    new_row = pd.DataFrame([[user_id, movie_id, action, timestamp]],
                           columns=["userId", "movieId", "action", "timestamp"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    # Retrain synchronously to ensure immediate model update
    retrain_model()

    # Return updated recommendations
    recs = recommend(user_id, force_refresh=True).get_json()
    return jsonify({
        "message": f"Action '{action}' recorded for user {user_id}",
        "updated_recommendations": recs
    }), 200  # FIXED: proper (response, status) tuple

# --- RECOMMENDATION ENDPOINT ---
@app.route("/recommend/<int:user_id>", methods=["GET"])
def recommend(user_id, force_refresh=False):
    # NEW: coerce user_id to int for internal (non-route) calls
    try:
        user_id = int(user_id)
    except Exception:
        return jsonify([])

    movies = safe_read_csv(MOVIE_PATH, ["movieId", "title", "genres", "poster_url"])
    users = safe_read_csv(USER_PATH)
    # NEW: read with fixed columns so dtype casting never KeyErrors
    interactions = safe_read_csv(DATA_PATH, ["userId", "movieId", "action", "timestamp"])

    if movies.empty:
        return jsonify([])

    # ensure correct data types
    try:
        movies["movieId"] = movies["movieId"].astype(int)
    except Exception:
        pass
    try:
        interactions["movieId"] = interactions["movieId"].astype(int)
        interactions["userId"] = interactions["userId"].astype(int)
    except Exception:
        # if interactions is empty, columns still exist due to safe_read_csv default
        pass

    # build genre sets once
    if "genre_set" not in movies.columns:
        movies["genre_set"] = movies["genres"].apply(parse_genres)

    # derive favorite genres saved on signup (if any)
    fav_genres_set = set()
    if not users.empty and "favorite_genres" in users.columns and (user_id in users.get("userId", pd.Series([], dtype=int)).values):
        try:
            fav_str = users.loc[users["userId"] == user_id, "favorite_genres"].astype(str).iloc[0]
            fav_genres_set = set(g.strip().lower() for g in fav_str.split(",") if g.strip())
        except Exception:
            fav_genres_set = set()

    # Cold start stays: use favorites to seed and exclude Horror unless explicitly preferred
    if interactions.empty or user_id not in interactions["userId"].values:
        fav_list = sorted(list(fav_genres_set))
        if fav_list:
            # Hard filter to favorite genres; exclude Horror if not explicitly preferred
            include_mask = movies["genre_set"].apply(lambda gs: any(g in gs for g in fav_genres_set))
            exclude_horror = "horror" not in fav_genres_set
            if exclude_horror:
                include_mask &= ~movies["genre_set"].apply(lambda gs: "horror" in gs)
            recommended = movies[include_mask].head(12)
        else:
            recommended = movies.sample(n=min(12, len(movies)), random_state=42)
        return jsonify(recommended.drop(columns=["genre_set"], errors="ignore").to_dict(orient="records"))

    # Personalized recommendation
    if not os.path.exists(MODEL_PATH):
        retrain_model()
    with open(MODEL_PATH, "rb") as f:
        saved_model = pickle.load(f)

    # Support both SVD and popularity fallback models
    if isinstance(saved_model, dict) and saved_model.get("type") == "popularity":
        pop_scores = saved_model.get("scores", {})
        def estimate(mid):
            return float(pop_scores.get(int(mid), 0.0))
    else:
        model = saved_model
        def estimate(mid):
            # If surprise is not available and file contains SVD, avoid crash
            if not SURPRISE_AVAILABLE:
                return 0.0
            return model.predict(user_id, int(mid)).est

    # Build a recency-weighted genre profile from interactions (adaptive)
    user_hist = interactions[(interactions["userId"] == user_id)]
    action_weights = {
        "seed_like": 1.0,
        "clicked": 2.0,
        "added_to_list": 3.0,
        "watched": 4.0,
        "liked": 5.0
    }
    user_hist = user_hist[user_hist["action"].isin(action_weights.keys())]
    now_ts = pd.Timestamp.utcnow()

    # Separate organic interactions (exclude seed_like for adaptation logic)
    organic_hist = user_hist[user_hist["action"] != "seed_like"]
    organic_count = len(organic_hist)

    genre_scores = {}
    if not user_hist.empty:
        hist = user_hist.merge(movies[["movieId", "genre_set"]], on="movieId", how="left")
        for _, row in hist.iterrows():
            # Skip seed_like once we have enough organic interactions
            if row["action"] == "seed_like" and organic_count >= 10:
                continue
            w = action_weights.get(row["action"], 0.0) * recency_weight(row.get("timestamp"), now=now_ts, half_life_days=14)
            for g in row.get("genre_set", set()):
                genre_scores[g] = genre_scores.get(g, 0.0) + w

    # Diversity & prior decay
    # Decay favorite prior as organic interactions grow (after 15 interactions it ~0)
    prior_decay = max(0.0, 1.0 - organic_count / 15.0)
    if fav_genres_set and prior_decay > 0:
        for g in fav_genres_set:
            genre_scores[g] = genre_scores.get(g, 0.0) + 0.3 * prior_decay  # CHANGED: decaying prior

    preferred_genres = set(sorted(genre_scores.keys(), key=lambda k: genre_scores[k], reverse=True)[:3]) or set(fav_genres_set)

    # --- Trend detection (NEW) ---
    recent_share, overall_share = genre_trend(organic_hist, movies)
    rising_genres = {g for g in recent_share if recent_share[g] - overall_share.get(g, 0) > 0.08}
    cooling_genres = {g for g in overall_share if overall_share[g] - recent_share.get(g, 0) > 0.10}

    # Dampen oversaturated genre scores (prevents comedy lock-in)
    if overall_share:
        max_genre = max(overall_share, key=overall_share.get)
        max_share = overall_share[max_genre]
        if max_share > 0.40:  # earlier than previous penalty trigger
            genre_scores[max_genre] = genre_scores.get(max_genre, 0.0) * (0.85 - (max_share - 0.40) * 0.5)

    # Small adaptive boost for rising genres before preferred_genres selection
    for g in rising_genres:
        genre_scores[g] = genre_scores.get(g, 0.0) * 1.10

    # Recompute preferred genres after adjustments
    preferred_genres = set(sorted(genre_scores.keys(), key=lambda k: genre_scores[k], reverse=True)[:3]) or set(fav_genres_set)

    # NEW: snapshot the current genre ranking for optional debug output
    genre_ranking = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)

    # Detect specific genre engagement (NEW)
    merged_hist = user_hist.merge(movies[["movieId", "genre_set"]], on="movieId", how="left")
    horror_engagement = 0
    for _, r in merged_hist.iterrows():
        if "horror" in r.get("genre_set", set()) and r.get("action") in {"liked", "watched", "added_to_list", "clicked"}:
            horror_engagement += 1
    has_horror_interest = horror_engagement >= 2  # require at least 2 meaningful interactions

    # Adjust horror ban logic (CHANGED)
    ban_horror = (
        genre_scores.get("horror", 0.0) <= 0.0
        and ("horror" not in fav_genres_set)
        and not has_horror_interest
    )

    seen = interactions[interactions["userId"] == user_id]["movieId"].values
    unseen = movies[~movies["movieId"].isin(seen)].copy()

    # Modify candidate filtering to allow horror if user showed interest (CHANGED)
    candidates = unseen
    small_history = organic_count < 5
    if small_history and preferred_genres:
        candidates = candidates[candidates["genre_set"].apply(
            lambda gs: (len(gs & preferred_genres) > 0) or (has_horror_interest and "horror" in gs)
        )]
    if ban_horror:
        candidates = candidates[~candidates["genre_set"].apply(lambda gs: "horror" in gs)]

    # Fallbacks
    if candidates.empty:
        candidates = unseen if not ban_horror else unseen[~unseen["genre_set"].apply(lambda gs: "horror" in gs)]
    if candidates.empty:
        candidates = unseen

    # Genre saturation penalty: penalize genres dominating organic interactions (>55%)
    genre_counts = {}
    if not organic_hist.empty:
        merged_org = organic_hist.merge(movies[["movieId", "genre_set"]], on="movieId", how="left")
        for _, row in merged_org.iterrows():
            for g in row.get("genre_set", set()):
                genre_counts[g] = genre_counts.get(g, 0) + 1
    total_org = sum(genre_counts.values())

    # Replace saturation_penalty with earlier nonlinear penalty (CHANGED)
    def saturation_penalty(gs):
        if total_org == 0:
            return 0.0
        penalties = []
        for g in gs:
            frac = genre_counts.get(g, 0) / total_org
            threshold = 0.45  # lowered
            excess = max(0.0, frac - threshold)
            penalties.append(excess ** 1.25)
        return sum(penalties) / len(penalties) if penalties else 0.0

    # Score with added saturation penalty
    total_profile = sum(genre_scores.values()) or 1.0
    predictions = []
    for _, row in candidates.iterrows():
        est = estimate(row["movieId"])
        gs = row["genre_set"]

        overlap = len(gs & preferred_genres) if preferred_genres else 0
        gscore = sum(genre_scores.get(g, 0.0) for g in gs) / total_profile
        horror_penalty = 1.0 if ("horror" in gs and ban_horror) else 0.0
        sat_pen = saturation_penalty(gs)

        # Trend & exploration boosts (NEW)
        trend_boost = sum(0.30 * recent_share.get(g, 0) for g in gs if g in rising_genres)
        exploration_boost = sum(0.18 * recent_share.get(g, 0) for g in gs
                                if overall_share.get(g, 0) < 0.12 and recent_share.get(g, 0) > 0)
        cooling_penalty = sum(0.10 * (overall_share.get(g, 0) - recent_share.get(g, 0))
                              for g in gs if g in cooling_genres)

        # Add exploration boost for emerging horror when its share is still low
        horror_explore_boost = 0.0
        if "horror" in gs and has_horror_interest and overall_share.get("horror", 0) < 0.15:
            horror_explore_boost = 0.25 * recent_share.get("horror", 0)

        final_score = (
            est
            + 1.0 * gscore
            + 0.22 * overlap
            + trend_boost
            + exploration_boost
            - cooling_penalty
            - 1.2 * horror_penalty
            - 0.9 * sat_pen
            + horror_explore_boost  # NEW
        )
        predictions.append((int(row["movieId"]), final_score))

    if not predictions:
        for _, row in unseen.iterrows():
            est = estimate(row["movieId"])
            predictions.append((int(row["movieId"]), est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_ids = [mid for mid, _ in predictions[:20]]
    pool = movies[movies["movieId"].isin(top_ids)]
    recommended = pool.sample(n=min(8, len(top_ids)), random_state=None if force_refresh else 42)

    # NEW: Optional debug response with genre ranking. Keeps default list shape when debug is off.
    payload = recommended.drop(columns=["genre_set"], errors="ignore").to_dict(orient="records")
    try:
        debug_mode = request.args.get("debug", type=int) == 1
    except Exception:
        debug_mode = False

    if debug_mode:
        return jsonify({
            "genre_ranking": [{"genre": g, "score": float(round(s, 4))} for g, s in genre_ranking[:10]],
            "preferred_genres": sorted(list(preferred_genres)),
            "rising_genres": sorted(list(rising_genres)),
            "cooling_genres": sorted(list(cooling_genres)),
            "recent_share": {g: float(round(v, 4)) for g, v in recent_share.items()},
            "overall_share": {g: float(round(v, 4)) for g, v in overall_share.items()},
            "recommendations": payload
        })
    return jsonify(payload)

# --- ALL MOVIES ENDPOINT ---
@app.route("/movies", methods=["GET"])
def get_movies():
    movies = safe_read_csv(MOVIE_PATH)
    if movies.empty:
        return jsonify({"error": "No movies found"}), 404

    # NEW: optional genre filter (?genre=comedy or comma/pipe/slash-separated)
    genre_param = request.args.get("genre", default=None, type=str)
    if genre_param:
        # normalize tokens and build per-row genre_set
        if "genre_set" not in movies.columns:
            movies["genre_set"] = movies["genres"].apply(parse_genres)
        tokens = (
            genre_param.replace("|", ",").replace("/", ",")
        ).split(",")
        wanted = {t.strip().lower() for t in tokens if t.strip()}
        if wanted:
            movies = movies[movies["genre_set"].apply(lambda gs: any(g in gs for g in wanted))]

    if movies.empty:
        return jsonify([]), 200

    # Query params
    req_limit = request.args.get("limit", type=int)
    page = request.args.get("page", type=int)  # 1-based
    size = request.args.get("size", type=int)  # page size override
    # Default limit if no pagination
    default_limit = 150
    if req_limit is None:
        req_limit = default_limit
    # Clamp limit
    max_allowed = min(MAX_MOVIE_LIMIT, len(movies))
    req_limit = max(1, min(req_limit, max_allowed))

    # Shuffle every call (no fixed seed) for variety
    shuffled = movies.sample(frac=1.0, replace=False)  # full shuffle

    # Pagination if page & size provided
    if page and size:
        size = max(1, min(size, max_allowed))
        total_pages = (len(shuffled) + size - 1) // size
        page = max(1, min(page, total_pages))
        start = (page - 1) * size
        end = start + size
        page_slice = shuffled.iloc[start:end].drop(columns=["genre_set"], errors="ignore")
        return jsonify({
            "meta": {
                "total": int(len(shuffled)),
                "page": page,
                "size": size,
                "total_pages": total_pages,
                "returned": int(len(page_slice))
            },
            "movies": page_slice.to_dict(orient="records")
        })

    # Non-paginated: just take first req_limit after shuffle
    subset = shuffled.head(req_limit).drop(columns=["genre_set"], errors="ignore")
    return jsonify(subset.to_dict(orient="records"))

# NEW: list all available genres for navbar dropdown
@app.route("/genres", methods=["GET"])
def list_genres():
    movies = safe_read_csv(MOVIE_PATH)
    if movies.empty:
        return jsonify({"genres": []})
    if "genre_set" not in movies.columns:
        movies["genre_set"] = movies["genres"].apply(parse_genres)
    all_genres = sorted({g for gs in movies["genre_set"] for g in gs})
    return jsonify({"genres": all_genres})


@app.route("/movie/<int:movie_id>", methods=["GET"])
def get_movie(movie_id):
    movies = safe_read_csv(MOVIE_PATH)
    if movies.empty:
        return jsonify({"error": "No movies found"}), 404
    # Ensure movieId type
    try:
        movies["movieId"] = movies["movieId"].astype(int)
    except Exception:
        pass
    row = movies[movies["movieId"] == movie_id]
    if row.empty:
        return jsonify({"error": "Movie not found"}), 404
    return jsonify(row.iloc[0].to_dict())


@app.route("/movie/search", methods=["GET"])
def search_movie():
    q = request.args.get("query", "").strip()
    if not q:
        return jsonify({"error": "query parameter required"}), 400
    movies = safe_read_csv(MOVIE_PATH)
    if movies.empty:
        return jsonify({"error": "No movies found"}), 404
    # Case-insensitive substring match on title
    mask = movies["title"].str.contains(q, case=False, na=False)
    results = movies[mask]
    if results.empty:
        # Try looser match: split words and match any
        parts = [p for p in q.split() if p]
        if parts:
            mask2 = movies["title"].apply(lambda t: any(p.lower() in str(t).lower() for p in parts))
            results = movies[mask2]

    if results.empty:
        return jsonify([]), 200

    # Ensure movieId type and return list of matches
    try:
        results["movieId"] = results["movieId"].astype(int)
    except Exception:
        pass
    return jsonify(results.to_dict(orient="records"))


# --- HEALTH CHECK ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🎬 Movie Recommender API running!"})
@app.route("/recommend/update/<int:user_id>", methods=["GET"])
def update_recommendations(user_id):
    """
    Retrain the model synchronously (wait until done)
    and return freshly updated recommendations.
    """
    print(f"🔁 Refreshing recommendations synchronously for user {user_id}...")

    retrain_model()  # wait until done, not in thread
    print("✅ Model retrained, now fetching updated recommendations.")

    return recommend(user_id)
@app.route("/recommend/refresh/<int:user_id>", methods=["GET"])
def refresh_recommendations(user_id):
    """
    Retrain synchronously and return updated recommendations immediately.
    This ensures the frontend sees updated recommendations right after a movie is liked.
    """
    print(f"🔁 Force retraining for user {user_id}")
    retrain_model()  # synchronous retrain
    return recommend(user_id)




# ===== AI Agent endpoint (NEW) =====
@app.route("/agent/chat", methods=["POST"])
def agent_chat():
    data = request.json or {}
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "message is required"}), 400

    top_k = int(data.get("top_k", 5))

    # ==================================
    # 1) FULL LLM MOVIE IDENTIFICATION
    # ==================================
    identified = []
    summary = ""
    trailer_query = ""
    extra_info = ""

    if openai_client:
        prompt = f"""
        The user said: "{user_msg}".

        1. Identify the real movie(s) the user is talking about.
           Example: "3 friends go to Vegas, get drunk" → "The Hangover (2009)"

        2. If no exact movie matches, suggest the closest real movies.

        3. Write a summary for the MAIN movie using your external knowledge.
           You are allowed to use any data you know.

        4. Suggest a YouTube trailer search query.

        Respond in this JSON structure:

        {{
            "movies": ["The Hangover (2009)", "The Hangover Part II"],
            "summary": "string",
            "trailer_query": "string",
            "keywords": ["vegas", "friends", "comedy"],
            "genres": ["comedy"]
        }}
        """

        try:
            resp = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                response_format={"type": "json_object"},
                temperature=0.2,
                messages=[
                    {"role":"system","content":"You are a movie expert and film historian."},
                    {"role":"user","content": prompt}
                ]
            )
            import json
            parsed = json.loads(resp.choices[0].message.content)

            identified = parsed.get("movies", [])
            summary = parsed.get("summary", "")
            trailer_query = parsed.get("trailer_query", "")
            genres = [g.lower() for g in parsed.get("genres", [])]
            keywords = [k.lower() for k in parsed.get("keywords", [])]

        except Exception as e:
            print("OpenAI error:", e)
            identified = []
            summary = ""
            genres = []
            keywords = []
            trailer_query = ""
    else:
        return jsonify({"error":"OpenAI not available."}), 500

    # ==================================
    # 2) LOAD DATASET
    # ==================================
    movies = safe_read_csv(MOVIE_PATH)
    if movies.empty:
        return jsonify({
            "identified_movie": identified[0] if identified else None,
            "summary": summary,
            "dataset_match": None,
            "similar_movies": []
        })

    movies["clean_title"] = movies["title"].str.lower().str.replace(",", "").str.replace("the ", "")

    # ==================================
    # 3) TRY EXACT MATCH IN DATASET
    # ==================================
    dataset_match = None
    if identified:
        target = identified[0].lower().replace(",", "")
        for _, row in movies.iterrows():
            if target in row["clean_title"]:
                dataset_match = row
                break

    # ==================================
    # 4) FALLBACK – FIND SIMILAR MOVIES IN DATASET
    # ==================================
    movies["score"] = 0

    for g in genres:
        movies["score"] += movies["genres"].str.lower().str.contains(g, na=False).astype(int) * 2

    for kw in keywords:
        movies["score"] += movies["clean_title"].str.contains(kw, na=False).astype(int)

    similar = movies.sort_values("score", ascending=False).head(top_k)

    # ==================================
    # 5) TRAILER LINK
    # ==================================
    import urllib.parse
    trailer_link = ""
    if trailer_query:
        trailer_link = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(trailer_query)

    # ==================================
    # 6) RESPONSE
    # ==================================
    return jsonify({
        "agent_reply": "Here’s what I found!",
        "identified_movie": identified[0] if identified else None,
        "summary": summary,
        "trailer_link": trailer_link,
        "dataset_match": dataset_match.to_dict() if dataset_match is not None else None,
        "similar_from_dataset": similar[["movieId", "title", "poster_url", "genres"]].to_dict(orient="records")
    })

# REPLACED: reliable free-port check bound to the chosen host
def _find_free_port(start_port=5000, tries=20, host="0.0.0.0"):
    for p in range(start_port, start_port + tries):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, p))
            s.listen(1)
            s.close()
            return p
        except OSError:
            try:
                s.close()
            except Exception:
                pass
            continue
    return start_port

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    env_port = os.getenv("PORT")
    debug = os.getenv("FLASK_DEBUG", "1") == "1"

    if env_port:
        port = int(env_port)
        print(f"Starting server on {host}:{port} (from PORT env)")
    else:
        port = _find_free_port(5000, host=host)
        if port != 5000:
            print(f"Port 5000 busy. Using free port {port} instead.")
        else:
            print("Using default port 5000.")
        # NEW: pin the chosen port for the reloader child process
        os.environ["PORT"] = str(port)

    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    print(f"App will be available at: http://{display_host}:{port}")

    try:
        app.run(host=host, port=port, debug=debug)
    except OSError as e:
        if "Address already in use" in str(e) or getattr(e, "errno", None) in (48, 98):
            fallback_port = _find_free_port(port + 1, host=host)
            print(f"Port {port} became busy. Falling back to {fallback_port}.")
            # Also pin the fallback for any reloader
            os.environ["PORT"] = str(fallback_port)
            app.run(host=host, port=fallback_port, debug=debug, use_reloader=False)
        else:
            raise
