"""
ML core: feature extraction, candidate scoring (rules + ML), logging, training.
"""

import os
import uuid
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression
from config import DB_URL, MODEL_PATH

# DB engine
engine = create_engine(DB_URL, echo=False)

# -----------------------------
# Data fetching helpers
# -----------------------------
def fetch_booking_history(primary_user_id, secondary_user_id, limit=200):
    sql = text("""
      SELECT id, start_time, end_time, status
      FROM bookings
      WHERE primary_user_id = :p AND secondary_user_id = :s
      ORDER BY start_time DESC
      LIMIT :limit
    """)
    return pd.read_sql(sql, engine, params={"p": primary_user_id, "s": secondary_user_id, "limit": limit})

def fetch_future_avail_slots(primary_user_id, window_days=30):
    sql = text("""
      SELECT id, slot_time, is_booked
      FROM avail_slots
      WHERE primary_user_id = :p
        AND slot_time >= NOW()
        AND slot_time <= NOW() + INTERVAL ':days days'
    """.replace(':days', str(window_days)))
    return pd.read_sql(sql, engine, params={"p": primary_user_id})

# -----------------------------
# Feature engineering
# -----------------------------
def build_candidate_features(primary_user_id, secondary_user_id, candidate_slots_df, history_df):
    """
    Build features for each candidate slot using the user's history.
    Returns a DataFrame with features for each candidate slot.
    """
    prev = history_df[history_df['status'] != 'cancelled']
    prev_row = prev.iloc[0] if prev.shape[0] > 0 else None

    features = []
    for _, row in candidate_slots_df.iterrows():
        slot_ts = pd.to_datetime(row['slot_time'])
        slot_hour = slot_ts.hour
        slot_dow = slot_ts.dayofweek  # Monday=0
        slot_is_free = 0 if row.get('is_booked', False) else 1

        # Defaults
        same_hour = 0
        same_dow = 0
        hour_diff = 999.0
        days_since_last = 999
        recent_count = 0

        if prev_row is not None:
            prev_ts = pd.to_datetime(prev_row['start_time'])
            same_hour = 1 if prev_ts.hour == slot_hour else 0
            same_dow = 1 if prev_ts.dayofweek == slot_dow else 0
            hour_diff = abs((slot_ts - prev_ts).total_seconds()) / 3600.0
            days_since_last = (slot_ts.date() - prev_ts.date()).days
            recent = prev.head(12)
            recent_count = int((recent['start_time'].apply(lambda x: pd.to_datetime(x).hour) == slot_hour).sum())

        features.append({
            'slot_id': int(row['id']),
            'slot_time': slot_ts,
            'slot_hour': int(slot_hour),
            'slot_dow': int(slot_dow),
            'slot_is_free': int(slot_is_free),
            'same_hour': int(same_hour),
            'same_dow': int(same_dow),
            'hour_diff': float(hour_diff),
            'days_since_last': int(days_since_last),
            'recent_count': int(recent_count)
        })

    return pd.DataFrame(features)

# -----------------------------
# Model utilities
# -----------------------------
def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def train_ranking_model(training_df, save_path=MODEL_PATH):
    """
    Train a logistic regression ranking model on the training DataFrame.
    training_df must contain columns: slot_is_free, same_hour, same_dow, hour_diff, days_since_last, recent_count, chosen
    """
    if training_df.empty:
        raise ValueError("Empty training data")
    X = training_df[['slot_is_free','same_hour','same_dow','hour_diff','days_since_last','recent_count']].fillna(0)
    y = training_df['chosen'].astype(int)
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    return model

# -----------------------------
# Scoring & recommend
# -----------------------------
def score_candidates(candidate_df, model=None):
    # Prepare features for ML
    X = candidate_df[['slot_is_free','same_hour','same_dow','hour_diff','days_since_last','recent_count']].fillna(0)
    ml_score = np.zeros(len(candidate_df))
    if model is not None:
        try:
            ml_score = model.predict_proba(X)[:,1]
        except Exception:
            ml_score = np.zeros(len(candidate_df))

    # Simple rule-based boost
    rule_score = (
        candidate_df['same_hour'] * 0.5 +
        candidate_df['same_dow'] * 0.3 +
        candidate_df['slot_is_free'] * 0.2 +
        candidate_df['recent_count'] * 0.05
    )

    # hour diff penalty -> convert to decaying score
    diff_score = np.exp(-candidate_df['hour_diff'] / 24.0)

    total_score = 0.6 * ml_score + 0.3 * rule_score + 0.1 * diff_score
    candidate_df = candidate_df.copy()
    candidate_df['score'] = total_score
    candidate_df = candidate_df.sort_values('score', ascending=False).reset_index(drop=True)
    return candidate_df

def recommend_top_k(primary_user_id, secondary_user_id, k=3, window_days=30, return_candidates=False):
    """
    Returns top-k recommended slots.
    If return_candidates=True, returns (results_list, candidate_features_df)
    else returns results_list
    """
    history = fetch_booking_history(primary_user_id, secondary_user_id, limit=200)
    slots = fetch_future_avail_slots(primary_user_id, window_days=window_days)
    if slots.empty:
        if return_candidates:
            return [], pd.DataFrame()
        return []

    cand_feats = build_candidate_features(primary_user_id, secondary_user_id, slots, history)
    model = load_model()
    scored = score_candidates(cand_feats, model)
    topk = scored.head(k)
    results = [
        {"slot_id": int(r['slot_id']), "slot_time": r['slot_time'].isoformat(), "score": float(r['score'])}
        for _, r in topk.iterrows()
    ]
    if return_candidates:
        return results, scored
    return results

# -----------------------------
# Logging (training data)
# -----------------------------
def log_recommendation_session(primary_user_id, secondary_user_id, candidate_df):
    """
    Logs candidate rows to recommendation_logs with chosen=0 default.
    candidate_df expected to have columns: slot_id, slot_time, same_hour, same_dow, hour_diff, slot_is_free, recent_count
    Returns session_id
    """
    session_id = str(uuid.uuid4())
    if candidate_df is None or candidate_df.empty:
        return session_id

    with engine.begin() as conn:
        for _, row in candidate_df.iterrows():
            sql = text("""
              INSERT INTO recommendation_logs
              (session_id, primary_user_id, secondary_user_id, slot_id, slot_time,
               same_hour, same_dow, hour_diff, slot_is_free, recent_count, chosen)
              VALUES (:sid,:p,:s,:slot_id,:slot_time,
                      :same_hour,:same_dow,:hour_diff,:slot_is_free,:recent_count,0)
            """)
            conn.execute(sql, {
                "sid": session_id,
                "p": primary_user_id,
                "s": secondary_user_id,
                "slot_id": int(row['slot_id']),
                "slot_time": row['slot_time'].to_pydatetime(),
                "same_hour": int(row['same_hour']),
                "same_dow": int(row['same_dow']),
                "hour_diff": float(row['hour_diff']),
                "slot_is_free": int(row['slot_is_free']),
                "recent_count": int(row['recent_count'])
            })
    return session_id

def update_chosen_slot(session_id, chosen_slot_id):
    """
    Mark chosen slot=1 for the given session_id and slot_id.
    """
    if session_id is None:
        return
    sql = text("""
      UPDATE recommendation_logs
      SET chosen = 1
      WHERE session_id = :sid AND slot_id = :slot
    """)
    with engine.begin() as conn:
        conn.execute(sql, {"sid": session_id, "slot": chosen_slot_id})

def get_training_data(limit=5000):
    """
    Pull recent recommendation_logs used for training.
    Returns a pandas DataFrame with training columns plus 'chosen'.
    """
    sql = text("""
      SELECT same_hour, same_dow, hour_diff, slot_is_free, recent_count, chosen
      FROM recommendation_logs
      ORDER BY created_at DESC
      LIMIT :limit
    """)
    return pd.read_sql(sql, engine, params={"limit": limit})

