"""
Flask API + Scheduler: endpoints for recommendation, confirm, reminder setting,
reminder-triggered slot view and background jobs.
"""
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from recommender_service import recommend_top_k, log_recommendation_session, update_chosen_slot, get_training_data, train_ranking_model
from config import REMINDER_CHECK_INTERVAL, MODEL_RETRAIN_INTERVAL, DB_URL, NOTIF_ENDPOINT
from sqlalchemy import text, create_engine
from datetime import datetime, timedelta
import pandas as pd
import requests

engine = create_engine(DB_URL, echo=False)
app = Flask(__name__)
sched = BackgroundScheduler()
sched.start()

# -------------------------
# Recommend slots (UI path)
# Returns 1 manual option + 2 AI suggested slots
# -------------------------
@app.route("/recommend_slots", methods=["POST"])
def recommend_slots():
    data = request.json
    primary_user_id = int(data['primary_user_id'])
    secondary_user_id = int(data['secondary_user_id'])
    # request 2 AI slots and add manual option as first choice
    topk, cand_df = recommend_top_k(primary_user_id, secondary_user_id, k=2, return_candidates=True)
    manual_slot = {"slot_id": None, "slot_time": "manual_input", "score": 1.0}
    slots = [manual_slot] + topk
    session_id = log_recommendation_session(primary_user_id, secondary_user_id, cand_df)
    return jsonify({"slots": slots, "session_id": session_id})

# -------------------------
# Reminder-triggered slots (only last appointment time)
# -------------------------
@app.route("/reminder_slots", methods=["POST"])
def reminder_slots():
    data = request.json
    p = int(data['primary_user_id'])
    s = int(data['secondary_user_id'])
    with engine.connect() as conn:
        last_b = conn.execute(text("""
            SELECT start_time FROM bookings
            WHERE primary_user_id=:p AND secondary_user_id=:s
            ORDER BY start_time DESC LIMIT 1
        """), {"p": p, "s": s}).fetchone()

    if not last_b:
        return jsonify({"slots": [], "message": "No past booking found."})
    slot = {"slot_id": None, "slot_time": str(last_b['start_time']), "score": 1.0}
    return jsonify({"slots": [slot]})

# -------------------------
# Set / update reminder (doctor)
# -------------------------
@app.route("/set_reminder", methods=["POST"])
def set_reminder():
    data = request.json
    p = int(data['primary_user_id'])
    s = int(data['secondary_user_id'])
    d = int(data['interval_days'])
    sql = text("""
      INSERT INTO reminder_settings (primary_user_id, secondary_user_id, reminder_interval_days, updated_at, active)
      VALUES (:p,:s,:d,NOW(), true)
      ON CONFLICT (primary_user_id, secondary_user_id)
      DO UPDATE SET reminder_interval_days = :d, updated_at = NOW(), active = true
    """)
    with engine.begin() as conn:
        conn.execute(sql, {"p": p, "s": s, "d": d})
    return jsonify({"status": "ok"})

# -------------------------
# Confirm appointment (both manual and AI slot)
# -------------------------
@app.route("/confirm_appointment", methods=["POST"])
def confirm_appointment():
    data = request.json
    p = int(data['primary_user_id'])
    s = int(data['secondary_user_id'])
    slot_id = data.get('slot_id')  # can be None for manual
    start = pd.to_datetime(data['slot_time'])
    end = start + pd.Timedelta(minutes=int(data.get('duration_minutes', 30)))
    with engine.begin() as conn:
        conn.execute(text("""
          INSERT INTO bookings (primary_user_id, secondary_user_id, start_time, end_time, status, created_at)
          VALUES (:p,:s,:start,:end,'booked',NOW())
        """), {"p": p, "s": s, "start": start, "end": end})
        if slot_id is not None:
            conn.execute(text("UPDATE avail_slots SET is_booked=true WHERE id=:sid"), {"sid": int(slot_id)})
    # update training log: mark chosen slot if session_id provided
    session_id = data.get('session_id')
    if session_id:
        try:
            update_chosen_slot(session_id, slot_id)
        except Exception:
            pass
    return jsonify({"status": "ok"})

# -------------------------
# Background reminder job
# -------------------------
def reminder_job():
    """
    Runs every REMINDER_CHECK_INTERVAL minutes.
    For each active reminder_settings entry:
      - get last booking for that primary-secondary pair
      - compute remind_time = last_booking + interval_days
      - if now >= remind_time and last_reminder_sent is None or older -> send reminder
      - reminder payload uses reminder_slots API (only last appointment slot)
    """
    sql = text("SELECT * FROM reminder_settings WHERE active=true")
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    now = datetime.utcnow()
    for row in rows:
        p, s = row['primary_user_id'], row['secondary_user_id']
        interval = row['reminder_interval_days']
        last_sent = row['last_reminder_sent']
        with engine.connect() as conn:
            last_b = conn.execute(text("""
              SELECT start_time FROM bookings
              WHERE primary_user_id=:p AND secondary_user_id=:s
              ORDER BY start_time DESC LIMIT 1
            """), {"p": p, "s": s}).fetchone()
        if not last_b:
            continue
        remind_time = last_b['start_time'] + timedelta(days=interval)
        # send if never sent before or now passed the remind_time
        if last_sent is None or now >= remind_time:
            # call local endpoint to get the single slot
            try:
                r = requests.post(f"http://127.0.0.1:8000/reminder_slots", json={"primary_user_id": p, "secondary_user_id": s}, timeout=5)
                slots = r.json().get("slots", [])
            except Exception:
                slots = []
            session_id = None
            # log candidates into recommendation_logs for record-keeping (if any)
            if slots:
                # create a small candidate_df to log (one row)
                try:
                    import pandas as pd
                    cand_df = pd.DataFrame([{
                        'slot_id': None,
                        'slot_time': pd.to_datetime(slots[0]['slot_time']) if slots[0]['slot_time'] != 'manual_input' else pd.Timestamp.now(),
                        'same_hour': 1,
                        'same_dow': 1,
                        'hour_diff': 0.0,
                        'slot_is_free': 1,
                        'recent_count': 0
                    }])
                    session_id = log_recommendation_session(p, s, cand_df)
                except Exception:
                    session_id = None
            payload = {"to_secondary_user_id": s, "primary_user_id": p, "message": "Your next appointment is due.", "recommended_slots": slots, "session_id": session_id}
            try:
                requests.post(NOTIF_ENDPOINT, json=payload, timeout=5)
            except Exception:
                pass
            # update last_reminder_sent
            with engine.begin() as conn:
                conn.execute(text("UPDATE reminder_settings SET last_reminder_sent = NOW() WHERE id=:id"), {"id": row['id']})

# -------------------------
# Retrain job
# -------------------------
def retrain_job():
    try:
        df = get_training_data(limit=5000)
        if df is not None and not df.empty:
            train_ranking_model(df)
    except Exception:
        pass

# schedule background jobs
sched.add_job(reminder_job, 'interval', minutes=REMINDER_CHECK_INTERVAL, id='reminder_job', replace_existing=True)
sched.add_job(retrain_job, 'interval', days=MODEL_RETRAIN_INTERVAL, id='retrain_job', replace_existing=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

