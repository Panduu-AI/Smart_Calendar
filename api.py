from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from recommender_service import recommend_top_k, log_recommendation_session, update_chosen_slot, get_training_data, train_ranking_model
from config import REMINDER_CHECK_INTERVAL, MODEL_RETRAIN_INTERVAL
from sqlalchemy import text
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import os, requests, pandas as pd

from config import DB_URL, NOTIF_ENDPOINT
engine = create_engine(DB_URL, echo=False)

app = Flask(__name__)
sched = BackgroundScheduler()
sched.start()

# 1. Recommend slots
@app.route("/recommend_slots", methods=["POST"])
def recommend_slots():
    data = request.json
    primary_user_id = data['primary_user_id']
    secondary_user_id = data['secondary_user_id']
    topk, cand_df = recommend_top_k(primary_user_id, secondary_user_id, k=3, return_candidates=True)
    session_id = log_recommendation_session(primary_user_id, secondary_user_id, cand_df)
    return jsonify({"slots": topk, "session_id": session_id})

# 2. Set reminder interval
@app.route("/set_reminder", methods=["POST"])
def set_reminder():
    data = request.json
    sql = text("""
      INSERT INTO reminder_settings (primary_user_id, secondary_user_id, reminder_interval_days, updated_at, active)
      VALUES (:p,:s,:d,NOW(), true)
      ON CONFLICT (primary_user_id, secondary_user_id)
      DO UPDATE SET reminder_interval_days = :d, updated_at = NOW(), active = true
    """)
    with engine.begin() as conn:
        conn.execute(sql, {"p": data['primary_user_id'], "s": data['secondary_user_id'], "d": data['interval_days']})
    return jsonify({"status":"ok"})

# 3. Confirm appointment
@app.route("/confirm_appointment", methods=["POST"])
def confirm_appointment():
    data = request.json
    start = pd.to_datetime(data['slot_time'])
    end = start + pd.Timedelta(minutes=data.get('duration_minutes', 30))
    with engine.begin() as conn:
        conn.execute(text("""
          INSERT INTO bookings (primary_user_id, secondary_user_id, start_time, end_time, status, created_at)
          VALUES (:p,:s,:start,:end,'booked',NOW())
        """), {"p":data['primary_user_id'], "s":data['secondary_user_id'], "start":start, "end":end})
        conn.execute(text("UPDATE avail_slots SET is_booked=true WHERE id=:sid"), {"sid":data['slot_id']})
    update_chosen_slot(data['session_id'], data['slot_id'])
    return jsonify({"status":"ok"})

# 4. Reminder job
def reminder_job():
    sql = text("SELECT * FROM reminder_settings WHERE active=true")
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    now = datetime.utcnow()
    for row in rows:
        p, s = row['primary_user_id'], row['secondary_user_id']
        interval = row['reminder_interval_days']
        last_sent = row['last_reminder_sent']
        # fetch last booking
        last_b = conn.execute(text("SELECT start_time FROM bookings WHERE primary_user_id=:p AND secondary_user_id=:s ORDER BY start_time DESC LIMIT 1"),
                              {"p":p,"s":s}).fetchone()
        if not last_b: continue
        remind_time = last_b['start_time'] + timedelta(days=interval)
        if last_sent is None or now > remind_time:
            # get top slots
            topk, cand_df = recommend_top_k(p,s,return_candidates=True)
            session_id = log_recommendation_session(p,s,cand_df)
            payload = {"to_secondary_user_id":s,"primary_user_id":p,"message":"Your next appointment is due.","recommended_slots":topk,"session_id":session_id}
            try:
                requests.post(NOTIF_ENDPOINT,json=payload,timeout=5)
            except: pass
            conn.execute(text("UPDATE reminder_settings SET last_reminder_sent=NOW() WHERE id=:id"),{"id":row['id']})

# 5. Auto retrain job
def retrain_job():
    df = get_training_data(limit=5000)
    if not df.empty:
        train_ranking_model(df)

# schedule jobs
sched.add_job(reminder_job,'interval',minutes=REMINDER_CHECK_INTERVAL)
sched.add_job(retrain_job,'interval',days=MODEL_RETRAIN_INTERVAL)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000)
