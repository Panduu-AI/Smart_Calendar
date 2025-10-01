import os

# Database connection (update this in your environment)
DB_URL = os.getenv("DB_URL", "postgresql://user:pass@localhost:5432/appointments")

# ML model path
MODEL_PATH = os.getenv("MODEL_PATH", "models/ranking_model.pkl")

# Notification service endpoint (to be provided by backend or use local receiver)
NOTIF_ENDPOINT = os.getenv("NOTIF_ENDPOINT", "http://localhost:5001/notify")

# Scheduler intervals
REMINDER_CHECK_INTERVAL = int(os.getenv("REMINDER_CHECK_INTERVAL_MINUTES", "10"))   # minutes
MODEL_RETRAIN_INTERVAL = int(os.getenv("MODEL_RETRAIN_INTERVAL_DAYS", "7"))        # days


