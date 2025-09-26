import os

# Database connection
DB_URL = os.getenv("DB_URL", "postgresql://user:pass@localhost:5432/dbname")

# ML model path
MODEL_PATH = os.getenv("MODEL_PATH", "models/ranking_model.pkl")

# Notification service endpoint
NOTIF_ENDPOINT = os.getenv("NOTIF_ENDPOINT", "http://notification-service/send")

# Scheduler intervals
REMINDER_CHECK_INTERVAL = 10   # minutes
MODEL_RETRAIN_INTERVAL = 7     # days
