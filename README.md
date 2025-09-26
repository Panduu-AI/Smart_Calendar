#Appointment Recommender (AI/ML)

An AI-powered appointment reminder & recommender system that:

Sends weekly reminders to secondary users.

Recommends 3 best time slots based on:

Primary user’s availability.

Secondary user’s booking history.

Continuously learns from confirmations and retrains itself.

##Features

Hybrid rule-based + ML scoring.

Automatic reminder notifications.

Self-growing training dataset (recommendation_logs).

Weekly auto retraining of model.

REST API with Flask.

PostgreSQL integration.

🛠️ Tech Stack

Python 3.10+

Flask (API service)

SQLAlchemy + PostgreSQL (database)

scikit-learn (ML model)

APScheduler (scheduled jobs)

📂 Project Structure
appointment-recommender-ml/
│── README.md
│── requirements.txt
│── config.py
│── recommender_service.py   # ML core
│── api.py                   # Flask API + scheduler
│── retrain.py               # standalone retraining
│── models/                  
│── sql/schema.sql          
│── docs/    
