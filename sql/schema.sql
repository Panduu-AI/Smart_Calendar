-- Reminder settings (unique per primary_user & secondary_user)
CREATE TABLE IF NOT EXISTS reminder_settings (
  id SERIAL PRIMARY KEY,
  primary_user_id INT NOT NULL,
  secondary_user_id INT NOT NULL,
  reminder_interval_days INT NOT NULL DEFAULT 7,
  last_reminder_sent TIMESTAMP NULL,
  active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  UNIQUE (primary_user_id, secondary_user_id)
);

-- Bookings
CREATE TABLE IF NOT EXISTS bookings (
  id SERIAL PRIMARY KEY,
  primary_user_id INT NOT NULL,
  secondary_user_id INT NOT NULL,
  start_time TIMESTAMP NOT NULL,
  end_time TIMESTAMP NOT NULL,
  status VARCHAR(30) DEFAULT 'booked',
  created_at TIMESTAMP DEFAULT NOW()
);

-- Available slots
CREATE TABLE IF NOT EXISTS avail_slots (
  id SERIAL PRIMARY KEY,
  primary_user_id INT NOT NULL,
  slot_time TIMESTAMP NOT NULL,
  is_booked BOOLEAN DEFAULT FALSE
);

-- Recommendation logs (for training)
-- Using session_id as TEXT to avoid requiring UUID extension
CREATE TABLE IF NOT EXISTS recommendation_logs (
  id SERIAL PRIMARY KEY,
  session_id TEXT NOT NULL,
  primary_user_id INT NOT NULL,
  secondary_user_id INT NOT NULL,
  slot_id INT NOT NULL,
  slot_time TIMESTAMP NOT NULL,
  same_hour INT,
  same_dow INT,
  hour_diff FLOAT,
  slot_is_free INT,
  recent_count INT,
  chosen INT DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW()
);

