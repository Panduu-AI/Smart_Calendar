# Appointment Recommender ML Module

This repository contains the AI/ML microservice for appointment reminders and slot recommendations.

# What it does
- Primary user (doctor) can set a custom reminder interval per patient.
- System sends reminders automatically based on last booking + interval.
- Patients can book via:
  - Book Appointment page → 1 manual option + 2 AI-suggested slots.
  - Reminder notification → preferred single-slot page (last appointment time) + confirm.
- The system logs candidates and chosen slots to build a training dataset. Weekly retrain or manual retrain available.

   
