#!/usr/bin/env python3
"""Setup Firestore - Seed 10 demo patients for HDP Risk Predictor"""
import firebase_admin
from firebase_admin import credentials, firestore
from pathlib import Path
from datetime import datetime, timedelta
import random

cred_path = Path('firebase-service-account.json')
if not cred_path.exists():
    print("❌ firebase-service-account.json not found!")
    exit(1)

cred = credentials.Certificate(str(cred_path))
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()
print("✅ Connected to Firestore")

# 10 demo patients with varying visits (3-5 each)
patients = [
    ("patient_001", "Alice Johnson", "1990-03-15", 4),
    ("patient_002", "Sarah Mitchell", "1988-07-22", 5),
    ("patient_003", "Emma Rodriguez", "1992-11-08", 3),
    ("patient_004", "Jennifer Lee", "1989-05-19", 3),
    ("patient_005", "Rebecca Davis", "1991-09-27", 3),
    ("patient_006", "Michelle Garcia", "1987-02-14", 4),
    ("patient_007", "Laura Martinez", "1993-08-03", 4),
    ("patient_008", "Patricia Thompson", "1986-12-10", 5),
    ("patient_009", "Katherine White", "1994-06-21", 3),
    ("patient_010", "Victoria Brown", "1985-04-17", 5),
]

risk_ranges = [
    (0.02, 0.06),   # Very low
    (0.08, 0.14),   # Low
    (0.15, 0.25),   # Low-moderate
    (0.38, 0.50),   # Moderate
    (0.52, 0.65),   # Moderate-high
    (0.68, 0.82),   # High
    (0.85, 0.94),   # Very high
    (0.92, 0.99),   # Critical
]

# Clear old data
for doc in list(db.collection('patients').stream()):
    for v in doc.reference.collection('visits').stream():
        v.reference.delete()
    doc.reference.delete()
print("✓ Cleared old data")

# Seed 10 patients
total = 0
for pid, name, dob, num_visits in patients:
    db.collection('patients').document(pid).set({'name': name, 'dob': dob, 'updated_at': firestore.SERVER_TIMESTAMP})
    
    for v in range(num_visits):
        week = 6 + (v * 8)
        days_ago = 130 - (v * 30)
        risk_range = risk_ranges[v % len(risk_ranges)]
        risk = risk_range[0] + (random.random() * (risk_range[1] - risk_range[0]))
        
        visit_data = {
            'label': f'Visit {v+1} (Week {week})',
            'date': (datetime.now() - timedelta(days=days_ago)).isoformat(),
            'sbp': 115 + (v * 4) + int(random.random() * 10),
            'dbp': 74 + (v * 3) + int(random.random() * 8),
            'bmi': 22 + (v * 0.3) + (random.random() * 1),
            'blood_sugar': 85 + (v * 5) + int(random.random() * 8),
            'risk': round(risk, 2),
            'notes': f'Follow-up visit {v+1}'
        }
        db.collection('patients').document(pid).collection('visits').document(visit_data['label']).set(visit_data)
    
    print(f"  ✓ {name}: {num_visits} visits")
    total += num_visits

print(f"\n✅ DONE: 10 patients, {total} total visits")
print("\nUp next: streamlit run app.py\n")
