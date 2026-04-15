#!/usr/bin/env python3
"""10 Demo Patients for HDP Risk Predictor"""
import firebase_admin
from firebase_admin import credentials, firestore
from pathlib import Path
from datetime import datetime, timedelta
import random

cred = credentials.Certificate(str(Path('firebase-service-account.json')))
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

print("Seeding 10 patients...")

patients = [
    ("patient_001", "Alice Johnson", "1990-03-15", 4, 0.15),
    ("patient_002", "Sarah Mitchell", "1988-07-22", 5, 0.75),
    ("patient_003", "Emma Rodriguez", "1992-11-08", 3, 0.05),
    ("patient_004", "Jennifer Lee", "1989-05-19", 3, 0.50),
    ("patient_005", "Rebecca Davis", "1991-09-27", 3, 0.18),
    ("patient_006", "Michelle Garcia", "1987-02-14", 4, 0.95),
    ("patient_007", "Laura Martinez", "1993-08-03", 4, 0.10),
    ("patient_008", "Patricia Thompson", "1986-12-10", 5, 0.60),
    ("patient_009", "Katherine White", "1994-06-21", 3, 0.07),
    ("patient_010", "Victoria Brown", "1985-04-17", 5, 0.90),
]

for pid, name, dob, visits, base_risk in patients:
    db.collection('patients').document(pid).set({'name': name, 'dob': dob})
    for v in range(visits):
        db.collection('patients').document(pid).collection('visits').document(f'Visit {v+1}').set({
            'label': f'Visit {v+1} (Week {6+v*8})',
            'date': (datetime.now() - timedelta(days=130-v*30)).isoformat(),
            'sbp': 115 + v*4 + random.randint(-2, 8),
            'dbp': 74 + v*3 + random.randint(-1, 6),
            'bmi': 22 + v*0.3 + random.random(),
            'blood_sugar': 85 + v*5 + random.randint(-2,6),
            'risk': round(base_risk + v*0.08 + random.random()*0.05, 2),
            'notes': f'Visit {v+1}'
        })
    print(f"  {name}: {visits} visits")

print("\n✅ DONE. Run: streamlit run app.py")
