import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Firestore integration (optional, skip if not installed)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ModuleNotFoundError:
    firebase_admin = None
    credentials = None
    firestore = None

from pathlib import Path

st.set_page_config(page_title='HDP Longitudinal Predictor', layout='wide')

# Firebase service setup
firebase_service_account_path = Path('firebase-service-account.json')
db = None
if firebase_admin and firestore and firebase_service_account_path.exists():
    cred = credentials.Certificate(str(firebase_service_account_path))
    firebase_admin.initialize_app(cred)
    db = firestore.client()


PASSWORD = 'health123'

DEFAULT_PATIENTS = [
    {
        'id': 'p001',
        'name': 'Amina Adamu',
        'dob': '1995-07-22',
        'visits': [
            {'label': 'Visit 1', 'date': '2026-01-10', 'sbp': 112, 'dbp': 72, 'risk': 0.17, 'notes': 'Normal monitoring.'},
            {'label': 'Visit 2', 'date': '2026-02-12', 'sbp': 118, 'dbp': 78, 'risk': 0.28, 'notes': 'Raised BP, start diet.'}
        ]
    },
    {
        'id': 'p002',
        'name': 'Bintu Kamara',
        'dob': '1990-03-05',
        'visits': [
            {'label': 'Visit 1', 'date': '2026-01-20', 'sbp': 130, 'dbp': 84, 'risk': 0.43, 'notes': 'Pre-hypertension alert.'},
            {'label': 'Visit 2', 'date': '2026-02-15', 'sbp': 136, 'dbp': 88, 'risk': 0.58, 'notes': 'Prescription started.'}
        ]
    },
    {
        'id': 'p003',
        'name': 'Chiamaka Nwosu',
        'dob': '1998-11-28',
        'visits': [
            {'label': 'Visit 1', 'date': '2026-01-05', 'sbp': 104, 'dbp': 66, 'risk': 0.08, 'notes': 'Healthy baseline.'}
        ]
    }
]

# Firestore persistence helpers

def save_patient_to_firestore(patient):
    if not db:
        return
    doc_ref = db.collection('patients').document(patient['id'])
    metadata = {k: patient[k] for k in ['name', 'dob'] if k in patient}
    metadata['updated_at'] = firestore.SERVER_TIMESTAMP
    doc_ref.set(metadata, merge=True)

    visits = patient.get('visits', [])
    for visit in visits:
        visit_ref = doc_ref.collection('visits').document(visit['label'])
        visit_data = {
            'date': visit.get('date'),
            'sbp': int(visit.get('sbp', 0)),
            'dbp': int(visit.get('dbp', 0)),
            'risk': float(visit.get('risk', 0.0)),
            'notes': visit.get('notes', ''),
            'created_at': firestore.SERVER_TIMESTAMP
        }
        visit_ref.set(visit_data, merge=True)


def load_patients_from_firestore():
    if not db:
        return DEFAULT_PATIENTS
    patients = []
    for pdoc in db.collection('patients').stream():
        pdata = pdoc.to_dict()
        pid = pdoc.id
        if not pdata:
            continue
        patient = {
            'id': pid,
            'name': pdata.get('name', 'Unknown'),
            'dob': pdata.get('dob', '1970-01-01'),
            'visits': []
        }
        visit_docs = db.collection('patients').document(pid).collection('visits').stream()
        for vdoc in visit_docs:
            vdata = vdoc.to_dict()
            patient['visits'].append({
                'label': vdoc.id,
                'date': vdata.get('date', ''),
                'sbp': int(vdata.get('sbp', 0)),
                'dbp': int(vdata.get('dbp', 0)),
                'risk': float(vdata.get('risk', 0.0)),
                'notes': vdata.get('notes', '')
            })
        patients.append(patient)
    return patients


if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'patients' not in st.session_state:
    st.session_state['patients'] = load_patients_from_firestore() if db else DEFAULT_PATIENTS
if 'selected_patient' not in st.session_state:
    st.session_state['selected_patient'] = st.session_state['patients'][0]['id'] if st.session_state['patients'] else None
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'Dark'
if 'enrollment_mode' not in st.session_state:
    st.session_state['enrollment_mode'] = False

if not st.session_state['logged_in']:
    st.title('Pregnancy Hypertensive Disorder (HDP) Longitudinal Risk Explorer')
    with st.form('login_form'):
        st.write('### User Login')
        username = st.text_input('Username', value='doctor')
        password = st.text_input('Password', type='password')
        submitted = st.form_submit_button('Login')
        if submitted:
            if password == PASSWORD:
                st.session_state['logged_in'] = True
                st.success('Login successful')
            else:
                st.error('Invalid credentials')
    st.stop()

st.title('Longitudinal Risk Explorer')

theme_choice = st.radio('Theme:', ['Dark', 'Light'], index=0 if st.session_state['theme'] == 'Dark' else 1, horizontal=True)
if theme_choice != st.session_state['theme']:
    st.session_state['theme'] = theme_choice

if st.session_state['theme'] == 'Dark':
    st.markdown('<style>body {background-color: #0E1117; color: white; margin-top: 8px;}</style>', unsafe_allow_html=True)
    st.markdown('<style>section.main {padding-top: 10px;}</style>', unsafe_allow_html=True)
else:
    st.markdown('<style>body {background-color: white; color: black; margin-top: 8px;}</style>', unsafe_allow_html=True)
    st.markdown('<style>section.main {padding-top: 10px;}</style>', unsafe_allow_html=True)

top_buttons_col1, top_buttons_col2, top_buttons_spacer = st.columns([1, 1, 3])

with top_buttons_col1:
    if st.button('🆕 Enroll New Patient', use_container_width=True):
        st.session_state['enrollment_mode'] = not st.session_state['enrollment_mode']

with top_buttons_col2:
    if st.button('👥 View Patients', use_container_width=True):
        st.session_state['enrollment_mode'] = False

if st.session_state['enrollment_mode']:
    st.markdown('---')
    st.subheader('Enroll New Patient & Initial Visit')
    
    enroll_left, enroll_right = st.columns([1, 2])
    
    with enroll_left:
        st.write('### Patient Information')
        enroll_name = st.text_input('Full name', key='enroll_name')
        enroll_dob = st.text_input('Date of birth (YYYY-MM-DD)', key='enroll_dob', placeholder='1990-03-15')
        enroll_age = st.number_input('Age', min_value=15, max_value=50, value=28, key='enroll_age')
        enroll_notes_patient = st.text_area('Patient history / notes', '', key='enroll_notes_patient', height=120)
    
    with enroll_right:
        st.write('### Initial Visit (Visit 1)')
        visit_sbp = st.number_input('SBP', value=115, min_value=80, max_value=220, key='visit_sbp')
        visit_dbp = st.number_input('DBP', value=75, min_value=50, max_value=150, key='visit_dbp')
        visit_bmi = st.number_input('BMI', value=26.0, min_value=16.0, max_value=45.0, key='visit_bmi')
        visit_notes = st.text_area('Clinician notes for this visit', '', key='visit_notes', height=120)
        
        if st.button('Predict & Review', use_container_width=True):
            seq_len = 20
            sbp_seq = np.linspace(visit_sbp - 12, visit_sbp, seq_len) + np.random.normal(0, 2, seq_len)
            dbp_seq = np.linspace(visit_dbp - 8, visit_dbp, seq_len) + np.random.normal(0, 1.5, seq_len)
            map_seq = 0.4 * sbp_seq + 0.6 * dbp_seq
            bmi_seq = np.full(seq_len, visit_bmi) + np.linspace(0, 1.2, seq_len)
            weight_seq = np.linspace(0, 2.0, seq_len)
            oliguria = np.where(np.random.rand(seq_len) < 0.05, 1, 0)
            proteinuria = np.where(np.random.rand(seq_len) < 0.03, 1, 0)
            
            input_data = np.vstack([sbp_seq, dbp_seq, map_seq, bmi_seq, weight_seq, oliguria, proteinuria]).T
            input_model = input_data.reshape(1, seq_len, 7)
            
            clf = joblib.load('artifacts/risk_classifier.pkl')
            reg = joblib.load('artifacts/time_regressor.pkl')
            mean_ = np.load('artifacts/risk_scaler_mean.npy')
            scale_ = np.load('artifacts/risk_scaler_scale.npy')
            scaler = StandardScaler()
            scaler.mean_ = mean_
            scaler.scale_ = scale_
            scaler.var_ = scale_**2
            scaler.n_features_in_ = 7
            
            input_scaled = scaler.transform(input_model.reshape(-1, 7)).reshape(1, seq_len, 7)
            input_flat = input_scaled.reshape(1, -1)  # Flatten for sklearn
            pred_risk = clf.predict_proba(input_flat)[0, 1]  # Probability of positive class
            pred_time = reg.predict(input_flat)[0]
            
            st.session_state['pred_risk'] = pred_risk
            st.session_state['pred_time'] = pred_time
    
    st.markdown('---')
    
    if 'pred_risk' in st.session_state:
        st.write('### Prediction Result')
        col_risk, col_time = st.columns(2)
        with col_risk:
            st.metric('HDP Risk', f"{st.session_state['pred_risk']*100:.1f}%")
        with col_time:
            st.metric('Time to Event (weeks)', f"{st.session_state['pred_time']:.1f}")
        
        st.markdown('---')
        st.write('### Confirm Patient Details')
        
        confirm_col_info, confirm_col_visit = st.columns([1, 1])
        with confirm_col_info:
            st.write('**Patient:**')
            st.write(f"- Name: {enroll_name}")
            st.write(f"- DOB: {enroll_dob}")
            st.write(f"- Age: {enroll_age}")
        with confirm_col_visit:
            st.write('**Visit 1:**')
            st.write(f"- SBP/DBP: {visit_sbp}/{visit_dbp}")
            st.write(f"- BMI: {visit_bmi}")
            st.write(f"- Risk: {st.session_state['pred_risk']*100:.1f}%")
        
        if st.warning('Are you sure you want to save this new patient?'):
            pass
        
        save_col1, save_col2 = st.columns(2)
        with save_col1:
            if st.button('✅ Confirm & Save', use_container_width=True):
                if not enroll_name or not enroll_dob:
                    st.error('Name and DOB required.')
                else:
                    new_id = f"p{100 + len(st.session_state['patients']) + 1:03d}"
                    new_patient = {
                        'id': new_id,
                        'name': enroll_name,
                        'dob': enroll_dob,
                        'visits': [{
                            'label': 'Visit 1',
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'sbp': int(visit_sbp),
                            'dbp': int(visit_dbp),
                            'risk': float(st.session_state['pred_risk']),
                            'notes': visit_notes or 'Initial visit'
                        }]
                    }
                    st.session_state['patients'].append(new_patient)
                    if db:
                        save_patient_to_firestore(new_patient)
                    st.session_state['enrollment_mode'] = False
                    st.success(f'✅ Patient {enroll_name} enrolled successfully!')
                    # Streamlit reruns automatically on user actions, explicit rerun removed
        
        with save_col2:
            if st.button('❌ Cancel', use_container_width=True):
                st.session_state['enrollment_mode'] = False
                st.info('Enrollment cancelled.')
                # Streamlit reruns automatically on user actions, explicit rerun removed

    # Define dashboard layout columns before use
    layout_left, layout_main = st.columns([1, 2])

    with layout_left:
        st.subheader('Doctor dashboard')
        st.write('Select patients below:')
        
        search_term = st.text_input('Search patients', '', placeholder='Name or ID', key='patient_search')
        
        filtered_patients = [p for p in st.session_state['patients'] 
                            if search_term.lower() in p['name'].lower() or search_term.lower() in p['id'].lower()]
        
        for p in filtered_patients:
            if st.button(p['name'], key=f"patient_{p['id']}", use_container_width=True):
                st.session_state['selected_patient'] = p['id']
        
        if search_term and not filtered_patients:
            st.info('No patients match your search.')
        
        st.markdown('---')
        if st.button('🗑️ Remove selected patient', use_container_width=True):
            if st.session_state['selected_patient'] is not None:
                before = len(st.session_state['patients'])
                st.session_state['patients'] = [p for p in st.session_state['patients'] if p['id'] != st.session_state['selected_patient']]
                if len(st.session_state['patients']) < before:
                    st.success('Removed selected patient')
                    if st.session_state['patients']:
                        st.session_state['selected_patient'] = st.session_state['patients'][0]['id']
                    else:
                        st.session_state['selected_patient'] = None
        
        st.markdown('---')
        selected_patient = next((p for p in st.session_state['patients'] if p['id'] == st.session_state['selected_patient']), None)
        if selected_patient:
            st.write('**Selected:**', selected_patient['name'])
            st.write('**DOB:**', selected_patient['dob'])
            st.write('Visits recorded:', len(selected_patient['visits']))
    
    with layout_main:
        selected_patient = next((p for p in st.session_state['patients'] if p['id'] == st.session_state['selected_patient']), None)
        
        if not selected_patient:
            st.error('No patient selected')
            st.stop()
        
        st.subheader(f"Patient: {selected_patient['name']}")
        
        visit_labels = [v.get('label', f"Visit {i+1}") for i, v in enumerate(selected_patient.get('visits', []))]
        if not visit_labels:
            st.warning('No visits available. Enter at least one visit first.')
            st.stop()

        selected_visit_label = st.selectbox('Choose visit history entry:', visit_labels)
        visit_record = next((v for v in selected_patient['visits'] if v.get('label') == selected_visit_label), selected_patient['visits'][0])
        
        history_col, entry_col = st.columns([1, 2])
        
        with history_col:
            st.markdown('### Existing visit')
            st.write('Date:', visit_record['date'])
            st.write('SBP:', visit_record['sbp'])
            st.write('DBP:', visit_record['dbp'])
            st.write('Risk:', f"{visit_record['risk']*100:.1f}%")
            st.write('Notes:', visit_record['notes'])
        
        with entry_col:
            st.markdown('### Add new measurements')
            new_sbp = st.number_input('SBP', value=visit_record['sbp'], min_value=80, max_value=220)
            new_dbp = st.number_input('DBP', value=visit_record['dbp'], min_value=50, max_value=150)
            new_bmi = st.number_input('BMI', value=26.0, min_value=16.0, max_value=45.0)
            new_note = st.text_area('Doctor notes', '')
            
            if st.button('Predict & Save new visit'):
                seq_len = 20
                sbp_seq = np.linspace(new_sbp - 12, new_sbp, seq_len) + np.random.normal(0, 2, seq_len)
                dbp_seq = np.linspace(new_dbp - 8, new_dbp, seq_len) + np.random.normal(0, 1.5, seq_len)
                map_seq = 0.4 * sbp_seq + 0.6 * dbp_seq
                bmi_seq = np.full(seq_len, new_bmi) + np.linspace(0, 1.2, seq_len)
                weight_seq = np.linspace(0, 2.0, seq_len)
                oliguria = np.where(np.random.rand(seq_len) < 0.05, 1, 0)
                proteinuria = np.where(np.random.rand(seq_len) < 0.03, 1, 0)
                
                input_data = np.vstack([sbp_seq, dbp_seq, map_seq, bmi_seq, weight_seq, oliguria, proteinuria]).T
                input_model = input_data.reshape(1, seq_len, 7)
                
                clf = joblib.load('artifacts/risk_classifier.pkl')
                reg = joblib.load('artifacts/time_regressor.pkl')
                mean_ = np.load('artifacts/risk_scaler_mean.npy')
                scale_ = np.load('artifacts/risk_scaler_scale.npy')
                scaler = StandardScaler()
                scaler.mean_ = mean_
                scaler.scale_ = scale_
                scaler.var_ = scale_**2
                scaler.n_features_in_ = 7
                
                input_scaled = scaler.transform(input_model.reshape(-1, 7)).reshape(1, seq_len, 7)
                input_flat = input_scaled.reshape(1, -1)  # Flatten for sklearn
                pred_risk = clf.predict_proba(input_flat)[0, 1]  # Probability of positive class
                pred_time = reg.predict(input_flat)[0]
                
                st.metric('HDP risk', f'{pred_risk*100:.1f}%')
                st.metric('Time to event (weeks)', f'{pred_time:.1f}')
                
                st.warning('Confirm save to patient record?')
                
                confirm_col1, confirm_col2 = st.columns(2)
                with confirm_col1:
                    if st.button('✅ Save visit', key='confirm_save_visit'):
                        adv_note = new_note if new_note else 'No note provided.'
                        next_label = f"Visit {len(selected_patient['visits'])+1}"
                        selected_patient['visits'].append({
                            'label': next_label,
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'sbp': int(new_sbp),
                            'dbp': int(new_dbp),
                            'risk': float(pred_risk),
                            'notes': adv_note
                        })
                        # Persist updated patient list in session state for multi-visit patients
                        st.session_state['patients'] = [
                            (p if p['id'] != selected_patient['id'] else selected_patient)
                            for p in st.session_state['patients']
                        ]
                        if db:
                            # Save visit to Firestore separately
                            visit_doc = selected_patient['visits'][-1]
                            db.collection('patients').document(selected_patient['id']).collection('visits').document(visit_doc['label']).set({
                                'date': visit_doc['date'],
                                'sbp': visit_doc['sbp'],
                                'dbp': visit_doc['dbp'],
                                'risk': float(visit_doc['risk']),
                                'notes': visit_doc['notes'],
                                'created_at': firestore.SERVER_TIMESTAMP
                            })
                        st.session_state['selected_patient'] = selected_patient['id']
                        st.success('New visit saved to patient history.')
                        st.success('✅ Visit added. Select another visit to verify data is shown.')
                
                with confirm_col2:
                    if st.button('❌ Cancel', key='cancel_save_visit'):
                        st.info('Visit not saved.')
        
        st.markdown('---')
        st.write('* Clinical disclaimer: this is demo data only. For production, validate with real patient data + regulatory review.*')


