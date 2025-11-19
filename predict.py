import pickle
import numpy as np

def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

def predict_malaria(patient_data):
    model, features = load_model()
    
    # Ensure correct feature order
    X = np.array([patient_data[f] for f in features]).reshape(1, -1)
    
    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    return {
        'malaria': int(prediction),
        'probability': float(probability)
    }

if __name__ == '__main__':
    # Test
    sample = {
        'Age': 25,
        'Sex': 1,  # Encoded: Male=1, Female=0
        'Residence_Area': 0, 
        'Fever': 1,
        'Headache': 1,
        'Abdominal_Pain': 0,
        'General_Body_Malaise': 1,
        'Dizziness': 0,
        'Vomiting': 0,
        'Confusion': 0,
        'Backache': 0,
        'Chest_Pain': 0,
        'Coughing': 0,
        'Joint_Pain': 1
        # Add other features...
    }
    result = predict_malaria(sample)
    print(result)

