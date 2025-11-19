 Malaria Diagnosis Prediction

ML Zoomcamp 2025 Midterm Project

## Problem

This project predicts malaria in patients using their symptoms and basic information. It helps doctors in Nigerian clinics screen patients quickly when lab testing is not immediately available.

## Dataset

Source: Kaggle Malaria Diagnosis Dataset (https://www.kaggle.com/datasets/ash316/malaria-diagnosis-dataset)

- 1,622 patient records
- 14 features: Age, Sex, Residence Area, and 11 symptoms (Fever, Headache, etc.)
- Target: Malaria positive (1) or negative (0)
- Balanced dataset: 50% positive, 50% negative

## Model

Trained 4 different models and compared them:

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| XGBoost | 96.0% | 0.9901 |
| Logistic Regression | 96.0% | 0.9945 |
| Random Forest | 94.5% | 0.9897 |
| Decision Tree | 93.2% | 0.9295 |

Selected XGBoost as final model.

## Files

- train2.py - trains the model
- predict.py - tests predictions
- server.py - runs the web service
- model_xgb.bin - saved model
- requirements.txt - Python packages needed
- Dockerfile - for running in a container
- malaria.ipynb - data exploration and analysis

## How to Use

### With Docker (recommended)

Build the container:

docker build -t malaria-diagnosis .

text

Run it:

docker run -p 9696:9696 malaria-diagnosis

text

Test it:

curl -X POST http://localhost:9696/predict
-H "Content-Type: application/json"
-d '{
"Age": 28,
"Sex": 1,
"Residence_Area": 0,
"Fever": 1,
"Headache": 1,
"Abdominal_Pain": 1,
"General_Body_Malaise": 1,
"Dizziness": 1,
"Vomiting": 0,
"Confusion": 0,
"Backache": 1,
"Chest_Pain": 0,
"Coughing": 0,
"Joint_Pain": 1
}'

text

### Without Docker

Install packages:

pip install -r requirements.txt

text

Train the model:

python train.py

text

Start the service:

python serve.py

text

Visit http://localhost:9696/docs to see the API documentation.

## API

The service has these endpoints:

- GET / - basic info
- GET /health - check if service is running
- GET /features - list of required features
- POST /predict - make a prediction

To make a prediction, send patient data as JSON. All symptoms are 0 (no) or 1 (yes).

Sex: 0 = Female, 1 = Male

Example response:

{
"malaria_probability": 0.9977,
"malaria": true,
"status": "Positive",
"confidence": "High"
}

text

## Requirements

- Python 3.10
- pandas
- numpy
- scikit-learn
- xgboost-cpu
- fastapi
- uvicorn
- pydantic
