from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import xgboost as xgb
import numpy as np
from typing import Dict


# Load model at startup
print("Loading model...")
model_file = 'model_xgb.bin'

with open(model_file, 'rb') as f_in:
    model, features = pickle.load(f_in)

print(f"✅ Model loaded with {len(features)} features")
print(f"Features: {features}")


# Initialize FastAPI app
app = FastAPI(
    title="Malaria Diagnosis API",
    description="Predict malaria from clinical symptoms and patient demographics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Request model with validation
class PatientData(BaseModel):
    Age: int = Field(..., ge=0, le=120, description="Patient age (0-120 years)")
    Sex: int = Field(..., ge=0, le=1, description="Sex: 0=Female, 1=Male")
    Residence_Area: int = Field(..., ge=0, le=2, description="Residence area (encoded)")
    Fever: int = Field(..., ge=0, le=1, description="Fever: 0=No, 1=Yes")
    Headache: int = Field(..., ge=0, le=1, description="Headache: 0=No, 1=Yes")
    Abdominal_Pain: int = Field(..., ge=0, le=1, description="Abdominal pain: 0=No, 1=Yes")
    General_Body_Malaise: int = Field(..., ge=0, le=1, description="General malaise: 0=No, 1=Yes")
    Dizziness: int = Field(..., ge=0, le=1, description="Dizziness: 0=No, 1=Yes")
    Vomiting: int = Field(..., ge=0, le=1, description="Vomiting: 0=No, 1=Yes")
    Confusion: int = Field(..., ge=0, le=1, description="Confusion: 0=No, 1=Yes")
    Backache: int = Field(..., ge=0, le=1, description="Backache: 0=No, 1=Yes")
    Chest_Pain: int = Field(..., ge=0, le=1, description="Chest pain: 0=No, 1=Yes")
    Coughing: int = Field(..., ge=0, le=1, description="Coughing: 0=No, 1=Yes")
    Joint_Pain: int = Field(..., ge=0, le=1, description="Joint pain: 0=No, 1=Yes")
    
model_config = {
    "json_schema_extra": {
        "examples": [  
            {
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
            }
        ]
    }
}


# Response model
class PredictionResponse(BaseModel):
    malaria_probability: float
    malaria: bool
    status: str
    confidence: str


@app.get("/")
def root():
    """Root endpoint - API information"""
    return {
        "message": "Malaria Diagnosis API",
        "version": "1.0.0",
        "model": "XGBoost",
        "status": "healthy",
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict",
            "health": "/health",
            "features": "/features"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_count": len(features)
    }


@app.get("/features")
def get_features():
    """Get list of required features"""
    return {
        "features": features,
        "feature_count": len(features)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    """
    Predict malaria from patient symptoms
    
    Returns:
        - malaria_probability: Probability of malaria (0-1)
        - malaria: Boolean prediction (True/False)
        - status: "Positive" or "Negative"
        - confidence: "High", "Medium", or "Low"
    """
    try:
        # Convert to dictionary
        patient_dict = patient.model_dump()
        
        # Prepare features in correct order
        X = [patient_dict[f] for f in features]
        dpatient = xgb.DMatrix([X], feature_names=features)
        
        # Predict
        y_pred = float(model.predict(dpatient)[0])
        
        # Determine prediction and confidence
        malaria = y_pred >= 0.5
        
        if y_pred > 0.85 or y_pred < 0.15:
            confidence = "High"
        elif y_pred > 0.65 or y_pred < 0.35:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return {
            "malaria_probability": round(y_pred, 4),
            "malaria": malaria,
            "status": "Positive" if malaria else "Negative",
            "confidence": confidence
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9696)
