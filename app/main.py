from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Initialize the API
app = FastAPI(title="Genomic Treatment Recommender API - v2.0")

# Define base directory to locate models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models")

# Load the trained artifacts
try:
    model = joblib.load(os.path.join(MODEL_PATH, 'xgboost_model.pkl'))
    pca = joblib.load(os.path.join(MODEL_PATH, 'pca_transformer.pkl'))
    scaler = joblib.load(os.path.join(MODEL_PATH, 'standard_scaler.pkl'))
    print("Models and Transformers loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")

# Define the input data schema
class PatientData(BaseModel):
    # Expecting 7 values: [PatientID, Age, Gender, Gene_X, Gene_Y, Smoking, Disease]
    gene_expression_values: list 

@app.get("/")
def home():
    return {
        "status": "Online",
        "project": "Precision Medicine & Genomics",
        "description": "3-Class Risk Stratification for Personalized Treatment"
    }

@app.post("/predict")
def predict_risk(data: PatientData):
    try:
        # 1. Validation: Ensure we have exactly 7 features
        if len(data.gene_expression_values) != 7:
            raise ValueError(f"Expected 7 features, got {len(data.gene_expression_values)}")

        # 2. Preprocessing
        raw_input = np.array(data.gene_expression_values).reshape(1, -1)
        scaled_input = scaler.transform(raw_input)
        reduced_input = pca.transform(scaled_input)
        
        # 3. Inference
        prediction = int(model.predict(reduced_input)[0])
        probabilities = model.predict_proba(reduced_input)[0]
        
        # 4. Clinical Mapping (Based on Dataset Analysis)
        # Class 0: Senior, Heavy Smoker, High Gene Expression -> High Risk
        # Class 1: Middle-aged, Occasional Smoker, Moderate Genes -> Moderate Risk
        # Class 2: Younger, Non-smoker, Low Genes -> Low Risk
        mapping = {
            0: {
                "label": "High Risk", 
                "guidance": "Personalized dosage adjustment required. Consider alternative therapies."
            },
            1: {
                "label": "Moderate Risk", 
                "guidance": "Standard dosage with close clinical monitoring recommended."
            },
            2: {
                "label": "Low Risk", 
                "guidance": "Standard protocol. High probability of treatment success."
            }
        }
        
        result = mapping.get(prediction, {"label": "Unknown", "guidance": "Consult specialist."})
        
        return {
            "prediction_summary": {
                "class_id": prediction,
                "confidence_score": round(float(probabilities[prediction]), 4),
                "risk_category": result["label"]
            },
            "clinical_decision_support": result["guidance"],
            "all_class_probabilities": {
                "High Risk (0)": round(float(probabilities[0]), 4),
                "Moderate Risk (1)": round(float(probabilities[1]), 4),
                "Low Risk (2)": round(float(probabilities[2]), 4)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
