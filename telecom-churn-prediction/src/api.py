from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title='Churn Prediction API')
MODEL_PATH = 'model/churn_model.joblib'

class InputData(BaseModel):
    data: dict

@app.on_event('startup')
def load_model():
    global model
    model = None
    try:
        model = joblib.load(MODEL_PATH)
        print('Model loaded from', MODEL_PATH)
    except Exception as e:
        print('Model not found:', e)

@app.post('/predict')
def predict(payload: InputData):
    df = pd.DataFrame([payload.data])
    if model is None:
        return {'error': 'Model not loaded. Train first with src/train_model.py'}
    pred = model.predict(df)[0]
    prob = model.predict_proba(df).max().item() if hasattr(model, 'predict_proba') else None
    return {'prediction': int(pred), 'probability': prob}
