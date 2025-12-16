"""
FastAPI application for credit risk model inference.
Loads best model (MLflow URI or local artifact) and exposes /predict.
"""

import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import Optional

from src.api.pydantic_models import PredictRequest, PredictResponse, HealthResponse

MODEL_URI_ENV = "MODEL_URI"  # e.g., "models:/credit-risk-model/Production" or local path
DEFAULT_LOCAL_MODEL = "./models/best_model"

app = FastAPI(
    title="Credit Risk Probability Model API",
    description="API for predicting credit risk using alternative data",
    version="0.2.0",
)


class ModelService:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.model_version = None
        self._load_model()

    def _load_model(self):
        uri = os.getenv(MODEL_URI_ENV, DEFAULT_LOCAL_MODEL)
        try:
            self.model = mlflow.sklearn.load_model(uri)
            if uri.startswith("models:/"):
                parts = uri.split("/")
                if len(parts) >= 3:
                    self.model_name = parts[1]
                    self.model_version = parts[2]
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {uri}: {e}")

    def predict(self, payload: PredictRequest) -> PredictResponse:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if not payload.features:
            raise ValueError("No features provided")

        X = pd.DataFrame([payload.features])
        try:
            if hasattr(self.model, "predict_proba"):
                prob = self.model.predict_proba(X)[:, 1][0]
            else:
                prob = float(self.model.predict(X)[0])
        except Exception as e:
            raise ValueError(f"Inference failed: {e}")

        risk_category = "high" if prob >= 0.5 else "low"
        return PredictResponse(
            customer_id=payload.customer_id,
            risk_probability=float(prob),
            risk_category=risk_category,
            model_name=self.model_name,
            model_version=self.model_version,
        )


model_service = ModelService()


@app.get("/", response_model=dict)
async def root():
    return {"message": "Credit Risk Probability Model API"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", model_loaded=model_service.model is not None)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        return model_service.predict(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
