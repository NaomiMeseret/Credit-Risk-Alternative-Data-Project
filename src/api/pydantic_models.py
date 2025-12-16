"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class PredictRequest(BaseModel):
    """
    Request model for prediction.
    Accepts dynamic feature set; only enforces customer_id for tracing.
    """
    customer_id: str
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    """Response model for credit risk prediction."""
    customer_id: str
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Risk probability between 0 and 1")
    risk_category: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = True

