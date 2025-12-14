"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class TransactionRequest(BaseModel):
    """Request model for transaction data."""
    transaction_id: str
    customer_id: str
    amount: float
    # Additional fields will be added based on model requirements


class CreditRiskPrediction(BaseModel):
    """Response model for credit risk prediction."""
    customer_id: str
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Risk probability between 0 and 1")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score between 300 and 850")
    risk_category: str = Field(..., description="Risk category: 'low', 'medium', or 'high'")


class LoanRecommendation(BaseModel):
    """Response model for loan amount and duration recommendation."""
    customer_id: str
    recommended_amount: float
    recommended_duration_months: int
    confidence: float

