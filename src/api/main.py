"""
FastAPI application for credit risk model inference.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Probability Model API",
    description="API for predicting credit risk using alternative data",
    version="0.1.0"
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Credit Risk Probability Model API"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Additional endpoints will be added as the model is developed

