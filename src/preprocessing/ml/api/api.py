"""
FastAPI Backend for FraudShield
Real-time fraud detection API with <100ms latency
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import uvicorn
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import json
import os
from collections import deque
import asyncio

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


app = FastAPI(
    title="FraudShield API",
    description="Real-time Fraud Detection System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class Transaction(BaseModel):
    user_id: str = Field(..., description="User identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    timestamp: Optional[str] = Field(None, description="Transaction timestamp (ISO format)")
    category: str = Field(..., description="Transaction category")
    merchant_id: str = Field(..., description="Merchant identifier")
    device_id: Optional[str] = Field(None, description="Device identifier")
    ip_address: Optional[str] = Field(None, description="IP address")
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    email: Optional[str] = Field(None, description="User email")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "USER_001234",
                "amount": 250.00,
                "timestamp": "2024-02-13T14:30:00",
                "category": "electronics",
                "merchant_id": "MERCHANT_BestBuy_042",
                "device_id": "DEVICE_a1b2c3d4",
                "ip_address": "192.168.1.100",
                "latitude": 37.7749,
                "longitude": -122.4194,
                "email": "user_001234@gmail.com"
            }
        }


class FraudResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str  # 'low', 'medium', 'high'
    decision: str  # 'approve', 'review', 'decline'
    processing_time_ms: float
    explanation: Dict
    recommendations: List[str]


class BatchTransactionRequest(BaseModel):
    transactions: List[Transaction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class StatsResponse(BaseModel):
    total_transactions: int
    fraud_detected: int
    fraud_rate: float
    avg_processing_time_ms: float


# Global state
class FraudDetectionService:
    """Singleton service for fraud detection"""
    
    def __init__(self):
        self.ensemble_model = None
        self.feature_engineer = None
        self.feature_names = None
        self.is_loaded = False
        
        # Transaction history for velocity checks
        self.transaction_history = deque(maxlen=100000)
        
        # Statistics
        self.stats = {
            'total_transactions': 0,
            'fraud_detected': 0,
            'processing_times': []
        }
        
        # Cache for user profiles
        self.user_cache = {}
    
    def load_models(self, model_dir: str = "../models"):
        """Load trained models"""
        
        print("Loading models...")
        
        try:
            # Load ensemble model
            ensemble_path = os.path.join(model_dir, 'ensemble_model.joblib')
            model_data = joblib.load(ensemble_path)
            self.ensemble_model = model_data
            
            # Load feature names
            feature_path = os.path.join(model_dir, 'feature_names.json')
            with open(feature_path, 'r') as f:
                self.feature_names = json.load(f)
            
            # Initialize feature engineer
            from preprocessing.feature_engineering import AdvancedFeatureEngineer
            self.feature_engineer = AdvancedFeatureEngineer()
            
            self.is_loaded = True
            print("Models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_loaded = False
    
    async def predict(self, transaction: Transaction) -> FraudResponse:
        """Make fraud prediction"""
        
        if not self.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        start_time = datetime.now()
        
        # Convert to DataFrame
        txn_dict = transaction.dict()
        
        # Set timestamp if not provided
        if not txn_dict['timestamp']:
            txn_dict['timestamp'] = datetime.now().isoformat()
        
        # Generate transaction ID
        transaction_id = f"TXN_{int(datetime.now().timestamp()*1000):016d}"
        
        # Add to history
        self.transaction_history.append(txn_dict)
        
        # Create DataFrame with history for velocity features
        recent_txns = list(self.transaction_history)[-1000:]  # Last 1000 transactions
        df = pd.DataFrame(recent_txns)
        
        # Engineer features
        try:
            df_features = self.feature_engineer.engineer_features(df)
            
            # Get features for this transaction (last row)
            X = df_features[self.feature_names].iloc[-1:].fillna(0)
            
            # Make prediction
            fraud_prob = float(self.ensemble_model.predict_proba(X)[0])
            
            # Determine risk level and decision
            if fraud_prob >= 0.8:
                risk_level = 'high'
                decision = 'decline'
            elif fraud_prob >= 0.5:
                risk_level = 'medium'
                decision = 'review'
            else:
                risk_level = 'low'
                decision = 'approve'
            
            is_fraud = fraud_prob >= 0.5
            
            # Generate explanation
            explanation = self._generate_explanation(X, fraud_prob, df_features.iloc[-1])
            
            # Generate recommendations
            recommendations = self._generate_recommendations(risk_level, explanation)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Fallback to safe defaults
            fraud_prob = 0.5
            risk_level = 'medium'
            decision = 'review'
            is_fraud = False
            explanation = {'error': str(e)}
            recommendations = ['Manual review required']
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update statistics
        self.stats['total_transactions'] += 1
        if is_fraud:
            self.stats['fraud_detected'] += 1
        self.stats['processing_times'].append(processing_time)
        
        # Keep only last 1000 processing times
        if len(self.stats['processing_times']) > 1000:
            self.stats['processing_times'] = self.stats['processing_times'][-1000:]
        
        return FraudResponse(
            transaction_id=transaction_id,
            is_fraud=is_fraud,
            fraud_probability=round(fraud_prob, 4),
            risk_level=risk_level,
            decision=decision,
            processing_time_ms=round(processing_time, 2),
            explanation=explanation,
            recommendations=recommendations
        )
    
    def _generate_explanation(
        self,
        X: pd.DataFrame,
        fraud_prob: float,
        transaction_features: pd.Series
    ) -> Dict:
        """Generate explanation for prediction"""
        
        # Get top risky features
        risk_factors = []
        
        # Check for velocity anomalies
        if 'velocity_count_1h' in transaction_features:
            if transaction_features['velocity_count_1h'] > 5:
                risk_factors.append({
                    'factor': 'High transaction velocity',
                    'value': f"{transaction_features['velocity_count_1h']} transactions in 1 hour",
                    'impact': 'high'
                })
        
        # Check for unusual amount
        if 'amount' in transaction_features:
            if transaction_features.get('is_high_amount', 0) == 1:
                risk_factors.append({
                    'factor': 'Unusually high amount',
                    'value': f"${transaction_features['amount']:.2f}",
                    'impact': 'medium'
                })
        
        # Check for unusual time
        if transaction_features.get('is_night', 0) == 1:
            risk_factors.append({
                'factor': 'Late night transaction',
                'value': f"{transaction_features.get('hour', 0)}:00",
                'impact': 'low'
            })
        
        # Check for new merchant
        if transaction_features.get('is_new_merchant', 0) == 1:
            risk_factors.append({
                'factor': 'New merchant for this user',
                'value': 'First time',
                'impact': 'medium'
            })
        
        return {
            'fraud_probability': fraud_prob,
            'confidence': 'high' if abs(fraud_prob - 0.5) > 0.3 else 'medium',
            'risk_factors': risk_factors[:5],  # Top 5
            'total_factors_analyzed': len(self.feature_names)
        }
    
    def _generate_recommendations(
        self,
        risk_level: str,
        explanation: Dict
    ) -> List[str]:
        """Generate action recommendations"""
        
        recommendations = []
        
        if risk_level == 'high':
            recommendations.extend([
                "Decline transaction immediately",
                "Contact user via verified phone number",
                "Temporarily freeze card",
                "Monitor for additional suspicious activity"
            ])
        elif risk_level == 'medium':
            recommendations.extend([
                "Send verification SMS to user",
                "Request additional authentication",
                "Monitor closely for 24 hours",
                "Flag for manual review if pattern continues"
            ])
        else:
            recommendations.extend([
                "Approve transaction",
                "Continue normal monitoring"
            ])
        
        return recommendations
    
    def get_stats(self) -> StatsResponse:
        """Get service statistics"""
        
        fraud_rate = (
            self.stats['fraud_detected'] / self.stats['total_transactions']
            if self.stats['total_transactions'] > 0
            else 0
        )
        
        avg_time = (
            np.mean(self.stats['processing_times'])
            if self.stats['processing_times']
            else 0
        )
        
        return StatsResponse(
            total_transactions=self.stats['total_transactions'],
            fraud_detected=self.stats['fraud_detected'],
            fraud_rate=round(fraud_rate, 4),
            avg_processing_time_ms=round(avg_time, 2)
        )


# Initialize service
fraud_service = FraudDetectionService()


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    fraud_service.load_models()


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "service": "FraudShield API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/api/v1/predict",
            "batch": "/api/v1/predict/batch",
            "stats": "/api/v1/stats"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if fraud_service.is_loaded else "unhealthy",
        model_loaded=fraud_service.is_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/v1/predict", response_model=FraudResponse)
async def predict_fraud(transaction: Transaction):
    """
    Predict fraud for a single transaction
    
    Returns fraud probability and recommended action
    """
    return await fraud_service.predict(transaction)


@app.post("/api/v1/predict/batch", response_model=List[FraudResponse])
async def predict_fraud_batch(request: BatchTransactionRequest):
    """
    Predict fraud for multiple transactions
    
    Processes transactions concurrently for better performance
    """
    
    # Process transactions concurrently
    tasks = [fraud_service.predict(txn) for txn in request.transactions]
    results = await asyncio.gather(*tasks)
    
    return results


@app.get("/api/v1/stats", response_model=StatsResponse)
async def get_statistics():
    """Get service statistics"""
    return fraud_service.get_stats()


@app.post("/api/v1/reload")
async def reload_models():
    """Reload models from disk"""
    fraud_service.load_models()
    return {"status": "Models reloaded", "success": fraud_service.is_loaded}


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )