"""
Autoencoder-based Anomaly Detection
Unsupervised learning to detect unusual transaction patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import matplotlib.pyplot as plt


class FraudAutoencoder(nn.Module):
    """
    Deep Autoencoder for anomaly detection
    
    Architecture:
    Encoder: Input -> 128 -> 64 -> 32 (bottleneck)
    Decoder: 32 -> 64 -> 128 -> Output
    
    Fraud transactions have higher reconstruction error
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super(FraudAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Get latent representation"""
        return self.encoder(x)


class AnomalyDetector:
    """
    Wrapper for autoencoder-based anomaly detection
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        contamination: float = 0.01
    ):
        self.model = FraudAutoencoder(input_dim, latent_dim)
        self.scaler = StandardScaler()
        self.threshold = None
        self.contamination = contamination
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(
        self,
        X_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 0.001
    ):
        """
        Train autoencoder on normal (non-fraud) transactions
        
        Args:
            X_train: Training data (preferably only normal transactions)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        losses = []
        
        print(f"Training Autoencoder on {self.device}...")
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in dataloader:
                x = batch[0]
                
                # Forward pass
                reconstructed = self.model(x)
                loss = criterion(reconstructed, x)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        # Calculate threshold based on reconstruction error
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        
        # Set threshold at contamination percentile
        self.threshold = np.percentile(errors, (1 - self.contamination) * 100)
        
        print(f"\nTraining complete!")
        print(f"Anomaly threshold set at: {self.threshold:.6f}")
        
        return losses
    
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly score (reconstruction error)
        
        Args:
            X: Input data
        
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        
        return errors
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (binary)
        
        Returns:
            1 for anomaly, 0 for normal
        """
        
        scores = self.predict_anomaly_score(X)
        return (scores > self.threshold).astype(int)
    
    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """
        Get latent space representation
        Useful for visualization and clustering
        """
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            latent = self.model.encode(X_tensor).cpu().numpy()
        
        return latent


class IsolationForestDetector:
    """
    Isolation Forest for anomaly detection
    Fast and effective for high-dimensional data
    """
    
    def __init__(self, contamination: float = 0.01):
        from sklearn.ensemble import IsolationForest
        
        self.model = IsolationForest(
            contamination=contamination,
            max_samples=256,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
    
    def train(self, X_train: np.ndarray):
        """Train Isolation Forest"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)
        print("Isolation Forest training complete")
    
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores
        Note: Isolation Forest returns negative scores
        We invert them so higher = more anomalous
        """
        X_scaled = self.scaler.transform(X)
        scores = -self.model.score_samples(X_scaled)
        return scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (-1 for anomaly, 1 for normal)"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Convert to 0/1
        return (predictions == -1).astype(int)


class HybridAnomalyDetector:
    """
    Combines Autoencoder and Isolation Forest
    for robust anomaly detection
    """
    
    def __init__(
        self,
        input_dim: int,
        ae_weight: float = 0.6,
        contamination: float = 0.01
    ):
        self.ae_detector = AnomalyDetector(input_dim, contamination=contamination)
        self.if_detector = IsolationForestDetector(contamination=contamination)
        self.ae_weight = ae_weight
        self.if_weight = 1 - ae_weight
    
    def train(self, X_train: np.ndarray, epochs: int = 50):
        """Train both detectors"""
        
        print("\n" + "="*50)
        print("Training Autoencoder...")
        print("="*50)
        self.ae_detector.train(X_train, epochs=epochs)
        
        print("\n" + "="*50)
        print("Training Isolation Forest...")
        print("="*50)
        self.if_detector.train(X_train)
    
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Ensemble anomaly score"""
        
        # Get scores from both models
        ae_scores = self.ae_detector.predict_anomaly_score(X)
        if_scores = self.if_detector.predict_anomaly_score(X)
        
        # Normalize scores to [0, 1]
        ae_scores_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-10)
        if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
        
        # Weighted ensemble
        ensemble_scores = (
            self.ae_weight * ae_scores_norm +
            self.if_weight * if_scores_norm
        )
        
        return ensemble_scores
    
    def predict(self, X: np.ndarray, threshold: float = 0.7) -> np.ndarray:
        """Predict anomalies using ensemble score"""
        scores = self.predict_anomaly_score(X)
        return (scores > threshold).astype(int)


if __name__ == "__main__":
    print("Anomaly Detection Module - Ready")
    print("Autoencoder + Isolation Forest for unsupervised fraud detection")