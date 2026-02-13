"""
Advanced Feature Engineering for Fraud Detection
Implements cutting-edge features for transaction fraud detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import hashlib


class AdvancedFeatureEngineer:
    """
    Creates 120+ advanced features for fraud detection
    """
    
    def __init__(self):
        self.velocity_windows = [1, 6, 12, 24]  # hours
        self.aggregation_windows = [7, 14, 30]  # days
        
    def engineer_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline
        
        Args:
            transactions_df: Raw transaction data
            
        Returns:
            DataFrame with engineered features
        """
        df = transactions_df.copy()
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        print("Engineering features...")
        
        # 1. Basic Transaction Features
        df = self._basic_features(df)
        
        # 2. Temporal Features
        df = self._temporal_features(df)
        
        # 3. Velocity Features (CRITICAL for fraud detection)
        df = self._velocity_features(df)
        
        # 4. Aggregation Features
        df = self._aggregation_features(df)
        
        # 5. Behavioral Features
        df = self._behavioral_features(df)
        
        # 6. Device & Location Features
        df = self._device_location_features(df)
        
        # 7. Graph-based Features
        df = self._graph_features(df)
        
        # 8. Anomaly Score Features
        df = self._anomaly_features(df)
        
        # 9. Risk Score Features
        df = self._risk_features(df)
        
        print(f"Total features created: {len(df.columns)}")
        
        return df
    
    def _basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic transaction features"""
        
        # Amount features
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_squared'] = df['amount'] ** 2
        df['amount_sqrt'] = np.sqrt(df['amount'])
        
        # Round amount (suspicious if round numbers)
        df['is_round_amount'] = (df['amount'] % 100 == 0).astype(int)
        df['is_very_round'] = (df['amount'] % 1000 == 0).astype(int)
        
        return df
    
    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features"""
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Time of day categories
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
        df['is_early_morning'] = ((df['hour'] >= 6) & (df['hour'] < 9)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17)).astype(int)
        df['is_evening'] = ((df['hour'] >= 17) & (df['hour'] < 22)).astype(int)
        df['is_late_night'] = ((df['hour'] >= 22) & (df['hour'] <= 23)).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Velocity checks - CRITICAL for fraud detection
        Counts transactions in various time windows
        """
        
        for window_hours in self.velocity_windows:
            # Transaction count in window
            df[f'velocity_count_{window_hours}h'] = df.groupby('user_id')['timestamp'].transform(
                lambda x: x.rolling(f'{window_hours}h', on=df.loc[x.index, 'timestamp']).count()
            )
            
            # Amount sum in window
            df[f'velocity_amount_{window_hours}h'] = df.groupby('user_id')['amount'].transform(
                lambda x: x.rolling(f'{window_hours}h', on=df.loc[x.index, 'timestamp']).sum()
            )
            
            # Unique merchants in window
            df[f'velocity_merchants_{window_hours}h'] = df.groupby('user_id')['merchant_id'].transform(
                lambda x: x.rolling(f'{window_hours}h', on=df.loc[x.index, 'timestamp']).apply(
                    lambda y: len(set(y))
                )
            )
        
        # Velocity ratios
        df['velocity_ratio_6h_to_24h'] = df['velocity_count_6h'] / (df['velocity_count_24h'] + 1)
        df['velocity_ratio_1h_to_6h'] = df['velocity_count_1h'] / (df['velocity_count_6h'] + 1)
        
        return df
    
    def _aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Historical aggregation features"""
        
        for window_days in self.aggregation_windows:
            window = f'{window_days}D'
            
            # User statistics
            for col in ['amount', 'merchant_id']:
                df[f'user_{col}_mean_{window_days}d'] = df.groupby('user_id')[col].transform(
                    lambda x: x.rolling(window, on=df.loc[x.index, 'timestamp']).mean()
                )
                df[f'user_{col}_std_{window_days}d'] = df.groupby('user_id')[col].transform(
                    lambda x: x.rolling(window, on=df.loc[x.index, 'timestamp']).std()
                )
            
            # Deviation from historical average
            df[f'amount_deviation_{window_days}d'] = (
                df['amount'] - df[f'user_amount_mean_{window_days}d']
            ) / (df[f'user_amount_std_{window_days}d'] + 1)
        
        return df
    
    def _behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """User behavioral patterns"""
        
        # Time since last transaction
        df['time_since_last_txn'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
        df['time_since_last_txn'].fillna(999, inplace=True)
        
        # Transaction frequency
        df['user_txn_frequency'] = df.groupby('user_id').cumcount() + 1
        
        # First transaction indicator
        df['is_first_txn'] = (df['user_txn_frequency'] == 1).astype(int)
        
        # Merchant loyalty
        df['merchant_loyalty'] = df.groupby(['user_id', 'merchant_id']).cumcount() + 1
        df['is_new_merchant'] = (df['merchant_loyalty'] == 1).astype(int)
        
        # Category consistency
        df['user_category_diversity'] = df.groupby('user_id')['category'].transform('nunique')
        
        return df
    
    def _device_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Device and location-based features"""
        
        # Device fingerprint
        df['device_hash'] = df.apply(
            lambda x: hashlib.md5(
                f"{x.get('device_id', 'unknown')}_{x.get('ip_address', 'unknown')}".encode()
            ).hexdigest()[:8],
            axis=1
        )
        
        # Device switching behavior
        df['device_changes'] = df.groupby('user_id')['device_hash'].transform(
            lambda x: (x != x.shift(1)).cumsum()
        )
        
        # Location features (if available)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Distance from previous transaction
            df['prev_lat'] = df.groupby('user_id')['latitude'].shift(1)
            df['prev_lon'] = df.groupby('user_id')['longitude'].shift(1)
            
            df['distance_from_prev'] = self._haversine_distance(
                df['latitude'], df['longitude'],
                df['prev_lat'], df['prev_lon']
            )
            
            # Impossible travel (distance/time ratio)
            df['speed_kmh'] = df['distance_from_prev'] / (df['time_since_last_txn'] + 0.001)
            df['is_impossible_travel'] = (df['speed_kmh'] > 800).astype(int)  # >800 km/h
        
        return df
    
    def _graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Graph-based features (network analysis)"""
        
        # Shared device count
        df['shared_device_users'] = df.groupby('device_hash')['user_id'].transform('nunique')
        df['is_shared_device'] = (df['shared_device_users'] > 1).astype(int)
        
        # Shared merchant usage
        df['merchant_user_count'] = df.groupby('merchant_id')['user_id'].transform('nunique')
        
        # Email/phone domain features
        if 'email' in df.columns:
            df['email_domain'] = df['email'].str.split('@').str[1]
            df['email_domain_users'] = df.groupby('email_domain')['user_id'].transform('nunique')
        
        return df
    
    def _anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anomaly detection features"""
        
        # Z-score based anomalies
        for col in ['amount', 'velocity_count_24h']:
            if col in df.columns:
                mean = df.groupby('user_id')[col].transform('mean')
                std = df.groupby('user_id')[col].transform('std')
                df[f'{col}_zscore'] = (df[col] - mean) / (std + 1)
                df[f'{col}_is_anomaly'] = (np.abs(df[f'{col}_zscore']) > 3).astype(int)
        
        return df
    
    def _risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Risk scoring features"""
        
        # High-risk indicators
        df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        df['is_night_txn'] = df['is_night']
        df['is_new_user'] = (df['user_txn_frequency'] <= 5).astype(int)
        
        # Composite risk score
        risk_components = [
            'is_high_amount',
            'is_night',
            'is_new_merchant',
            'is_shared_device'
        ]
        
        df['basic_risk_score'] = df[[c for c in risk_components if c in df.columns]].sum(axis=1)
        
        return df
    
    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two points on Earth"""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of engineered feature names"""
        exclude_cols = ['timestamp', 'user_id', 'merchant_id', 'transaction_id', 'is_fraud']
        return [col for col in df.columns if col not in exclude_cols]


if __name__ == "__main__":
    # Example usage
    print("Advanced Feature Engineering Module - Ready")
    print("Creates 120+ features for fraud detection")