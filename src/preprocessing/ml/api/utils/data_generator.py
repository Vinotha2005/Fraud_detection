"""
Synthetic Fraud Transaction Data Generator
Creates realistic transaction data with fraud patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Tuple
import hashlib


class FraudDataGenerator:
    """
    Generates realistic synthetic fraud transaction data
    
    Fraud patterns included:
    - Card testing (small amounts)
    - Velocity attacks (multiple transactions)
    - Geographic impossibility
    - Unusual time patterns
    - Round amounts
    - New merchant fraud
    """
    
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Realistic categories
        self.categories = [
            'grocery', 'electronics', 'clothing', 'restaurant',
            'gas_station', 'online_shopping', 'travel', 'entertainment',
            'healthcare', 'utilities', 'insurance', 'education'
        ]
        
        # Common merchant names
        self.merchants = [
            'Amazon', 'Walmart', 'Target', 'Costco', 'Best Buy',
            'Starbucks', 'McDonalds', 'Shell', 'Chevron', 'Apple',
            'Nike', 'Adidas', 'Zara', 'H&M', 'Home Depot',
            'CVS', 'Walgreens', 'Netflix', 'Spotify', 'Uber'
        ]
        
        # Device types
        self.devices = ['iPhone', 'Android', 'Desktop', 'Tablet']
        
    def generate_data(
        self,
        num_transactions: int = 100000,
        fraud_ratio: float = 0.02,
        num_users: int = 10000
    ) -> pd.DataFrame:
        """
        Generate synthetic transaction data
        
        Args:
            num_transactions: Total number of transactions
            fraud_ratio: Percentage of fraudulent transactions
            num_users: Number of unique users
        
        Returns:
            DataFrame with transactions
        """
        
        print(f"Generating {num_transactions} transactions...")
        print(f"Fraud ratio: {fraud_ratio*100:.1f}%")
        
        transactions = []
        num_fraud = int(num_transactions * fraud_ratio)
        num_legit = num_transactions - num_fraud
        
        # Generate legitimate transactions
        print("Creating legitimate transactions...")
        legit_txns = self._generate_legitimate_transactions(num_legit, num_users)
        transactions.extend(legit_txns)
        
        # Generate fraudulent transactions
        print("Creating fraudulent transactions...")
        fraud_txns = self._generate_fraudulent_transactions(num_fraud, num_users)
        transactions.extend(fraud_txns)
        
        # Create DataFrame
        df = pd.DataFrame(transactions)
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Add transaction IDs
        df['transaction_id'] = [f"TXN_{i:08d}" for i in range(len(df))]
        
        print(f"\nGeneration complete!")
        print(f"Total transactions: {len(df)}")
        print(f"Legitimate: {len(df[df['is_fraud']==0])}")
        print(f"Fraudulent: {len(df[df['is_fraud']==1])}")
        
        return df
    
    def _generate_legitimate_transactions(
        self,
        num_transactions: int,
        num_users: int
    ) -> list:
        """Generate realistic legitimate transactions"""
        
        transactions = []
        start_date = datetime.now() - timedelta(days=90)
        
        for _ in range(num_transactions):
            user_id = f"USER_{random.randint(1, num_users):06d}"
            
            # Realistic amount distribution
            amount = self._generate_normal_amount()
            
            # Realistic timestamp (business hours more likely)
            timestamp = self._generate_normal_timestamp(start_date)
            
            # Select merchant and category
            category = random.choice(self.categories)
            merchant_id = f"MERCHANT_{random.choice(self.merchants)}_{random.randint(1,100):03d}"
            
            # Device info
            device_id = self._generate_device_id(user_id)
            ip_address = self._generate_ip_address()
            
            # Location (consistent for each user)
            latitude, longitude = self._generate_location(user_id)
            
            # Email domain
            email = f"{user_id.lower()}@gmail.com"
            
            transactions.append({
                'user_id': user_id,
                'amount': round(amount, 2),
                'timestamp': timestamp,
                'category': category,
                'merchant_id': merchant_id,
                'device_id': device_id,
                'ip_address': ip_address,
                'latitude': latitude,
                'longitude': longitude,
                'email': email,
                'is_fraud': 0
            })
        
        return transactions
    
    def _generate_fraudulent_transactions(
        self,
        num_transactions: int,
        num_users: int
    ) -> list:
        """Generate fraudulent transactions with various fraud patterns"""
        
        transactions = []
        start_date = datetime.now() - timedelta(days=90)
        
        fraud_types = [
            'card_testing',
            'velocity_attack',
            'geographic_impossible',
            'unusual_time',
            'large_amount',
            'round_amount'
        ]
        
        for _ in range(num_transactions):
            fraud_type = random.choice(fraud_types)
            
            user_id = f"USER_{random.randint(1, num_users):06d}"
            
            if fraud_type == 'card_testing':
                # Small transactions to test card validity
                amount = round(random.uniform(1, 10), 2)
                timestamp = datetime.now() - timedelta(
                    days=random.randint(0, 90),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
            
            elif fraud_type == 'velocity_attack':
                # Multiple rapid transactions
                amount = round(random.uniform(50, 500), 2)
                base_time = datetime.now() - timedelta(days=random.randint(0, 90))
                timestamp = base_time + timedelta(minutes=random.randint(0, 30))
            
            elif fraud_type == 'geographic_impossible':
                # Transactions from impossible locations
                amount = round(random.uniform(100, 1000), 2)
                timestamp = self._generate_normal_timestamp(start_date)
                # Will set unusual location
            
            elif fraud_type == 'unusual_time':
                # Late night / early morning transactions
                amount = round(random.uniform(100, 800), 2)
                base_date = datetime.now() - timedelta(days=random.randint(0, 90))
                # 2 AM - 5 AM
                timestamp = base_date.replace(
                    hour=random.randint(2, 5),
                    minute=random.randint(0, 59)
                )
            
            elif fraud_type == 'large_amount':
                # Unusually large amounts
                amount = round(random.uniform(2000, 10000), 2)
                timestamp = self._generate_normal_timestamp(start_date)
            
            else:  # round_amount
                # Suspiciously round amounts
                amount = random.choice([100, 200, 500, 1000, 2000, 5000])
                timestamp = self._generate_normal_timestamp(start_date)
            
            # Random category and merchant
            category = random.choice(self.categories)
            merchant_id = f"MERCHANT_{random.choice(self.merchants)}_{random.randint(1,100):03d}"
            
            # Often different device for fraud
            device_id = self._generate_device_id(user_id, is_fraud=True)
            ip_address = self._generate_ip_address()
            
            # Potentially unusual location
            if fraud_type == 'geographic_impossible':
                latitude = random.uniform(-90, 90)
                longitude = random.uniform(-180, 180)
            else:
                latitude, longitude = self._generate_location(user_id)
            
            email = f"{user_id.lower()}@gmail.com"
            
            transactions.append({
                'user_id': user_id,
                'amount': amount,
                'timestamp': timestamp,
                'category': category,
                'merchant_id': merchant_id,
                'device_id': device_id,
                'ip_address': ip_address,
                'latitude': latitude,
                'longitude': longitude,
                'email': email,
                'is_fraud': 1
            })
        
        return transactions
    
    def _generate_normal_amount(self) -> float:
        """Generate realistic transaction amount"""
        # Log-normal distribution (most small, few large)
        amount = np.random.lognormal(mean=4, sigma=1)
        return min(amount, 5000)  # Cap at $5000
    
    def _generate_normal_timestamp(self, start_date: datetime) -> datetime:
        """Generate realistic timestamp (business hours more likely)"""
        
        days_offset = random.randint(0, 90)
        
        # Weight towards business hours
        hour_weights = [1]*24
        for h in range(9, 18):  # 9 AM - 6 PM
            hour_weights[h] = 3
        
        hour = random.choices(range(24), weights=hour_weights)[0]
        minute = random.randint(0, 59)
        
        timestamp = start_date + timedelta(
            days=days_offset,
            hours=hour,
            minutes=minute
        )
        
        return timestamp
    
    def _generate_device_id(self, user_id: str, is_fraud: bool = False) -> str:
        """Generate device ID"""
        
        if is_fraud and random.random() < 0.3:
            # Fraudulent transactions sometimes use different devices
            return f"DEVICE_{hashlib.md5(f'fraud_{random.random()}'.encode()).hexdigest()[:8]}"
        
        # Consistent device for user
        return f"DEVICE_{hashlib.md5(user_id.encode()).hexdigest()[:8]}"
    
    def _generate_ip_address(self) -> str:
        """Generate random IP address"""
        return f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}"
    
    def _generate_location(self, user_id: str) -> Tuple[float, float]:
        """Generate consistent location for a user"""
        
        # Hash user_id to get consistent location
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        
        # US coordinates roughly
        latitude = 25 + (hash_val % 1000) / 1000 * 25  # 25 to 50
        longitude = -125 + (hash_val % 1000) / 1000 * 55  # -125 to -70
        
        return round(latitude, 6), round(longitude, 6)


def load_or_generate_data(
    filepath: str = None,
    num_transactions: int = 100000,
    fraud_ratio: float = 0.02
) -> pd.DataFrame:
    """
    Load existing data or generate new data
    
    Args:
        filepath: Path to existing CSV (if exists)
        num_transactions: Number of transactions to generate
        fraud_ratio: Fraud percentage
    
    Returns:
        DataFrame with transaction data
    """
    
    import os
    
    if filepath and os.path.exists(filepath):
        print(f"Loading existing data from {filepath}...")
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Loaded {len(df)} transactions")
        return df
    
    # Generate new data
    generator = FraudDataGenerator()
    df = generator.generate_data(
        num_transactions=num_transactions,
        fraud_ratio=fraud_ratio
    )
    
    # Save if filepath provided
    if filepath:
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    
    return df


if __name__ == "__main__":
    # Generate sample data
    df = load_or_generate_data(
        filepath="fraud_transactions.csv",
        num_transactions=100000,
        fraud_ratio=0.02
    )
    
    print("\nData sample:")
    print(df.head())
    print("\nFraud distribution:")
    print(df['is_fraud'].value_counts())