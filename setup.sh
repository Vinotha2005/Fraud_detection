#!/bin/bash

# FraudShield Quick Start Script
# Automates the entire setup process

set -e  # Exit on error

echo "=========================================="
echo "🛡️  FraudShield Setup Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}[1/8]${NC} Checking Python version..."
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"
else
    echo -e "${RED}✗${NC} Python 3.10+ required but not found"
    exit 1
fi

# Create virtual environment
echo -e "${BLUE}[2/8]${NC} Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment already exists"
fi

# Activate virtual environment
echo -e "${BLUE}[3/8]${NC} Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Install Python dependencies
echo -e "${BLUE}[4/8]${NC} Installing Python dependencies..."
echo "This may take a few minutes..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Dependencies installed"

# Create necessary directories
echo -e "${BLUE}[5/8]${NC} Creating project directories..."
mkdir -p data models logs
echo -e "${GREEN}✓${NC} Directories created"

# Generate synthetic data
echo -e "${BLUE}[6/8]${NC} Generating synthetic transaction data..."
cd src
python3 << 'EOF'
from utils.data_generator import load_or_generate_data
import os

print("  Generating 100,000 transactions...")
df = load_or_generate_data(
    filepath='../data/fraud_transactions.csv',
    num_transactions=100000,
    fraud_ratio=0.02
)
print(f"  ✓ Generated {len(df)} transactions")
print(f"  ✓ Fraud cases: {df['is_fraud'].sum()}")
print(f"  ✓ Saved to ../data/fraud_transactions.csv")
EOF
cd ..
echo -e "${GREEN}✓${NC} Data generation complete"

# Train models
echo -e "${BLUE}[7/8]${NC} Training machine learning models..."
echo "This will take 5-10 minutes..."
cd src
python3 train_model.py
cd ..
echo -e "${GREEN}✓${NC} Model training complete"

# Verify installation
echo -e "${BLUE}[8/8]${NC} Verifying installation..."
if [ -f "models/ensemble_model.joblib" ]; then
    echo -e "${GREEN}✓${NC} Models saved successfully"
else
    echo -e "${RED}✗${NC} Model training may have failed"
    exit 1
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start the API server:"
echo -e "   ${BLUE}cd src/api && python api.py${NC}"
echo ""
echo "2. In a new terminal, start the frontend:"
echo -e "   ${BLUE}cd frontend && npm install && npm start${NC}"
echo ""
echo "3. Open your browser to:"
echo -e "   ${BLUE}http://localhost:3000${NC} (Frontend Dashboard)"
echo -e "   ${BLUE}http://localhost:8000/docs${NC} (API Documentation)"
echo ""
echo "=========================================="
echo ""
echo "For more information, see README.md"
echo ""