#!/bin/bash
set -e

echo "==> Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete! Activate your environment with:"
echo "  source venv/bin/activate"
