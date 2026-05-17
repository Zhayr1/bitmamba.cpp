#!/bin/bash
# Setup script for BitMamba.cpp API server

set -e

echo "🚀 Setting up BitMamba.cpp API Server..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is required but not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment (optional)
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Python dependencies installed"

# Check if C++ executable is built
if [ ! -f "../build/bitmamba" ]; then
    echo "⚠️  C++ executable not found. Building..."
    cd ..
    cmake -B build
    cmake --build build
    cd python
fi

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Download model weights:"
echo "   wget https://huggingface.co/Zhayr1/BitMamba-2-1B/resolve/main/bitmamba_cpp/bitmamba_1b.bin"
echo "   wget https://huggingface.co/Zhayr1/BitMamba-2-1B/resolve/main/bitmamba_cpp/tokenizer.bin"
echo ""
echo "2. Start the server:"
echo "   python server.py --model ../bitmamba_1b.bin"
echo ""
echo "3. Test the API:"
echo "   python client.py --prompt \"Hello, how are you?\""
echo ""
echo "📚 For more details, see python/README.md"