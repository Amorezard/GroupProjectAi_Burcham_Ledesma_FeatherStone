#!/bin/bash
# Setup script for Merrimack College Wayfinding AI

echo "Setting up Merrimack College Wayfinding AI environment..."

# Check for Python
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "Error: Python not found. Please install Python 3.10 or later."
    exit 1
fi

# Check Python version
PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using Python version $PY_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment. Please install venv package."
        exit 1
    fi
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Unix/Linux/macOS
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies."
    exit 1
fi

# Test OSMnx installation
echo "Testing OSMnx installation..."
python test_osmnx.py
if [ $? -ne 0 ]; then
    echo "Warning: OSMnx test failed. Please check the error messages above."
    echo "You may need to manually install osmnx and its dependencies."
    echo "Try: pip install osmnx geopandas networkx"
else
    echo "OSMnx test successful!"
fi

echo ""
echo "Setup complete! To run the application:"
echo "1. Activate the virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   source venv/Scripts/activate"
else
    echo "   source venv/bin/activate"
fi
echo "2. Run the application:"
echo "   python app.py"
echo ""
echo "Open http://localhost:5000 in your browser to use the application." 