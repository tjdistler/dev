#!/bin/bash

# Script to create and activate a Python virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        echo "Make sure Python 3 is installed and accessible as 'python3'."
        exit 1
    fi
    
    echo "Virtual environment created successfully!"
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo ""
echo "Virtual environment activated!"
echo "You can now install packages with: pip install <package>"
echo "To deactivate, run: deactivate"
echo ""

# Keep the shell active with the virtual environment
exec $SHELL

