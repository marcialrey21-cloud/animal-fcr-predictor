#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Find the virtual environment activate script (it's often named 'activate.sh' on Render)
ENV_PATH=$(find /opt/render/project/src/ -name activate | head -n 1)

if [ -f "$ENV_PATH" ]; then
    # Source the environment script to load gunicorn into the PATH
    source "$ENV_PATH"
    echo "Sourced environment from $ENV_PATH"
else
    echo "Warning: Could not find virtual environment activation script. Using default gunicorn path."
fi

# Run the application using the simple command, now that the PATH is set
echo "Starting Gunicorn server..."
exec gunicorn web_app:app