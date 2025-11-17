#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Run the command in the background to ensure environment variables are loaded
# This is the most reliable path to the gunicorn executable in Render's environment.
/usr/local/bin/python -m gunicorn web_app:app