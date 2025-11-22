# Start from a stable, smaller Python base image
FROM python:3.10-slim-buster

# Set environment variables for Gunicorn
ENV PYTHONUNBUFFERED 1
ENV FLASK_APP wsgi.py

# Set the working directory
WORKDIR /usr/src/app

# Install system dependencies (for Python compilation and web service)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    # Clean up APT files to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for Docker cache efficiency)
COPY requirements.txt .

# Install Python dependencies (this is where numpy/scikit-learn get built)
# The stable 3.10 environment helps prevent the setuptools crash.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Gunicorn will listen on
EXPOSE 8080

# Command to run the application (Gunicorn)
# This is the equivalent of your Procfile
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "wsgi:app"]