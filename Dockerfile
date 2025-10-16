FROM python:3.10-slim-bookworm

WORKDIR /app

COPY requirements.txt .
COPY app.py .

# Install system dependencies for pythonocc-core (Debian 13+ compatible)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglu1-mesa libxrender1 libxext6 libsm6 && \
    rm -rf /var/lib/apt/lists/*

# Install everything with pip (lighter than conda)
RUN pip install --no-cache-dir numpy==1.26.4 pythonocc-core==7.6.3 && \
    pip install --no-cache-dir -r requirements.txt
