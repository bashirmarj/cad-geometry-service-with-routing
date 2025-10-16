
# ---------- CAD Geometry Service (Render Optimized) ----------
FROM python:3.10-slim-bookworm

WORKDIR /app
COPY requirements.txt .
COPY app.py .

# Install OpenCascade runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglu1-mesa libxrender1 libxext6 libsm6 && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages (OCC 7.7.2 works with Py3.10+)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy==1.26.4 pythonocc-core==7.7.2 && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]
