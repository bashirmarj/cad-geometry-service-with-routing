# Use Miniconda base image for compatibility with pythonOCC
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy project files
COPY app.py .
COPY requirements.txt .

# Install dependencies from conda and pip
# - pythonocc-core must be installed via conda (not pip)
# - gunicorn, flask, and flask-cors installed via pip for version consistency
RRUN conda install -c conda-forge python=3.11 pythonocc-core numpy -y && \
    pip install --no-cache-dir -r requirements.txt && \
    conda clean -afy

# Expose Flask port
EXPOSE 5000

# Run app with Gunicorn (Render auto-sets $PORT)
CMD ["bash", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 app:app"]
