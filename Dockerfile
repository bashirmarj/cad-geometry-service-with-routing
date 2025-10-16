# ---------- CAD Geometry Service (Render + OCC binaries) ----------
FROM continuumio/miniconda3:24.7.1-0

WORKDIR /app
COPY requirements.txt .
COPY app.py .

# Install pythonocc-core and numpy from conda-forge (precompiled)
RUN conda install -y -c conda-forge pythonocc-core=7.7.2 numpy && \
    pip install --no-cache-dir -r requirements.txt && \
    conda clean --all --yes

EXPOSE 5000
CMD ["python", "app.py"]
