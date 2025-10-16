FROM continuumio/miniconda3:latest

WORKDIR /app
COPY requirements.txt .
COPY app.py .

RUN conda install -y -c conda-forge pythonocc-core=7.6.3 numpy && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]
