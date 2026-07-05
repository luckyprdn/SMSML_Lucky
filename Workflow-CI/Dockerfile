# Menggunakan base image Python 3.12.7 slim untuk ukuran yang ringan dan production-ready
FROM python:3.12.7-slim

# Menentukan working directory di dalam container
WORKDIR /app

# Menginstal dependensi sistem dasar yang sering dibutuhkan library ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Menyalin seluruh file dari host (repository) ke dalam container
COPY . /app

# Menginstal dependensi Python yang sesuai dengan MLflow Project environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir mlflow scikit-learn pandas numpy dagshub

# Mengekspos port 5000 (Port standar MLflow)
EXPOSE 5000

# Command default untuk menjalankan MLflow project secara terisolasi tanpa virtualenv tambahan di dalam Docker
CMD ["mlflow", "run", ".", "--env-manager=local"]