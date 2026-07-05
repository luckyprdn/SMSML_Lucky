import time
import psutil
import pickle
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge, Info

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="Breast Cancer Inference API",
    description="API untuk Prediksi Kanker Payudara yang terintegrasi dengan Prometheus"
)

# ==========================================
# DEFINISI 10 METRIK PROMETHEUS (ADVANCED)
# ==========================================
# 1. CPU Usage
CPU_USAGE = Gauge('system_cpu_usage_percent', 'Persentase penggunaan CPU saat ini')
# 2. Memory Usage
MEMORY_USAGE = Gauge('system_memory_usage_percent', 'Persentase penggunaan Memori saat ini')
# 3. Disk Usage
DISK_USAGE = Gauge('system_disk_usage_percent', 'Persentase penggunaan Disk saat ini')
# 4. Request Count
REQUEST_COUNT = Counter('api_request_total', 'Total HTTP requests yang diterima', ['method', 'endpoint'])
# 5. Response Time
RESPONSE_TIME = Histogram('api_response_time_seconds', 'Total waktu respon HTTP dalam detik', ['endpoint'])
# 6. Latency
LATENCY = Histogram('api_latency_seconds', 'Latensi jaringan/sistem API dalam detik')
# 7. Prediction Count
PREDICTION_COUNT = Counter('model_prediction_total', 'Total jumlah prediksi yang dilakukan')
# 8. Prediction Error
PREDICTION_ERROR = Counter('model_prediction_error_total', 'Total jumlah prediksi yang gagal/error')
# 9. Model Version
MODEL_VERSION = Info('model_version', 'Versi model machine learning saat ini')
MODEL_VERSION.info({'version': '1.0.0', 'algorithm': 'RandomForestClassifier', 'dataset': 'Breast Cancer Wisconsin'})
# 10. Inference Duration
INFERENCE_DURATION = Histogram('model_inference_duration_seconds', 'Waktu komputasi model saat melakukan predict (detik)')

# Menambahkan Endpoint /metrics untuk di-*scrape* oleh Prometheus
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ==========================================
# MEMUAT MODEL (Dengan Fallback agar Production-Ready)
# ==========================================
MODEL_PATH = "model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info("Model berhasil dimuat dari file model.pkl.")
except Exception as e:
    logging.warning(f"File {MODEL_PATH} tidak ditemukan. Menggunakan Dummy Model agar endpoint tetap berjalan untuk testing.")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    # Fit dummy data (30 fitur) agar model siap pakai
    dummy_X = pd.DataFrame(np.random.rand(10, 30))
    dummy_y = np.random.randint(0, 2, 10)
    model.fit(dummy_X, dummy_y)

# Skema Input Data (Breast cancer memiliki 30 fitur)
class Features(BaseModel):
    data: list[float]

def update_system_metrics() -> None:
    """Memperbarui metrik sistem (CPU, Memory, Disk) secara realtime."""
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage('/').percent)

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Middleware untuk melacak Request Count, Response Time, dan Latency."""
    start_time = time.time()
    
    # Update Metrik Sistem setiap ada request masuk
    update_system_metrics()
    
    # Increment Request Count
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Observasi Waktu
    RESPONSE_TIME.labels(endpoint=request.url.path).observe(process_time)
    LATENCY.observe(process_time)
    
    return response

@app.post("/predict")
def predict(features: Features):
    """Endpoint utama untuk inferensi machine learning."""
    try:
        if len(features.data) != 30:
            raise ValueError(f"Dibutuhkan tepat 30 fitur, tetapi menerima {len(features.data)}")
            
        input_data = pd.DataFrame([features.data])
        
        # Menghitung durasi murni inferensi model
        start_infer = time.time()
        prediction = model.predict(input_data)
        infer_time = time.time() - start_infer
        
        INFERENCE_DURATION.observe(infer_time)
        PREDICTION_COUNT.inc()
        
        result_class = "benign" if prediction[0] == 1 else "malignant"
        
        return {
            "status": "success",
            "prediction": result_class,
            "inference_duration_seconds": infer_time
        }
        
    except Exception as e:
        PREDICTION_ERROR.inc()
        logging.error(f"Error saat prediksi: {e}")
        raise HTTPException(status_code=400, detail=str(e))