import os
import json
import logging
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_ci_training():
    """
    Fungsi training khusus untuk CI/CD.
    Fokus utama adalah melatih model dan menyimpan artefak model ke local MLflow 
    agar GitHub Actions bisa mem-build Docker imagenya.
    """
    logging.info("Memulai proses training pipeline CI/CD...")
    
    # Set tracking uri ke lokal folder ./mlruns agar tidak butuh autentikasi cloud
    mlflow.set_tracking_uri("file://" + os.path.abspath("./mlruns"))
    mlflow.set_experiment("Breast_Cancer_CI_Pipeline")

    # Muat dataset secara mandiri untuk menghindari dependensi folder lintas repo di CI
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    with mlflow.start_run() as run:
        # Training Model
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluasi Singkat
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Log Metrics & Model
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(rf, "model")
        
        run_id = run.info.run_id
        logging.info(f"Training CI/CD Selesai. Accuracy: {acc:.4f} | Run ID: {run_id}")
        
        # MENYIMPAN RUN ID ke file text agar bisa dibaca oleh GitHub Actions
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        logging.info("Run ID berhasil disimpan ke run_id.txt")

if __name__ == "__main__":
    run_ci_training()