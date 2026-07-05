import os
import time
import logging
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import mlflow
import dagshub

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_dagshub_and_mlflow() -> None:
    """
    Inisialisasi DagsHub dan konfigurasi MLflow Tracking.
    Script akan mengarahkan ke DagsHub, jika belum login akan muncul prompt autentikasi.
    """
    try:
        logging.info("Menginisialisasi DagsHub dan MLflow Tracking...")
        # Menggunakan nama repo dan owner berdasarkan project. 
        # Saat dijalankan, DagsHub akan meminta autentikasi otomatis di terminal/browser.
        dagshub.init(repo_owner='luckyprdn', repo_name='Eksperimen_SML_Lucky', mlflow=True)
        mlflow.set_experiment("Breast_Cancer_Base_Model")
        logging.info("DagsHub dan MLflow berhasil diinisialisasi.")
    except Exception as e:
        logging.error(f"Gagal menginisialisasi DagsHub/MLflow: {e}")
        raise

def load_preprocessed_data(data_dir: str = 'dataset') -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Memuat dataset yang sudah diproses dari folder dataset.
    
    Args:
        data_dir (str): Direktori tempat file CSV preprocessing disimpan.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    try:
        # Pengecekan path fallback jika dijalankan dari root repository
        if not os.path.exists(data_dir) and os.path.exists(f'../{data_dir}'):
            data_dir = f'../{data_dir}'
            
        logging.info(f"Memuat data dari direktori: {data_dir}")
        X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
        y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).squeeze("columns")
        y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).squeeze("columns")
        logging.info("Data berhasil dimuat.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Terjadi kesalahan saat memuat data: {e}")
        raise

def generate_and_save_artifacts(y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray, model: RandomForestClassifier, X_train: pd.DataFrame, artifact_dir: str = 'artifacts') -> None:
    """
    Menghasilkan plot dan teks untuk artefak MLflow (Confusion Matrix, ROC Curve, Feature Importance, dll).
    """
    os.makedirs(artifact_dir, exist_ok=True)
    logging.info(f"Menyimpan artefak ke folder {artifact_dir}...")

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'confusion_matrix.png'))
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'roc_curve.png'))
    plt.close()

    # 3. Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 10
    plt.figure(figsize=(8, 6))
    plt.title("Top 10 Feature Importances")
    plt.bar(range(top_n), importances[indices][:top_n], align="center")
    plt.xticks(range(top_n), X_train.columns[indices][:top_n], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, 'feature_importance.png'))
    plt.close()

    # 4. Classification Report (Text)
    report = classification_report(y_test, y_pred)
    with open(os.path.join(artifact_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # 5. Dataset Summary (Text)
    with open(os.path.join(artifact_dir, 'dataset_summary.txt'), 'w') as f:
        f.write(f"Training set size: {X_train.shape[0]} rows, {X_train.shape[1]} columns\n")
        f.write(f"Testing set size: {len(y_test)} rows\n")

def train_and_log_model() -> None:
    """
    Fungsi utama untuk training model RandomForestClassifier, mengevaluasi,
    dan mencatat (logging) semuanya ke MLflow secara manual.
    """
    setup_dagshub_and_mlflow()
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Parameter model
    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'random_state': 42
    }
    
    with mlflow.start_run(run_name="Base_RandomForest"):
        logging.info("Memulai proses training model...")
        
        # Log parameter
        mlflow.log_params(params)
        
        model = RandomForestClassifier(**params)
        
        # Hitung waktu training
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        logging.info("Proses training selesai. Memulai evaluasi...")
        
        # Prediksi
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Hitung Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'training_time_seconds': training_time
        }
        
        # Log Metrics
        mlflow.log_metrics(metrics)
        logging.info(f"Metrics: {metrics}")
        
        # Hitung Ukuran Model (Model Size)
        model_path = "temp_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        mlflow.log_metric('model_size_mb', model_size_mb)
        os.remove(model_path) # Bersihkan file sementara
        
        # Generate & Log Artifacts
        artifact_dir = 'artifacts'
        generate_and_save_artifacts(y_test, y_pred, y_prob, model, X_train, artifact_dir)
        mlflow.log_artifacts(artifact_dir)
        
        # Log Model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        logging.info(f"Semua data, metrik, dan artefak berhasil dilacak ke MLflow (Run ID: {mlflow.active_run().info.run_id})")

if __name__ == "__main__":
    train_and_log_model()