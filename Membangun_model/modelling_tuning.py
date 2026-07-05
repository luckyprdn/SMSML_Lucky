import os
import time
import logging
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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
    """
    try:
        logging.info("Menginisialisasi DagsHub dan MLflow Tracking...")
        dagshub.init(repo_owner='luckyprdn', repo_name='Eksperimen_SML_Lucky', mlflow=True)
        mlflow.set_experiment("Breast_Cancer_Tuned_Model")
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

def generate_and_save_artifacts(y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray, model: RandomForestClassifier, X_train: pd.DataFrame, artifact_dir: str = 'artifacts_tuning') -> None:
    """
    Menghasilkan plot dan teks untuk artefak MLflow secara manual.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    logging.info(f"Menyimpan artefak ke folder {artifact_dir}...")

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Tuned Model)')
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
    plt.title('ROC Curve (Tuned Model)')
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
    plt.title("Top 10 Feature Importances (Tuned Model)")
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

def tune_and_log_model() -> None:
    """
    Fungsi utama untuk melakukan Hyperparameter Tuning dengan GridSearchCV,
    mengevaluasi model terbaik, dan mencatat semuanya ke MLflow secara manual.
    """
    setup_dagshub_and_mlflow()
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Menentukan Parameter Grid untuk Tuning
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    
    with mlflow.start_run(run_name="Tuned_RandomForest"):
        logging.info("Memulai proses Hyperparameter Tuning (GridSearchCV)...")
        
        # Hitung waktu training tuning
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logging.info(f"Tuning selesai. Parameter terbaik: {best_params}")
        
        # Log Best Parameters
        mlflow.log_params(best_params)
        
        # Prediksi menggunakan model terbaik
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Hitung Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'tuning_and_training_time_seconds': tuning_time
        }
        
        # Log Metrics
        mlflow.log_metrics(metrics)
        logging.info(f"Metrics (Tuned Model): {metrics}")
        
        # Hitung Ukuran Model (Model Size)
        model_path = "temp_tuned_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        mlflow.log_metric('model_size_mb', model_size_mb)
        os.remove(model_path)
        
        # Generate & Log Artifacts
        artifact_dir = 'artifacts_tuning'
        generate_and_save_artifacts(y_test, y_pred, y_prob, best_model, X_train, artifact_dir)
        mlflow.log_artifacts(artifact_dir)
        
        # Log Model
        mlflow.sklearn.log_model(best_model, "tuned_random_forest_model")
        
        logging.info(f"Model hasil tuning beserta metrik & artefak berhasil dilacak ke MLflow (Run ID: {mlflow.active_run().info.run_id})")

if __name__ == "__main__":
    tune_and_log_model()