import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import dagshub
import os
import time

# Otorisasi Headless menggunakan Token (Tidak butuh pop-up browser)
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/luckyprdn/Eksperimen_SML_Lucky.git"
dagshub.init(repo_owner='luckyprdn', repo_name='Eksperimen_SML_Lucky', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/luckyprdn/Eksperimen_SML_Lucky.mlflow")

def train_and_log_model():
    print("Membaca dataset...")
    # Pastikan file CSV tersedia di dalam GitHub repository
    df = pd.read_csv('../Membangun_model/dataset/data_prepared.csv')
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Automated_CI_Pipeline")
    
    with mlflow.start_run():
        start_time = time.time()
        
        # Hyperparameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
        
        mlflow.log_params(params)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        predictions = model.predict(X_test)
        
        # Metrik
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Log model ke MLflow
        mlflow.sklearn.log_model(model, "model_pipeline")
        print("Model berhasil di-training dan di-log ke DagsHub via CI/CD!")

if __name__ == "__main__":
    train_and_log_model()