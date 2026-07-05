import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import dagshub
import os
import time

# Otorisasi Headless menggunakan Token 
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/luckyprdn/Eksperimen_SML_Lucky.git"
dagshub.init(repo_owner='luckyprdn', repo_name='Eksperimen_SML_Lucky', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/luckyprdn/Eksperimen_SML_Lucky.mlflow")

def train_and_log_model():
    print("Membaca dataset...")
    df = pd.read_csv('../Membangun_model/dataset/data_prepared.csv')
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Automated_CI_Pipeline")
    
    with mlflow.start_run() as run:
        start_time = time.time()
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        mlflow.log_params(params)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        predictions = model.predict(X_test)
        
        mlflow.log_metric("accuracy", accuracy_score(y_test, predictions))
        mlflow.log_metric("f1_score", f1_score(y_test, predictions, average='weighted'))
        mlflow.log_metric("training_time_seconds", training_time)
        
        mlflow.sklearn.log_model(model, "model_pipeline")
        print("Model berhasil di-training dan di-log!")

        # SIMPAN RUN ID BUAT DIBACA SAMA GITHUB ACTIONS
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

if __name__ == "__main__":
    train_and_log_model()