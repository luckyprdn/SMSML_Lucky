import os
import logging
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data() -> pd.DataFrame:
    """
    Memuat dataset Breast Cancer Wisconsin dari scikit-learn.
    
    Returns:
        pd.DataFrame: DataFrame berisi fitur dan target.
    """
    try:
        logging.info("Memulai proses pemuatan dataset Breast Cancer...")
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        logging.info(f"Dataset berhasil dimuat dengan dimensi: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Terjadi kesalahan saat memuat dataset: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Melakukan preprocessing data meliputi pemisahan target, 
    standarisasi fitur, dan train-test split.
    
    Args:
        df (pd.DataFrame): Dataset mentah.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test yang sudah diproses.
    """
    try:
        logging.info("Memulai proses preprocessing (Scaling dan Splitting)...")
        X = df.drop(columns=['target'])
        y = df['target']

        # Standarisasi fitur
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        # Train-Test Split (80:20) dengan stratifikasi
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logging.info("Preprocessing selesai.")
        logging.info(f"Dimensi X_train: {X_train.shape}, X_test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Terjadi kesalahan saat preprocessing: {e}")
        raise

def save_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, output_dir: str = 'dataset') -> None:
    """
    Menyimpan hasil preprocessing ke dalam format CSV di folder yang ditentukan.
    
    Args:
        X_train (pd.DataFrame): Fitur training.
        X_test (pd.DataFrame): Fitur testing.
        y_train (pd.Series): Target training.
        y_test (pd.Series): Target testing.
        output_dir (str): Direktori penyimpanan (default: 'dataset').
    """
    try:
        logging.info(f"Menyimpan data hasil preprocessing ke folder '{output_dir}/'...")
        os.makedirs(output_dir, exist_ok=True)
        
        X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
        
        logging.info("Semua file berhasil disimpan: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    except Exception as e:
        logging.error(f"Terjadi kesalahan saat menyimpan data: {e}")
        raise

def main() -> None:
    """
    Fungsi utama untuk menjalankan seluruh pipeline data secara berurutan.
    """
    logging.info("=== Memulai Pipeline Data Otomatis ===")
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    save_data(X_train, X_test, y_train, y_test)
    logging.info("=== Pipeline Data Selesai Tanpa Error ===")

if __name__ == "__main__":
    main()