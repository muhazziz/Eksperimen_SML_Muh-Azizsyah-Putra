import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("Bank_Customer_Churn_Basic")

    mlflow.sklearn.autolog()

    print("Memuat dataset...")
    train_df = pd.read_csv("churn_preprocessing/train.csv")
    test_df = pd.read_csv("churn_preprocessing/test.csv")

    X_train = train_df.drop('Exited', axis=1)
    y_train = train_df['Exited']
    X_test = test_df.drop('Exited', axis=1)
    y_test = test_df['Exited']

    with mlflow.start_run():
        print("Melatih model Random Forest (Basic)...")
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Model berhasil dilatih dengan Akurasi: {acc:.4f}")
        print("Artefak tersimpan di MLflow Tracking UI lokal.")

if __name__ == "__main__":
    main()