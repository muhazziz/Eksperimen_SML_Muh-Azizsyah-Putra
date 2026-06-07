import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import mlflow
import dagshub

def main():
    dagshub.init(repo_owner='muhazziz', repo_name='mlsystem-studi-kasus-cs', mlflow=True)

    mlflow.set_experiment("Bank_Customer_Churn_Prediction")

    mlflow.sklearn.autolog()

    print("Memuat dataset...")
    train_df = pd.read_csv("churn_preprocessing/train.csv")
    test_df = pd.read_csv("churn_preprocessing/test.csv")

    X_train = train_df.drop('Exited', axis=1)
    y_train = train_df['Exited']
    X_test = test_df.drop('Exited', axis=1)
    y_test = test_df['Exited']

    with mlflow.start_run():
        print("Memulai Hyperparameter Tuning dengan GridSearchCV...")
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'random_state': [42]
        }
        
        rf = RandomForestClassifier()
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"Model terbaik ditemukan: {grid_search.best_params_}")

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", acc)
        
        print("Menciptakan dan mencatat artefak tambahan...")
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = "training_confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()
        
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10,6))
        plt.title("Feature Importances")
        plt.bar(range(X_train.shape[1]), importances[indices], align="center")
        plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
        plt.tight_layout()
        fi_path = "feature_importance.png"
        plt.savefig(fi_path)
        mlflow.log_artifact(fi_path)
        plt.close()
        
        print("Selesai! Seluruh metadata, model, dan artefak telah tersinkronisasi ke DagsHub.")

if __name__ == "__main__":
    main()