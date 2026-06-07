import dagshub
import mlflow

dagshub.init(repo_owner='muhazziz', repo_name='mlsystem-studi-kasus-cs', mlflow=True)

with mlflow.start_run():
    print("MLflow run started successfully!")