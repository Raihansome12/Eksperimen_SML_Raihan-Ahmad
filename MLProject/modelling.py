import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('heart-disease_preprocessing.csv')
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow experiment
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("eksperimen_sml")

with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    mlflow.log_metric("accuracy_test", acc)
    mlflow.sklearn.save_model(
        sk_model=model,
        path="model"
    )
    mlflow.log_artifacts(
        local_dir="model",
        artifact_path="model"
    )

    print(f"Accuracy: {acc:.4f}")

print("Training selesai. Jalankan: mlflow ui")