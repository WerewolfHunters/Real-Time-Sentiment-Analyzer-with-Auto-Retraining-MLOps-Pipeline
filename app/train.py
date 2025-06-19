import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df_train = pd.read_csv('data/Train.csv')
df_test = pd.read_csv('data/Test.csv')

X_train = df_train['text']
y_train = df_train['label']

X_test = df_test['text']
y_test = df_test['label']


# Set MLflow experiment
mlflow.set_experiment("Sentiment_Classifier")

with mlflow.start_run():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression())
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(pipe, "sentiment_model")

    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

    # Save model locally
    joblib.dump(pipe, "models/sentiment_model.pkl")
