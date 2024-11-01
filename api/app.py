from flask import Flask, Response, request

from .errors import errors
from .handlers import predict as predict_handler

import json
import os
import time
from datetime import datetime
from typing import Any, List, Optional

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)
# app.register_blueprint(errors)

# Constants
np.random.seed(42)
TIMESTAMP_FMT = "%m-%d-%Y, %H:%M:%S"
LABEL: str = "target"

NUMERIC_FEATURES: List[str] = [
    "age",
    "trestbps",
    "chol",
    "fbs",
    "thalach",
    "exang",
    "oldpeak",
]

CATEGORICAL_FEATURES: List[str] = ["sex", "cp", "restecg", "ca", "slope", "thal"]


# Create the pipeline function
def create_pipeline(categorical_features: List[str], numeric_features: List[str]) -> Pipeline:

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())])


# Train route
@app.route("/train", methods=["POST"])
def train():
    data = request.json
    # path = data.get("path")
    # path = os.getenv("MODEL_PATH", "data/pipeline.pkl"),
    # model_path = os.getenv("MODEL_PATH", "data/pipeline.pkl"),
    # metrics_path  = os.getenv("METRICS_PATH", "data/metrics.json"),
    path = r"D:\sklearn-flask-api-demo\data\heart-disease.csv"
    model_path = r"D:\sklearn-flask-api-demo\data\pipeline.pkl"
    metrics_path = r"D:\sklearn-flask-api-demo\data\pmetrics.json"
    # model_path = data.get("model_path", "data/pipeline.pkl")
    # metrics_path = data.get("metrics_path", "data/metrics.json")
    test_size = data.get("test_size", 0.2)
    dump = data.get("dump", True)

    categorical_features = data.get("categorical_features", CATEGORICAL_FEATURES)
    numeric_features = data.get("numeric_features", NUMERIC_FEATURES)
    label = data.get("label", LABEL)

    start = time.time()

    # Load dataset
    print(f"read_csv path {path}")
    data_frame = pd.read_csv(path)
    features = data_frame[categorical_features + numeric_features]
    target = data_frame[label]

    # Train test split by scikit-learn
    # tx: The training set for the input features.
    # vx: The test/validation set for the input features.
    # ty: The training set for the target (label).
    # vy: The test/validation set for the target (label).
    tx, vx, ty, vy = train_test_split(features, target, test_size=test_size)

    # Create model and train

    # model the pipeline object
    model = create_pipeline(categorical_features=categorical_features, numeric_features=numeric_features)
    model.fit(tx, ty)

    end = time.time()

    # Calculate metrics

    #Score on train set
    acc = accuracy_score(model.predict(tx), ty) * 100
    # Score on test set
    val_acc = accuracy_score(model.predict(vx), vy) * 100
    # Score on test set
    roc_auc = roc_auc_score(vy, model.predict_proba(vx)[:, -1])

    metrics = dict(
        elapsed=end - start,
        acc=acc,
        val_acc=val_acc,
        roc_auc=roc_auc,
        timestamp=datetime.now().strftime(TIMESTAMP_FMT),
    )

    # Save model and metrics
    if dump:
        joblib.dump(model, model_path)
        json.dump(metrics, open(metrics_path, "w"))

    return jsonify(metrics), 200



@app.route("/predict", methods=["POST"])
def predict():
    return predict_handler(request)


# feel free to add as many handlers in here as you like!


@app.route("/health")
def health():
    return Response("OK", status=200)
