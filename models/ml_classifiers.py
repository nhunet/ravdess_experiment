"""ML classifier pipelines: SVM (RBF), Random Forest, KNN."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import config


def get_svm_pipeline(seed: int = config.DEFAULT_SEED) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(**config.SVM_PARAMS, random_state=seed)),
    ])


def get_rf_pipeline(seed: int = config.DEFAULT_SEED) -> Pipeline:
    params = {**config.RF_PARAMS, "random_state": seed}
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(**params)),
    ])


def get_knn_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(**config.KNN_PARAMS)),
    ])


ML_CLASSIFIERS = {
    "SVM": get_svm_pipeline,
    "RF": get_rf_pipeline,
    "KNN": get_knn_pipeline,
}
