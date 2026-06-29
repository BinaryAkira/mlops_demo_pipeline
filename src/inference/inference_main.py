import joblib
import numpy as np
import os

MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.pkl")
_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Set MODEL_PATH or mount the model file into the container."
            )
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def predict(features: list):
    arr = np.array(features).reshape(1, -1)
    model = _load_model()
    return model.predict(arr).tolist()
