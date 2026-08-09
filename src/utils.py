import os
import joblib

def save_artifact(obj, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"Artefacts saved: {filepath}")

def load_artifact(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    print(f"Lade Artefakt aus: {filepath}")
    return joblib.load(filepath)