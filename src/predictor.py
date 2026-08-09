import os
import numpy as np
from src.config import MODEL_DIR
from src.utils import load_artifact

class EnvironmentNewsPredictor:
    def __init__(self, model_path=None, vectorizer_path=None, threshold=0.45):
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "best_model.joblib")
        if vectorizer_path is None:
            vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
            
        self.model = load_artifact(model_path)
        self.vectorizer = load_artifact(vectorizer_path)
        self.threshold = threshold

    def predict(self, text: str):
        if not text or not isinstance(text, str):
            return {"category": "Invalid Input", "confidence": 0.0, "probabilities": {}}

        vec = self.vectorizer.transform([text])
        probs = self.model.predict_proba(vec)[0]
        
        classes = self.model.classes_
        prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
        
        max_idx = np.argmax(probs)
        max_prob = float(probs[max_idx])
        
        if max_prob < self.threshold:
            predicted_category = "Uncertain / No Clear Focus"
        else:
            predicted_category = classes[max_idx]

        return {
            "category": predicted_category,
            "confidence": max_prob,
            "probabilities": prob_dict
        }