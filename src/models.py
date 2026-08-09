import os
from typing import Dict, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB

from src.config import MODEL_DIR, TEST_SIZE, RANDOM_STATE, MAX_FEATURES
from src.utils import save_artifact

from src.labeling import lemmatize_texts

def prepare_features(df: pd.DataFrame) -> Tuple[TfidfVectorizer, any, any, any, any, any, any]:
    if "lemmatized_text" not in df.columns:
        print("'lemmatized_text' not found in DataFrame. Lammatize ...")
        df = df.copy()
        df["lemmatized_text"] = lemmatize_texts(df["full_text"], desc="lemmatized text")

    X_train_lem, X_test_lem, y_train, y_test, X_train_raw, X_test_raw = train_test_split(
        df["lemmatized_text"].fillna(""),
        df["target_category"],
        df["full_text"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target_category"]
    )

    print("using cached lemmatized text for TF-IDF...")

    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        ngram_range=(1, 2)
    )

    X_train_vec = tfidf.fit_transform(X_train_lem)
    X_test_vec = tfidf.transform(X_test_lem)

    save_artifact(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))

    return tfidf, X_train_vec, X_test_vec, y_train, y_test, X_test_raw, y_test


def run_benchmark_and_save(X_train_vec, X_test_vec, y_train, y_test) -> Tuple[Dict[str, any], any]:
    base_svc = LinearSVC(random_state=RANDOM_STATE)
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, cv=3)

    models = {
        "Linear SVC (Calibrated)": calibrated_svc,
        "Multinomial Naive Bayes": MultinomialNB(),
    }

    results = {}
    print("\nBENCHMARK RESULTS:")
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"model": model, "accuracy": acc}
        print(f"- {name}: Accuracy = {acc:.4f}")

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    
    print(f"\nBest Model: {best_name}\n")
    y_pred_best = best_model.predict(X_test_vec)
    print(classification_report(y_test, y_pred_best))

    model_save_path = os.path.join(MODEL_DIR, "best_model.joblib")
    save_artifact(best_model, model_save_path)

    return results, best_model