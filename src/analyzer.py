import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.config import MODEL_DIR

def analyze_top_features(model, tfidf_vectorizer, top_n=10):
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    
    if hasattr(model, "calibrated_classifiers_"):
        coefs = np.mean([clf.estimator.coef_ for clf in model.calibrated_classifiers_], axis=0)
        classes = model.classes_
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        classes = model.classes_
    else:
        print("Feature importance extraction not supported for this model type.")
        return

    print("\nTOP WORDS PER CATEGORY:")
    for i, category in enumerate(classes):
        top_indices = np.argsort(coefs[i])[-top_n:][::-1]
        top_words = feature_names[top_indices]
        print(f"- {category}: {', '.join(top_words)}")

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    save_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nConfusion matrix plot saved to: {save_path}")

def analyze_predictions_and_errors(model, tfidf_vectorizer, X_test_raw, y_test, num_samples=3):
    X_test_vec = tfidf_vectorizer.transform(X_test_raw)
    
    probs = model.predict_proba(X_test_vec)
    preds = model.classes_[np.argmax(probs, axis=1)]
    confidences = np.max(probs, axis=1)

    results_df = pd.DataFrame({
        "text": X_test_raw.values,
        "true_label": y_test.values,
        "predicted_label": preds,
        "confidence": confidences
    })

    print("\nERROR ANALYSIS:")
    errors = results_df[results_df["true_label"] != results_df["predicted_label"]].sort_values(by="confidence", ascending=False)
    print(f"Top {num_samples} High-Confidence Errors:")
    for idx, row in errors.head(num_samples).iterrows():
        print(f"True: {row['true_label']} | Pred: {row['predicted_label']} ({row['confidence']:.2%}) -> {row['text'][:120]}...")

    print(f"\nTop {num_samples} Most Uncertain Predictions:")
    uncertain = results_df.sort_values(by="confidence", ascending=True)
    for idx, row in uncertain.head(num_samples).iterrows():
        print(f"True: {row['true_label']} | Pred: {row['predicted_label']} ({row['confidence']:.2%}) -> {row['text'][:120]}...")