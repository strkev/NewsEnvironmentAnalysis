import pandas as pd
from src.config import CATEGORIES

def assign_weak_label(text: str) -> str:
    if not isinstance(text, str) or len(text) < 20:
        return None

    text_lower = text.lower()
    scores = {cat: 0 for cat in CATEGORIES}

    for category, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in text_lower:
                scores[category] += 1

    max_score = max(scores.values())
    if max_score == 0:
        return "Unlabeled"

    return max(scores, key=scores.get)

def apply_weak_supervision(df: pd.DataFrame) -> pd.DataFrame:
    print("Weak Supervision ...")
    df = df.copy()
    df["target_category"] = df["full_text"].apply(assign_weak_label)
    
    df_labeled = df[df["target_category"] != "Unlabeled"].copy()
    
    print("\nGenerated Labels:")
    print(df_labeled["target_category"].value_counts())
    return df_labeled