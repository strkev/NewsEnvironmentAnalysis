import multiprocessing as mp

import pandas as pd
import spacy
from tqdm import tqdm

from src.config import CATEGORIES

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def lemmatize_texts(texts, desc="Lemmatisiere Texte"):
    cleaned_texts = []
    num_cpus = max(1, mp.cpu_count() - 1)

    for doc in tqdm(
        nlp.pipe(texts, batch_size=256, n_process=num_cpus), total=len(texts), desc=desc
    ):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and token.is_alpha
        ]
        cleaned_texts.append(" ".join(tokens))

    return cleaned_texts


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

    print("\nLammatizing and caching of data...")
    df_labeled["lemmatized_text"] = lemmatize_texts(
        df_labeled["full_text"], desc="Pre-Lemmatization"
    )

    return df_labeled
