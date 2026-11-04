"""Clean Reddit comments, produce EDA artifacts, and create reproducible splits."""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.data.download_dataset import OUTPUT, download_dataset

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def clean_text(value: str) -> str:
    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@[\w-]+", " USER ", text)
    text = re.sub(r"#[\w-]+", " HASHTAG ", text)
    text = re.sub(r"[^\w\s!?']", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = download_dataset(OUTPUT).rename(columns={"clean_comment": "text", "category": "label"})
    frame["text"] = frame["text"].map(clean_text)
    frame = frame[frame["text"].str.len() > 2].drop_duplicates("text")
    frame = frame[frame["label"].isin([-1, 0, 1])]
    train, test = train_test_split(frame, test_size=0.2, random_state=42, stratify=frame["label"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / "train.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)
    stats = frame.assign(text_length=frame["text"].str.len())
    plt.figure(figsize=(7, 4))
    sns.countplot(data=stats, x="label", order=[-1, 0, 1])
    plt.title("Distribution des classes")
    plt.tight_layout()
    plt.savefig(PROCESSED / "class_distribution.png", dpi=140)
    plt.close()
    stats["text_length"].describe().to_csv(PROCESSED / "text_length_stats.csv")
    print(f"Après nettoyage: {len(frame)} commentaires; train={len(train)}, test={len(test)}")
    return train, test


if __name__ == "__main__":
    prepare()
