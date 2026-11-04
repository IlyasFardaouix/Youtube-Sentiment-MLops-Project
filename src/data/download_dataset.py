"""Download and validate the official Reddit sentiment dataset."""
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

URL = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "raw" / "reddit.csv"


def download_dataset(output: Path = OUTPUT) -> pd.DataFrame:
    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=30) as response:
        output.write_bytes(response.read())
    frame = pd.read_csv(output)
    if not {"clean_comment", "category"}.issubset(frame.columns):
        raise ValueError(f"Unexpected columns: {list(frame.columns)}")
    frame["category"] = pd.to_numeric(frame["category"], errors="coerce")
    frame = frame.dropna(subset=["clean_comment", "category"])
    frame["category"] = frame["category"].astype(int)
    print(f"Commentaires: {len(frame)}")
    print("Distribution des labels:")
    print(frame["category"].value_counts().sort_index().to_string())
    return frame


if __name__ == "__main__":
    download_dataset()
