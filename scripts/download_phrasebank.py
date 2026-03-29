from pathlib import Path

import pandas as pd
from datasets import load_dataset

print("Downloading Financial PhraseBank...")

ds = load_dataset("financial_phrasebank", "sentences_allagree")

df = pd.DataFrame(list(ds["train"]))  # type: ignore

df.columns = ["sentence", "label"]

label_map = {0: "negative", 1: "neutral", 2: "positive"}
df["sentiment"] = df["label"].map(label_map)

save_path = Path("data/external/financial_phrasebank.csv")
save_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(save_path, index=False)

print(f"Done! Saved {len(df)} sentences to {save_path}")
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())
print("\nSample rows:")
print(df.head(3))
