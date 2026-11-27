"""Write the QQP train split as TSV, so the Burn trainer can read it without Python.

This only converts the file format (Parquet to TSV). The leak filter against
the dev and test sets runs in the Burn trainer itself.

Run from the repository root:
    uv run --project train python train/export_qqp.py --out data/qqp-train-raw.tsv
"""

import argparse

from datasets import load_dataset


def clean(text: str) -> str:
    # Tabs and line breaks would break the TSV. scripts/fetch-qqp.sh does the same.
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="data/qqp-train-raw.tsv")
    args = p.parse_args()

    train = load_dataset("nyu-mll/glue", "qqp", split="train")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("question1\tquestion2\tlabel\n")
        for r in train:
            f.write(f"{clean(r['question1'])}\t{clean(r['question2'])}\t{r['label']}\n")
    print(f"wrote {len(train)} pairs to {args.out}")


if __name__ == "__main__":
    main()
