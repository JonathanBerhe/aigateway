"""Fine-tune all-MiniLM-L6-v2 for the semantic cache on Quora Question Pairs.

A cache needs a similarity that separates "same question" from "same topic".
OnlineContrastiveLoss pulls paraphrase pairs together. It also pushes apart
the hard non-paraphrase pairs, the ones closer than the margin. A high cache
threshold depends on exactly that separation.

Data hygiene: QQP reuses questions across pairs. Every training pair that
shares a question with a holdout set (dev or test) is removed, so the
evaluation never sees a question that the model trained on.

Run from the repository root:
    uv run --project train python train/finetune_cache.py --out models/minilm-l6-qqp
"""

import argparse
import csv
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.evaluation import BinaryClassificationEvaluator
from sentence_transformers.sentence_transformer.losses import OnlineContrastiveLoss


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def read_pairs(path: str) -> list[tuple[str, str, int]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(rows)  # header
        return [(r[0], r[1], int(r[2])) for r in rows if len(r) >= 3]


def cosine_summary(scores: dict) -> str:
    keys = ("cosine_ap", "cosine_f1", "cosine_f1_threshold", "cosine_precision", "cosine_recall")
    found = {k: v for k, v in scores.items() for key in keys if k.endswith(key)}
    return ", ".join(f"{k.split('_', 1)[1]}={v:.3f}" for k, v in sorted(found.items()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--out", default="models/minilm-l6-qqp")
    p.add_argument("--holdout", nargs="+", default=["data/qqp-dev.tsv", "data/qqp-test.tsv"],
                   help="TSV pair files to keep out of training; the first one is evaluated")
    p.add_argument("--max-train", type=int, default=0, help="cap on training pairs, 0 = all")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--margin", type=float, default=0.5, help="cosine distance margin for negatives")
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    holdout = {path: read_pairs(path) for path in args.holdout}
    held = {norm(q) for rows in holdout.values() for q1, q2, _ in rows for q in (q1, q2)}

    train = load_dataset("nyu-mll/glue", "qqp", split="train")
    total = len(train)
    train = train.filter(lambda r: norm(r["question1"]) not in held and norm(r["question2"]) not in held)
    print(f"train pairs: {len(train)} of {total} "
          f"({total - len(train)} removed because they share a question with a holdout set)")
    train = train.shuffle(seed=args.seed)
    if args.max_train:
        train = train.select(range(min(args.max_train, len(train))))
    train = train.select_columns(["question1", "question2", "label"])
    positives = sum(train["label"])
    print(f"training on {len(train)} pairs ({positives} paraphrases)")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(args.base, device=device)
    model.max_seq_length = args.max_seq_length

    dev_path, dev = args.holdout[0], holdout[args.holdout[0]]
    evaluator = BinaryClassificationEvaluator(
        [r[0] for r in dev], [r[1] for r in dev], [r[2] for r in dev],
        name="dev", batch_size=128, write_csv=False,
    )
    print(f"{dev_path} before: {cosine_summary(evaluator(model))}")

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(Path(args.out) / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=0.1,  # a float is a ratio of the total steps in Transformers v5
        logging_steps=250,
        save_strategy="no",
        eval_strategy="no",
        seed=args.seed,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        loss=OnlineContrastiveLoss(model, margin=args.margin),
    )
    started = time.time()
    trainer.train()
    print(f"trained in {time.time() - started:.0f}s on {device}")

    print(f"{dev_path} after: {cosine_summary(evaluator(model))}")
    model.save(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
