"""Write the prompt-injection datasets of the guardrail as JSONL.

Label 1 means "prompt injection or jailbreak", 0 means "benign".

- xTRam1/safe-guard-prompt-injection (10,296 prompts, 30% injections, no
  license stated): the train split gives train and dev (90/10, seeded), the
  test split gives test.
- Three sets from other sources:
  - deepset/prompt-injections (662, Apache-2.0): short questions and
    injections, partly in German. Its train split goes to guard-extra.jsonl,
    its test split is guard-deepset.jsonl.
  - jackhhao/jailbreak-classification (1,306, Apache-2.0): long jailbreak
    role-play prompts and benign role prompts. Its train split goes to
    guard-extra.jsonl, its test split is guard-jailbreak.jsonl.
  - Lakera/gandalf_ignore_instructions (1,000, MIT): real attempts to make
    the Gandalf game reveal its password. All are injections. Test only, so
    it stays a source that no model trains on.

finetune-guard trains on guard-train.jsonl, and also on guard-extra.jsonl
when you pass it. A text that is also in a test set is removed from train,
dev and extra.

Run from the repository root:
    uv run --project train python -I train/export_guard.py
"""

import argparse
import json
import os
import random

from datasets import concatenate_datasets, load_dataset


def key(text: str) -> str:
    return " ".join(text.lower().split())


def write(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    injections = sum(r["label"] for r in rows)
    print(f"{path}: {len(rows)} prompts, {injections} injections")


def unique(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        k = key(r["text"])
        if k and k not in seen:
            seen.add(k)
            out.append(r)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", default="data")
    p.add_argument("--dev-share", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sg = load_dataset("xTRam1/safe-guard-prompt-injection")
    source = "safe-guard"
    pool = unique([{"text": r["text"], "label": int(r["label"]), "source": source} for r in sg["train"]])
    test = unique([{"text": r["text"], "label": int(r["label"]), "source": source} for r in sg["test"]])

    ds = load_dataset("deepset/prompt-injections")
    deepset_row = lambda r: {"text": r["text"], "label": int(r["label"]), "source": "deepset"}
    deepset = unique([deepset_row(r) for r in ds["test"]])
    jb = load_dataset("jackhhao/jailbreak-classification")
    jailbreak_row = lambda r: {"text": r["prompt"], "label": int(r["type"] == "jailbreak"), "source": "jailbreak"}
    jailbreak = unique([jailbreak_row(r) for r in jb["test"]])
    extra = unique([deepset_row(r) for r in ds["train"]] + [jailbreak_row(r) for r in jb["train"]])
    gd = load_dataset("Lakera/gandalf_ignore_instructions")
    gandalf = unique([{"text": r["text"], "label": 1, "source": "gandalf"}
                      for r in concatenate_datasets([gd["train"], gd["validation"], gd["test"]])])

    held_out = {key(r["text"]) for r in test + deepset + jailbreak + gandalf}
    before = len(pool) + len(extra)
    pool = [r for r in pool if key(r["text"]) not in held_out]
    extra = [r for r in extra if key(r["text"]) not in held_out]
    print(f"removed {before - len(pool) - len(extra)} train prompts that are also in a test set")

    random.Random(args.seed).shuffle(pool)
    n_dev = int(len(pool) * args.dev_share)
    write(os.path.join(args.out_dir, "guard-train.jsonl"), pool[n_dev:])
    write(os.path.join(args.out_dir, "guard-dev.jsonl"), pool[:n_dev])
    write(os.path.join(args.out_dir, "guard-extra.jsonl"), extra)
    write(os.path.join(args.out_dir, "guard-test.jsonl"), test)
    write(os.path.join(args.out_dir, "guard-deepset.jsonl"), deepset)
    write(os.path.join(args.out_dir, "guard-jailbreak.jsonl"), jailbreak)
    write(os.path.join(args.out_dir, "guard-gandalf.jsonl"), gandalf)


if __name__ == "__main__":
    main()
