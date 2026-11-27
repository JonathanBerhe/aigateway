"""Write the SPROUT router dataset as JSONL, one file per split.

SPROUT (CARROT-LLM-Routing/SPROUT on Hugging Face) has 44,241 prompts. For each
prompt, it keeps the answers of 13 models, a score from 0 to 1 that an LLM
judge gave each answer, and the token counts. With the scores of a cheap and
a strong model, the router evaluation computes the quality of any routing
rule offline, with no LLM calls.

The script reads only the key, dataset, prompt, score and token columns,
about 26 MB of the 581 MB of Parquet files. The answers and the judge texts stay on the Hub.
The dataset card states no license, so the files stay in data/ (gitignored).

Run from the repository root:
    uv run --project train python train/export_sprout.py
"""

import argparse
import json
import os

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "datasets/CARROT-LLM-Routing/SPROUT"
SPLITS = {"train": "train", "validation": "dev", "test": "test"}
FIELDS = ("score", "num_input_tokens", "num_output_tokens")


def models_of(pf: pq.ParquetFile) -> list[str]:
    """The struct columns that have a score: one per model."""
    paths = (pf.schema.column(i).path for i in range(len(pf.schema)))
    return sorted({p.rsplit(".", 1)[0] for p in paths if p.endswith(".score")})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", default="data")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fs = HfFileSystem()
    for split, name in SPLITS.items():
        files = sorted(fs.glob(f"{REPO}/data/{split}-*.parquet"))
        out = os.path.join(args.out_dir, f"sprout-{name}.jsonl")
        rows = 0
        with open(out, "w", encoding="utf-8") as f:
            for path in files:
                # No read cache: each read is one range request of the size
                # that pyarrow asks for. The default cache fetched about 160
                # times more bytes than pyarrow read.
                with fs.open(path, "rb", cache_type="none") as src:
                    pf = pq.ParquetFile(src)
                    models = models_of(pf)
                    columns = ["key", "dataset", "prompt"] + [f"{m}.{x}" for m in models for x in FIELDS]
                    if not models:
                        raise SystemExit(f"{path}: no model columns with a score")
                    table = pf.read(columns=columns).to_pylist()
                for r in table:
                    record = {
                        "key": r["key"],
                        "dataset": r["dataset"],
                        "prompt": r["prompt"],
                        "score": {m: r[m]["score"] for m in models},
                        "input_tokens": {m: r[m]["num_input_tokens"] for m in models},
                        "output_tokens": {m: r[m]["num_output_tokens"] for m in models},
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rows += 1
        print(f"{split}: wrote {rows} prompts to {out}")


if __name__ == "__main__":
    main()
