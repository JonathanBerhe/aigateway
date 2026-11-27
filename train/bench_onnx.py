"""Time ONNX Runtime on the cache model and compare its embeddings with Burn.

It embeds the prompts of the sidecar load test one at a time (tokenizer
included) with ONNX Runtime on the CPU, with 1 thread and with all cores, and
prints p50, p95 and p99. Then it embeds the parity sentences with ONNX Runtime
and with Burn (semcache-embed) and prints the largest difference. The Burn
latency comes from `make bench` (one connection), which adds the sidecar
protocol, about 0.1 ms.

Run from the repository root, after train/export_onnx.py:
    uv run --project train python train/bench_onnx.py
"""

import argparse
import json
import subprocess
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from parity import SENTENCES


def session(path: str, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads  # 0 = ONNX Runtime picks (all cores)
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def embed(sess, tok, text: str) -> np.ndarray:
    e = tok.encode(text)
    feed = {
        "input_ids": np.array([e.ids], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids], dtype=np.int64),
    }
    return sess.run(["embedding"], feed)[0][0]


def latency(sess, tok, texts: list[str]) -> tuple[float, float, float]:
    for t in texts[:20]:
        embed(sess, tok, t)
    ms = []
    for t in texts:
        t0 = time.perf_counter()
        embed(sess, tok, t)
        ms.append((time.perf_counter() - t0) * 1000)
    return tuple(float(np.percentile(ms, p)) for p in (50, 95, 99))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="models/minilm-l6-qqp-burn")
    p.add_argument("--embed-bin", default="semcache/target/release/semcache-embed")
    args = p.parse_args()

    tok = Tokenizer.from_file(f"{args.model}/tokenizer.json")
    tok.enable_truncation(max_length=256)  # the sidecar's MAX_TOKENS
    tok.no_padding()
    texts = {
        "short": open("data/bench-short.txt").read().splitlines()[:500],
        "long": open("data/bench-long.txt").read().splitlines()[:300],
    }
    print("| ONNX Runtime threads | Prompts | p50 ms | p95 ms | p99 ms |")
    print("|---|---|---|---|---|")
    for threads, label in [(1, "1"), (0, "all cores")]:
        sess = session(f"{args.model}/model.onnx", threads)
        for kind, lines in texts.items():
            p50, p95, p99 = latency(sess, tok, lines)
            print(f"| {label} | {kind} | {p50:.1f} | {p95:.1f} | {p99:.1f} |")

    sess = session(f"{args.model}/model.onnx", 1)
    sentences = [s.strip() for s in SENTENCES]
    onnx_vecs = np.array([embed(sess, tok, s) for s in sentences])
    run = subprocess.run([args.embed_bin, "--model", args.model], input="".join(s + "\n" for s in sentences),
                         capture_output=True, text=True, check=True)
    burn_vecs = np.array([json.loads(line) for line in run.stdout.splitlines()], dtype=np.float32)
    diff = float(np.abs(onnx_vecs - burn_vecs).max())
    cos = float((onnx_vecs * burn_vecs).sum(1).min())
    print(f"\nparity with Burn: largest abs diff {diff:.2e}, lowest cosine {cos:.6f}")


if __name__ == "__main__":
    main()
