"""Make sure that Burn (Rust) and PyTorch give the same model output.

A model trained in one framework is served by Burn. If the two disagree, the
threshold chosen in one does not hold in the other. This script embeds the
same sentences with sentence-transformers and with semcache-embed, and
compares the vectors. With --rerank, it scores the same pairs with the
cross-encoder in transformers and in Burn, and compares the probabilities,
because the threshold applies to them. With --router, it does the same for
a classifier router on single texts, and --guard for the guardrail classifier. Logits of a trained classifier are
large, so float32 rounding gives them larger absolute differences.

Run from the repository root:
    uv run --project train python train/parity.py --model models/minilm-l6-qqp
    uv run --project train python train/parity.py --model sentence-transformers/all-MiniLM-L6-v2 --burn-model l6
    uv run --project train python train/parity.py --rerank --model models/minilm-l6-qqp-rerank
    uv run --project train python train/parity.py --router --model models/router-classifier
    uv run --project train python train/parity.py --guard --model models/guard-classifier
"""

import argparse
import json
import subprocess
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

SENTENCES = [
    "What is the capital of France?",
    "whats the capital city of france",
    "How do I bake sourdough bread at home?",
    "Why are African-Americans so beautiful?",
    "Why are hispanics so beautiful?",
    "Perché il cielo è blu?",
    "Refactor this Rust function and explain why, step by step.",
    "東京の天気はどうですか？",
    "a",
    "word " * 300,  # longer than 256 tokens: both sides must truncate the same way
]

PAIRS = [
    ("How do I convert dollars to euros?", "How do I convert euros to dollars?"),
    ("What is the capital of France?", "whats the capital city of france"),
    ("Why are African-Americans so beautiful?", "Why are hispanics so beautiful?"),
    ("How do I bake sourdough bread at home?", "What is the best way to learn Rust?"),
    ("Perché il cielo è blu?", "Why is the sky blue?"),
    ("a", "b"),
    ("word " * 300, "other " * 300),  # longer than 512 tokens: both sides must cut the same way
]


def rerank_parity(args) -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    pairs = [(a.strip(), b.strip()) for a, b in PAIRS]
    with torch.no_grad():
        batch = tokenizer([a for a, _ in pairs], [b for _, b in pairs], padding=True,
                          truncation="longest_first", max_length=512, return_tensors="pt")
        torch_logits = model(**batch).logits[:, 0].numpy()

    run = subprocess.run(
        [args.embed_bin, "--rerank", args.burn_model or args.model],
        input="".join(f"{a}\t{b}\n" for a, b in pairs),
        capture_output=True, text=True, check=True,
    )
    burn_logits = np.array([float(line) for line in run.stdout.splitlines()], dtype=np.float32)

    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    diff = np.abs(sigmoid(torch_logits) - sigmoid(burn_logits))
    logit_diff = np.abs(torch_logits - burn_logits)
    for (a, b), t, d, ld in zip(pairs, torch_logits, diff, logit_diff):
        print(f"p {sigmoid(t):.4f}  abs diff {d:.2e}  (logit {t:+.3f}, diff {ld:.1e})  {a[:30]!r} | {b[:30]!r}")
    worst = float(diff.max())
    ok = worst <= args.tolerance
    print(f"\nparity {'OK' if ok else 'FAILED'}: worst abs diff {worst:.2e} (tolerance {args.tolerance:.0e})")
    return 0 if ok else 1


def classifier_parity(args, settings: str) -> int:
    """A single-text classifier: router.json or guard.json gives the token limit."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    with open(f"{args.model}/{settings}") as f:
        max_tokens = json.load(f)["max_tokens"]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    texts = [s.strip() for s in SENTENCES]
    with torch.no_grad():
        batch = tokenizer(texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt")
        torch_p = torch.sigmoid(model(**batch).logits[:, 0]).numpy()

    run = subprocess.run(
        [args.embed_bin, "--classifier", args.burn_model or args.model, "--max-tokens", str(max_tokens)],
        input="".join(f"{t}\n" for t in texts),
        capture_output=True, text=True, check=True,
    )
    burn_p = np.array([float(line) for line in run.stdout.splitlines()], dtype=np.float32)
    diff = np.abs(torch_p - burn_p)
    for t, p, d in zip(texts, torch_p, diff):
        print(f"score {p:.4f}  abs diff {d:.2e}  {t[:40]!r}")
    worst = float(diff.max())
    ok = worst <= args.tolerance
    print(f"\nparity {'OK' if ok else 'FAILED'}: worst abs diff {worst:.2e} (tolerance {args.tolerance:.0e})")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help="model folder or Hub id for PyTorch")
    p.add_argument("--burn-model", help="model for semcache-embed (default: same as --model)")
    p.add_argument("--embed-bin", default="semcache/target/release/semcache-embed")
    p.add_argument("--tolerance", type=float, default=1e-4)
    p.add_argument("--rerank", action="store_true", help="compare a cross-encoder folder instead")
    p.add_argument("--router", action="store_true", help="compare a classifier router folder instead")
    p.add_argument("--guard", action="store_true", help="compare a guardrail classifier folder instead")
    args = p.parse_args()
    if args.rerank:
        return rerank_parity(args)
    if args.router:
        return classifier_parity(args, "router.json")
    if args.guard:
        return classifier_parity(args, "guard.json")

    model = SentenceTransformer(args.model, device="cpu")
    model.max_seq_length = 256  # the Rust side truncates at 256 tokens

    run = subprocess.run(
        [args.embed_bin, "--model", args.burn_model or args.model],
        input="\n".join(s.strip() for s in SENTENCES) + "\n",
        capture_output=True, text=True, check=True,
    )
    burn_vecs = np.array([json.loads(line) for line in run.stdout.splitlines()], dtype=np.float32)

    torch_vecs = model.encode([s.strip() for s in SENTENCES], normalize_embeddings=True, convert_to_numpy=True)
    diff = np.abs(torch_vecs - burn_vecs).max(axis=1)
    cos = (torch_vecs * burn_vecs).sum(axis=1)
    for s, d, c in zip(SENTENCES, diff, cos):
        print(f"max abs diff {d:.2e}  cosine {c:.6f}  {s.strip()[:50]!r}")

    worst = float(diff.max())
    ok = worst <= args.tolerance
    print(f"\nparity {'OK' if ok else 'FAILED'}: worst max abs diff {worst:.2e} (tolerance {args.tolerance:.0e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
