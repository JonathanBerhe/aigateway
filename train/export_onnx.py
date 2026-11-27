"""Export a MiniLM sentence model to ONNX, with pooling, for ONNX Runtime.

The graph takes input_ids, attention_mask and token_type_ids and returns the
sentence embedding that the sidecar computes: the mean of the token vectors
under the attention mask, then L2-normalized. So ONNX Runtime and Burn can be
compared on the same output (train/bench_onnx.py).

Run from the repository root:
    uv run --project train python train/export_onnx.py --model models/minilm-l6-qqp-burn
"""

import argparse
import os

import torch
from transformers import AutoModel


class SentenceEmbedding(torch.nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="models/minilm-l6-qqp-burn")
    p.add_argument("--out", help="ONNX file (default: <model>/model.onnx)")
    args = p.parse_args()
    out = args.out or os.path.join(args.model, "model.onnx")

    model = SentenceEmbedding(AutoModel.from_pretrained(args.model)).eval()
    ids = torch.ones(1, 8, dtype=torch.long)
    sample = (ids, torch.ones_like(ids), torch.zeros_like(ids))
    dynamic = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
               "token_type_ids": {0: "batch", 1: "seq"}, "embedding": {0: "batch"}}
    torch.onnx.export(model, sample, out, input_names=["input_ids", "attention_mask", "token_type_ids"],
                      output_names=["embedding"], dynamic_axes=dynamic, opset_version=17, dynamo=False)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
