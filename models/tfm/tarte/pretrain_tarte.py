"""Pretrain a small TARTE on sampled YAGO entities and save its backbone."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .common import SEED, get_device, set_seed
    from .pretrained import DEFAULT_CHECKPOINT, FastTextEncoder, RowEncoder, SmallTARTE, collate
    from .yago_data import sample_yago
except ImportError:
    from common import SEED, get_device, set_seed
    from pretrained import DEFAULT_CHECKPOINT, FastTextEncoder, RowEncoder, SmallTARTE, collate
    from yago_data import sample_yago


SAMPLE_ROWS = 15_000
STEPS = 3_000
BATCH_SIZE = 64
MAX_FACTS = 32
LEARNING_RATE = 1e-3
LOG_EVERY = 100


def contrastive_loss(first, second):
    """Symmetric Gaussian InfoNCE with detached median distance bandwidth."""
    distances = torch.cdist(first, second).square()
    off_diagonal = ~torch.eye(len(first), dtype=torch.bool, device=first.device)
    bandwidth = distances.detach()[off_diagonal].median().clamp_min(1e-6)
    logits = -distances / (2 * bandwidth)
    labels = torch.arange(len(first), device=first.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def train(rows, text, device):
    encoder = RowEncoder(text).fit(rows)
    encoded = [encoder.encode(row) for row in rows]
    if any(len(row[0]) == 0 for row in encoded):
        raise ValueError("YAGO sample contains empty rows after encoding.")
    model = SmallTARTE(text.dim).to(device)
    hidden_dim = model.config["hidden_dim"]
    ff_dim = model.config["ff_dim"]
    projection = nn.Sequential(nn.Linear(hidden_dim, ff_dim), nn.ReLU(),
                               nn.Linear(ff_dim, hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(projection.parameters()),
                                 lr=LEARNING_RATE)
    rng = np.random.default_rng(SEED)
    history = []
    model.train()
    for step in range(1, STEPS + 1):
        indices = rng.choice(len(encoded), min(BATCH_SIZE, len(encoded)), replace=False)
        # Random subsets bound attention cost for entities with many facts.
        batch = []
        for index in indices:
            columns, cells = encoded[index]
            selected = rng.choice(len(columns), min(len(columns), MAX_FACTS), replace=False)
            batch.append((columns[selected], cells[selected]))
        columns, cells, missing = collate(batch, device)
        positive_columns, positive_cells = columns.clone(), cells.clone()
        for index, row in enumerate(batch):
            # Replace at most two facts while retaining context where possible.
            # Singleton entities use dropout as their view augmentation.
            count = min(int(rng.integers(1, 3)), len(row[0]) - 1)
            for position in rng.choice(len(row[0]), count, replace=False):
                donor = (index + int(rng.integers(1, len(batch)))) % len(batch)
                donor_position = int(rng.integers(len(batch[donor][0])))
                positive_columns[index, position] = columns[donor, donor_position]
                positive_cells[index, position] = cells[donor, donor_position]
        optimizer.zero_grad(set_to_none=True)
        loss = contrastive_loss(projection(model(columns, cells, missing)),
                                projection(model(positive_columns, positive_cells, missing)))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite pretraining loss at step {step}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(projection.parameters()), 1.0)
        optimizer.step()
        if step == 1 or step % LOG_EVERY == 0 or step == STEPS:
            history.append({"step": step, "loss": float(loss.detach())})
            print(f"Step {step}/{STEPS}: loss={history[-1]['loss']:.5f}", flush=True)
    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yago-path", type=Path, required=True)
    parser.add_argument("--yago-format", choices=["table", "jsonl", "triples"], default="table")
    parser.add_argument("--drop-columns", nargs="*", default=[], help="Exclude IDs/metadata in table exports")
    parser.add_argument("--fasttext-model", type=Path, required=True)
    args = parser.parse_args()
    set_seed(SEED)
    device = get_device()
    text = FastTextEncoder(args.fasttext_model)
    print("Sampling YAGO (scanning the full export)...", flush=True)
    rows, population = sample_yago(args.yago_path, SAMPLE_ROWS, SEED,
                                  args.yago_format, args.drop_columns)
    print(f"Sampled {len(rows)} of {population} entity rows; device={device}", flush=True)
    started = perf_counter()
    model, history = train(rows, text, device)
    metadata = {"source": str(args.yago_path.resolve()), "source_format": args.yago_format,
                "sample_rows": len(rows), "population_rows": population, "seed": SEED,
                "steps": STEPS, "batch_size": BATCH_SIZE, "max_facts": MAX_FACTS,
                "learning_rate": LEARNING_RATE, "drop_columns": args.drop_columns,
                "fasttext_model": text.path, "pretraining_seconds": perf_counter() - started}
    DEFAULT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "config": model.config,
                "backbone_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "fasttext_signature": torch.tensor(np.stack([text.encode(s) for s in
                                                             ("age", "country", "medical")])),
                "metadata": metadata}, DEFAULT_CHECKPOINT)
    DEFAULT_CHECKPOINT.with_suffix(".json").write_text(
        json.dumps({"config": model.config, **metadata, "loss_history": history}, indent=2) + "\n")
    print(f"Saved {DEFAULT_CHECKPOINT}")


if __name__ == "__main__":
    main()
