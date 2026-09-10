"""Schema-independent, small TARTE with pretrained FastText inputs.

Shared seed, device, and projection utilities live in common.py.
"""
from functools import lru_cache
from pathlib import Path
import re
from urllib.parse import unquote

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import PowerTransformer, StandardScaler

try:
    from .common import ProjectionLayer
except ImportError:
    from common import ProjectionLayer

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "tarte_yago.pt"
HIDDEN_DIM = 64
N_HEADS = 4
N_LAYERS = 2
FF_DIM = 128
DROPOUT = 0.1
BATCH_SIZE = 64


class FastTextEncoder:
    def __init__(self, path):
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"FastText model not found: {path}. Supply a pretrained .bin model.")
        try:
            import fasttext
        except ImportError as error:
            raise ImportError("Install models/tfm/tarte/requirements.txt to use FastText.") from error
        self.model = fasttext.load_model(str(path))
        self.path = str(path)
        self.dim = self.model.get_dimension()

    @lru_cache(maxsize=50000)
    def encode(self, text):
        text = unquote(str(text).strip().strip("<>"))
        if text.startswith(("http://", "https://")):
            text = re.split(r"[/#]", text)[-1]
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).replace("_", " ")
        text = " ".join(text.split()).lower()
        return np.asarray(self.model.get_sentence_vector(text), dtype=np.float32)


class RowEncoder:
    """Relation-wise numeric transforms fitted exclusively on training rows."""
    def __init__(self, text_encoder):
        self.text = text_encoder
        self.transforms = {}

    @staticmethod
    def number(value):
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
            return number if np.isfinite(number) else None
        except (ValueError, TypeError):
            return None

    def fit(self, rows):
        values = {}
        for row in rows:
            for relation, value in row:
                number = self.number(value)
                if number is not None:
                    values.setdefault(relation, []).append(number)
        self.transforms = {}
        for relation, numbers in values.items():
            array = np.asarray(numbers).reshape(-1, 1)
            if len(numbers) > 1 and np.ptp(array) > 1e-8:
                self.transforms[relation] = PowerTransformer().fit(array)
            else:
                self.transforms[relation] = StandardScaler().fit(array)
        return self

    def encode(self, row):
        columns, cells = [], []
        for relation, value in row:
            if value is None or str(value).strip().lower() in {"", "nan", "null"}:
                continue
            column = self.text.encode(relation)
            number = self.number(value)
            if number is None:
                cell = self.text.encode(str(value))
            else:
                transform = self.transforms.get(relation)
                number = transform.transform([[number]])[0, 0] if transform else number
                cell = np.float32(number) * column
            columns.append(column)
            cells.append(cell)
        # An empty row contains only the model's readout token.
        return (np.asarray(columns, dtype=np.float32).reshape(-1, self.text.dim),
                np.asarray(cells, dtype=np.float32).reshape(-1, self.text.dim))


def collate(rows, device):
    width = max(1, max(len(row[0]) for row in rows))
    dim = rows[0][0].shape[-1]
    columns = torch.zeros(len(rows), width, dim, device=device)
    cells = torch.zeros_like(columns)
    missing = torch.ones(len(rows), width, dtype=torch.bool, device=device)
    for index, (column, cell) in enumerate(rows):
        size = len(column)
        columns[index, :size] = torch.as_tensor(column, device=device)
        cells[index, :size] = torch.as_tensor(cell, device=device)
        missing[index, :size] = False
    return columns, cells, missing


class SmallTARTE(nn.Module):
    def __init__(self, text_dim):
        super().__init__()
        self.config = dict(text_dim=text_dim, hidden_dim=HIDDEN_DIM, n_heads=N_HEADS,
                           n_layers=N_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT)
        self.column_projection = ProjectionLayer(text_dim, HIDDEN_DIM)
        self.cell_projection = ProjectionLayer(text_dim, HIDDEN_DIM)
        self.readout = nn.Parameter(torch.randn(1, 1, HIDDEN_DIM) * 0.02)
        layer = nn.TransformerEncoderLayer(HIDDEN_DIM, N_HEADS, FF_DIM, DROPOUT,
                                            batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, N_LAYERS, enable_nested_tensor=False)

    def forward(self, columns, cells, missing):
        tokens = self.column_projection(columns) + self.cell_projection(cells)
        tokens = torch.cat((self.readout.expand(len(tokens), -1, -1), tokens), dim=1)
        mask = torch.cat((missing.new_zeros(len(tokens), 1), missing), dim=1)
        return self.transformer(tokens, src_key_padding_mask=mask)[:, 0]


def load_backbone(path, text_encoder, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint. Run pretrain_tarte.py first.")
    if checkpoint["config"]["text_dim"] != text_encoder.dim:
        raise ValueError("FastText dimension differs from the pretraining model.")
    signature = torch.tensor(np.stack([text_encoder.encode(s) for s in ("age", "country", "medical")]))
    if not torch.allclose(signature, checkpoint["fasttext_signature"], atol=1e-6, rtol=1e-5):
        raise ValueError("Use the same pretrained FastText model as in pretraining.")
    model = SmallTARTE(text_encoder.dim)
    if checkpoint["config"] != model.config:
        raise ValueError("Checkpoint architecture differs from the fixed TARTE architecture.")
    model.load_state_dict(checkpoint["backbone_state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint


def extract_features(model, rows, device):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(*collate(rows[start:start + BATCH_SIZE], device)).cpu().numpy()
                               for start in range(0, len(rows), BATCH_SIZE)])
