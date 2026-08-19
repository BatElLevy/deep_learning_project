#!/usr/bin/env python
# coding: utf-8

"""
Final runtime inference.

Required interface:
    python main.py <ofile> <DBP> <DNA>

Example:
    python main.py DBP17.txt DBP17 test_seqs.txt

Final prediction:
    0.5 * mean(V7 folds 1..10)
  + 0.5 * mean(SimBind folds 1..10)

All expensive protein preprocessing is precomputed:
- test_protein_embeddings.pt
- test_similarities/fold_i/fold_i_test_pairwise_AA_SID.csv
"""

import csv
import gc
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Configuration
# =========================================================

ROOT = Path(__file__).resolve().parent

EMBEDDINGS_FILE = ROOT / "test_protein_embeddings.pt"
V7_DIR = ROOT / "checkpoints_v7"

# Real SimBind submission layout:
# simbind_custom/
#   models/fold_1 ... fold_10
#   test_similarities/fold_1 ... fold_10
SIMBIND_ROOT = ROOT / "simbind_custom"
SIMBIND_DIR = SIMBIND_ROOT / "models"
SIMILARITY_DIR = SIMBIND_ROOT / "test_similarities"
PRECOMPUTED_TOP14_FILE = ROOT / "simbind_top14_precomputed.npz"

N_FOLDS = 10
N_TEST_PROTEINS = 64

TOP_K = 14
SOFTMAX_SCALE = 8.0

V7_WEIGHT = 0.5
SIMBIND_WEIGHT = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

V7_BATCH = 4096 if DEVICE.type == "cuda" else 512
SIMBIND_BATCH = 1024 if DEVICE.type == "cuda" else 128

DNA_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


# =========================================================
# Input
# =========================================================

def read_dna(path):
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            seq = line.strip().upper()
            if not seq or seq.startswith(">"):
                continue
            if len(seq) != 36:
                raise ValueError(
                    f"DNA sequence on line {line_no} has length "
                    f"{len(seq)}; expected 36."
                )
            bad = set(seq) - set(DNA_TO_INDEX)
            if bad:
                raise ValueError(
                    f"Invalid DNA nucleotide(s) on line {line_no}: "
                    f"{sorted(bad)}"
                )
            seqs.append(seq)

    if not seqs:
        raise ValueError(f"No DNA sequences found in {path}")

    return seqs


def parse_dbp(name):
    m = re.fullmatch(r"DBP(\d+)", name.strip(), flags=re.I)
    if m is None:
        raise ValueError("DBP must have format DBP1 ... DBP64.")

    number = int(m.group(1))
    if not 1 <= number <= N_TEST_PROTEINS:
        raise ValueError("DBP number must be between 1 and 64.")

    index = number - 1
    return index, f"protein_{index:04d}"


def encode_dna_inputs(seqs):
    joined = "".join(seqs).encode("ascii")
    chars = np.frombuffer(
        joined,
        dtype=np.uint8,
    ).reshape(len(seqs), 36)

    lut = np.zeros(256, dtype=np.int64)
    lut[ord("A")] = 0
    lut[ord("C")] = 1
    lut[ord("G")] = 2
    lut[ord("T")] = 3

    indices = torch.from_numpy(lut[chars]).long()
    one_hot = F.one_hot(indices, num_classes=4).float()

    v7 = one_hot.permute(0, 2, 1).contiguous()

    simbind = torch.zeros(
        len(seqs),
        41,
        4,
        dtype=torch.float32,
    )
    simbind[:, :36, :] = one_hot

    return v7, simbind


def load_embeddings():
    if not EMBEDDINGS_FILE.is_file():
        raise FileNotFoundError(f"Missing {EMBEDDINGS_FILE}")

    obj = torch.load(EMBEDDINGS_FILE, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        x = obj.float()
    elif isinstance(obj, (list, tuple)):
        x = torch.stack([torch.as_tensor(v).float() for v in obj])
    elif isinstance(obj, dict):
        x = None

        for key in ("embeddings", "protein_embeddings", "test_protein_embeddings"):
            if key in obj:
                x = torch.as_tensor(obj[key]).float()
                break

        if x is None:
            numeric_keys = list(range(64))
            if all(k in obj for k in numeric_keys):
                x = torch.stack(
                    [torch.as_tensor(obj[k]).float() for k in numeric_keys]
                )

        if x is None:
            dbp_keys = [f"DBP{i}" for i in range(1, 65)]
            if all(k in obj for k in dbp_keys):
                x = torch.stack(
                    [torch.as_tensor(obj[k]).float() for k in dbp_keys]
                )

        if x is None:
            raise ValueError("Unsupported embedding dictionary format.")
    else:
        raise TypeError(f"Unsupported embedding object: {type(obj)}")

    if tuple(x.shape) != (64, 960):
        raise ValueError(
            f"Expected test protein embeddings [64, 960], got {tuple(x.shape)}."
        )

    return x.contiguous()


# =========================================================
# V7 architecture - final V7 configuration
# =========================================================

class GatedPooling1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Conv1d(channels, 1, 1)

    def forward(self, x):
        w = torch.softmax(self.gate(x), dim=2)
        return torch.sum(x * w, dim=2)


class DilatedResidualConvBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.activation(self.conv(x))
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        return self.dropout(x) + residual


class DNAEmbeddingCNN(nn.Module):
    def __init__(
        self,
        model_dim=64,
        protein_dim=256,
        dropout=0.2653,
        sequence_length=36,
        transformer_layers=2,
        transformer_heads=4,
        transformer_feedforward_dim=256,
        transformer_dropout=0.2653,
    ):
        super().__init__()

        self.input_projection = nn.Conv1d(4, model_dim, 1)
        self.sequence_length = sequence_length
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, model_dim)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        self.conv_block_1 = DilatedResidualConvBlock(
            model_dim, 5, 1, dropout
        )
        self.conv_block_2 = DilatedResidualConvBlock(
            model_dim, 9, 2, dropout
        )
        self.conv_block_3 = DilatedResidualConvBlock(
            model_dim, 13, 4, dropout
        )

        self.extra_conv = nn.Conv1d(model_dim, model_dim, 9, padding=4)
        self.extra_conv_weight = nn.Parameter(torch.tensor(0.0))

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_feedforward_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(model_dim),
        )

        self.protein_conditioner = nn.Sequential(
            nn.LayerNorm(protein_dim),
            nn.Linear(protein_dim, model_dim * 2),
        )
        self.conditioning_strength = nn.Parameter(torch.tensor(0.0))

        self.gated_pool = GatedPooling1D(model_dim)
        self.output_layer_norm = nn.LayerNorm(model_dim * 2)
        self.output_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(model_dim * 2, 64)
        self.output_activation = nn.GELU()

    def forward(self, x, protein_embedding):
        x = self.input_projection(x)
        length = x.size(2)

        if length > self.sequence_length:
            raise ValueError("DNA sequence exceeds configured V7 length.")

        x = x.transpose(1, 2)
        x = x + self.position_embedding[:, :length, :]
        x = x.transpose(1, 2)

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        x = x + self.extra_conv_weight * self.extra_conv(x)

        x = self.transformer(x.transpose(1, 2)).transpose(1, 2)

        conditioning = self.protein_conditioner(protein_embedding)
        gamma, beta = conditioning.chunk(2, dim=1)
        gamma = torch.tanh(gamma).unsqueeze(2)
        beta = torch.tanh(beta).unsqueeze(2)

        x = x + torch.tanh(self.conditioning_strength) * (gamma * x + beta)

        pooled = torch.cat(
            [torch.max(x, dim=2).values, self.gated_pool(x)],
            dim=1,
        )
        pooled = self.output_layer_norm(pooled)
        pooled = self.output_dropout(pooled)

        return self.output_activation(self.output_projection(pooled))


class ProteinProjection(nn.Module):
    def __init__(
        self,
        input_dim=960,
        output_dim=256,
        dropout=0.15,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.relu(self.fc(self.layer_norm(x))))


class GatedCosineLowRankInteraction(nn.Module):
    def __init__(
        self,
        protein_dim=256,
        dna_dim=64,
        rank=64,
        initial_gating_strength=1.0,
    ):
        super().__init__()
        self.protein_proj = nn.Linear(protein_dim, rank, bias=False)
        self.dna_proj = nn.Linear(dna_dim, rank, bias=False)
        self.protein_gate = nn.Linear(protein_dim, rank, bias=False)
        self.gating_strength = nn.Parameter(
            torch.tensor(float(initial_gating_strength))
        )

    def forward(self, protein_embedding, dna_embedding):
        protein_lowrank = self.protein_proj(protein_embedding)
        gate = (
            1.0
            + self.gating_strength
            * torch.tanh(self.protein_gate(protein_embedding))
        )
        protein_lowrank = F.normalize(
            protein_lowrank * gate, p=2, dim=-1, eps=1e-8
        )
        dna_lowrank = F.normalize(
            self.dna_proj(dna_embedding), p=2, dim=-1, eps=1e-8
        )
        return protein_lowrank * dna_lowrank


class PredictionHead(nn.Module):
    def __init__(self, protein_dim=256, dna_dim=64, interaction_dim=64):
        super().__init__()
        combined = protein_dim + dna_dim + interaction_dim
        self.net = nn.Sequential(
            nn.Linear(combined, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, protein_embedding, dna_embedding, interaction):
        x = torch.cat(
            [protein_embedding, dna_embedding, interaction],
            dim=1,
        )
        return self.net(x).squeeze(-1)


class ProteinDNABindingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dna_encoder = DNAEmbeddingCNN()
        self.protein_projection = ProteinProjection(
            input_dim=960,
            output_dim=256,
            dropout=0.3889,
        )
        self.bilinear = GatedCosineLowRankInteraction()
        self.prediction_head = PredictionHead()

    def forward(self, dna_onehot, protein_embedding):
        protein_embedding = self.protein_projection(protein_embedding)
        dna_embedding = self.dna_encoder(dna_onehot, protein_embedding)
        interaction = self.bilinear(protein_embedding, dna_embedding)
        return self.prediction_head(
            protein_embedding, dna_embedding, interaction
        )


def load_v7(fold):
    path = V7_DIR / f"best_model_fold_{fold}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing V7 checkpoint: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unexpected V7 checkpoint format: {path}")

    if "fold" in checkpoint and int(checkpoint["fold"]) != fold:
        raise ValueError(f"V7 fold mismatch in {path}")

    model = ProteinDNABindingModel()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(DEVICE).eval()
    return model


def v7_predict_batches(model, dna_device, protein_device, batch_size):
    out = []

    with torch.inference_mode():
        for start in range(0, len(dna_device), batch_size):
            end = min(start + batch_size, len(dna_device))
            dna = dna_device[start:end]
            protein = protein_device.unsqueeze(0).expand(
                end - start, -1
            )
            out.append(model(dna, protein).cpu())

    return torch.cat(out).numpy()


def v7_predict_fold(fold, dna_device, protein_device):
    model = load_v7(fold)
    batch = V7_BATCH

    while True:
        try:
            result = v7_predict_batches(
                model, dna_device, protein_device, batch
            )
            break
        except torch.cuda.OutOfMemoryError:
            if DEVICE.type != "cuda" or batch <= 64:
                raise
            batch = max(64, batch // 2)
            torch.cuda.empty_cache()

    del model
    return result


def v7_ensemble(dna_device, protein_device):
    total = np.zeros(len(dna_device), dtype=np.float64)

    for fold in range(1, N_FOLDS + 1):
        pred = v7_predict_fold(fold, dna_device, protein_device)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"V7 fold {fold} produced NaN/Inf.")
        total += pred

    return total / N_FOLDS


# =========================================================
# SimBind / MultiDBP architecture
# =========================================================

PARAMS = {
    "dropout": 0.362233801349954,
    "hidden1": 6029,
    "hidden2": 1168,
    "filters1": 2376,
    "hidden_sec": 152,
    "filters_sec": 151,
    "leaky_alpha": 0.23149394545024274,
    "filters_long_length": 24,
    "filters_long": 51,
}
PARAMS["merge_2"] = PARAMS["filters1"] * 4 + PARAMS["filters_long"]
PARAMS["output_layer"] = (
    PARAMS["hidden_sec"]
    + PARAMS["hidden2"]
    + PARAMS["hidden1"]
    + PARAMS["merge_2"]
)


class MultiTaskModel(nn.Module):
    def __init__(self, params, input_dim=4, output_dim=360):
        super().__init__()
        self.conv_kernel_long = nn.Conv1d(
            input_dim, params["filters_long"], params["filters_long_length"]
        )
        self.conv_kernel_11 = nn.Conv1d(
            input_dim, params["filters1"], 11
        )
        self.conv_kernel_9 = nn.Conv1d(
            input_dim, params["filters1"], 9
        )
        self.conv_kernel_7 = nn.Conv1d(
            input_dim, params["filters1"], 7
        )
        self.conv_kernel_5 = nn.Conv1d(
            input_dim, params["filters1"], 5
        )
        self.conv_kernel_5_sec = nn.Conv1d(
            input_dim, params["filters_sec"], 5
        )

        self.hidden_dense_relu = nn.Linear(
            params["merge_2"], params["hidden1"]
        )
        self.hidden_dense_relu1 = nn.Linear(
            params["hidden1"], params["hidden2"]
        )
        self.hidden_dense_sec = nn.Linear(
            params["filters_sec"], params["hidden_sec"]
        )
        self.output_layer = nn.Linear(
            params["output_layer"], output_dim
        )

        self.dropout = nn.Dropout(params["dropout"])
        self.leaky_relu = nn.LeakyReLU(params["leaky_alpha"])

    def features(self, x):
        x = x.permute(0, 2, 1)

        c_long = F.relu(self.conv_kernel_long(x))
        c_11 = F.relu(self.conv_kernel_11(x))
        c_9 = F.relu(self.conv_kernel_9(x))
        c_7 = F.relu(self.conv_kernel_7(x))
        c_5 = F.relu(self.conv_kernel_5(x))
        c_5_sec = F.relu(self.conv_kernel_5_sec(x))

        p_long = F.max_pool1d(c_long, c_long.size(-1)).flatten(1)
        p_11 = F.max_pool1d(c_11, c_11.size(-1)).flatten(1)
        p_9 = F.max_pool1d(c_9, c_9.size(-1)).flatten(1)
        p_7 = F.max_pool1d(c_7, c_7.size(-1)).flatten(1)
        p_5 = F.max_pool1d(c_5, c_5.size(-1)).flatten(1)
        p_5_sec = F.max_pool1d(c_5_sec, c_5_sec.size(-1)).flatten(1)

        merge_a = torch.cat(
            [p_11, p_7, p_long, p_9, p_5],
            dim=1,
        )
        merge_a_drop = self.dropout(merge_a)

        dense_a = F.relu(self.hidden_dense_relu(merge_a_drop))
        dense_a_drop = self.dropout(dense_a)
        dense_b = F.relu(self.hidden_dense_relu1(dense_a_drop))

        dense_sec = F.relu(
            self.hidden_dense_sec(self.dropout(p_5_sec))
        )

        return torch.cat(
            [dense_sec, dense_b, merge_a_drop, dense_a],
            dim=1,
        )

    def forward(self, x):
        return self.leaky_relu(self.output_layer(self.features(x)))

    def forward_selected(self, x, columns):
        """
        Exactly equivalent to model(x)[:, columns], but only calculates
        the 14 output neurons required by SimBind for the requested DBP.
        """
        z = self.features(x)
        weight = self.output_layer.weight[columns]
        bias = self.output_layer.bias[columns]
        return self.leaky_relu(F.linear(z, weight, bias))


def simbind_paths(fold):
    d = SIMBIND_DIR / f"fold_{fold}"
    return (
        d / f"MultiDBP_fold_{fold}.pt",
        d / f"MultiDBP_fold_{fold}_order_protein.pkl",
        SIMILARITY_DIR
        / f"fold_{fold}"
        / f"fold_{fold}_test_pairwise_AA_SID.csv",
    )


def load_simbind(fold):
    model_path, order_path, _ = simbind_paths(fold)

    if not model_path.is_file():
        raise FileNotFoundError(f"Missing SimBind model: {model_path}")
    if not order_path.is_file():
        raise FileNotFoundError(f"Missing SimBind protein order: {order_path}")

    with open(order_path, "rb") as f:
        order = list(pickle.load(f))

    if len(order) != 360:
        raise ValueError(
            f"Expected 360 proteins in {order_path}, got {len(order)}."
        )

    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model = MultiTaskModel(PARAMS, 4, 360)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()

    return model, order


def stable_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def load_precomputed_top14():
    if not PRECOMPUTED_TOP14_FILE.is_file():
        raise FileNotFoundError(
            f"Missing precomputed SimBind top14 file: "
            f"{PRECOMPUTED_TOP14_FILE}"
        )

    data = np.load(PRECOMPUTED_TOP14_FILE)

    columns = data["columns"]
    weights = data["weights"]

    if columns.shape != (64, 10, TOP_K):
        raise ValueError(
            f"Unexpected columns shape: {columns.shape}"
        )

    if weights.shape != (64, 10, TOP_K):
        raise ValueError(
            f"Unexpected weights shape: {weights.shape}"
        )

    return columns, weights


def top14(csv_path, query_protein, protein_order):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing similarity file: {csv_path}")

    to_col = {name: i for i, name in enumerate(protein_order)}
    candidates = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {
            "query_protein",
            "train_protein",
            "similarity_score",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected columns in {csv_path}")

        for row in reader:
            if row["query_protein"] != query_protein:
                continue

            train = row["train_protein"]
            if train not in to_col:
                continue

            candidates.append(
                (
                    float(row["similarity_score"]),
                    to_col[train],
                )
            )

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = candidates[:TOP_K]

    if len(selected) != TOP_K:
        raise RuntimeError(
            f"{query_protein} has only {len(selected)} usable "
            f"neighbors in fold similarity file."
        )

    scores = np.array([x[0] for x in selected], dtype=np.float64)
    columns = np.array([x[1] for x in selected], dtype=np.int64)
    weights = stable_softmax(scores * SOFTMAX_SCALE)

    return columns, weights


def simbind_predict_batches(model, dna_device, columns_np, weights_np, batch_size):
    columns = torch.as_tensor(
        columns_np, dtype=torch.long, device=DEVICE
    )
    weights = torch.as_tensor(
        weights_np, dtype=torch.float32, device=DEVICE
    )

    out = []

    with torch.inference_mode():
        for start in range(0, len(dna_device), batch_size):
            end = min(start + batch_size, len(dna_device))
            dna = dna_device[start:end]

            selected = model.forward_selected(dna, columns)
            pred = torch.sum(
                selected * weights.unsqueeze(0),
                dim=1,
            )
            out.append(pred.cpu())

    return torch.cat(out).numpy()


def simbind_predict_fold(
    fold,
    dna_device,
    test_index,
    precomputed_columns,
    precomputed_weights,
):
    model, order = load_simbind(fold)

    columns = precomputed_columns[
        test_index,
        fold - 1,
    ]
    weights = precomputed_weights[
        test_index,
        fold - 1,
    ]

    batch = SIMBIND_BATCH

    while True:
        try:
            result = simbind_predict_batches(
                model,
                dna_device,
                columns,
                weights,
                batch,
            )
            break
        except torch.cuda.OutOfMemoryError:
            if DEVICE.type != "cuda" or batch <= 16:
                raise
            batch = max(16, batch // 2)
            torch.cuda.empty_cache()

    del model
    return result


def simbind_ensemble(
    dna_device,
    test_index,
    precomputed_columns,
    precomputed_weights,
):
    total = np.zeros(len(dna_device), dtype=np.float64)

    for fold in range(1, N_FOLDS + 1):
        pred = simbind_predict_fold(
            fold,
            dna_device,
            test_index,
            precomputed_columns,
            precomputed_weights,
        )
        if not np.isfinite(pred).all():
            raise RuntimeError(f"SimBind fold {fold} produced NaN/Inf.")
        total += pred

    return total / N_FOLDS


# =========================================================
# Validation and output
# =========================================================

def validate_submission_files():
    missing = []

    if not EMBEDDINGS_FILE.is_file():
        missing.append(EMBEDDINGS_FILE)

    if not PRECOMPUTED_TOP14_FILE.is_file():
        missing.append(PRECOMPUTED_TOP14_FILE)

    for fold in range(1, N_FOLDS + 1):
        v7 = V7_DIR / f"best_model_fold_{fold}.pt"
        if not v7.is_file():
            missing.append(v7)

        model_path, order_path, _ = simbind_paths(fold)

        if not model_path.is_file():
            missing.append(model_path)

        if not order_path.is_file():
            missing.append(order_path)

    if missing:
        raise FileNotFoundError(
            "Missing required runtime files:\n"
            + "\n".join(str(p) for p in missing)
        )


def save_predictions(path, predictions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for value in predictions:
            f.write(f"{float(value):.10g}\n")


# =========================================================
# Main
# =========================================================

def main():
    if len(sys.argv) != 4:
        print(
            "Usage:\n"
            "  python main.py <ofile> <DBP> <DNA>\n\n"
            "Example:\n"
            "  python main.py DBP17.txt DBP17 test_seqs.txt",
            file=sys.stderr,
        )
        return 2

    output_file = sys.argv[1]
    dbp_name = sys.argv[2]
    dna_file = Path(sys.argv[3])

    if not dna_file.is_file():
        raise FileNotFoundError(f"DNA input file not found: {dna_file}")

    test_index, query_protein = parse_dbp(dbp_name)

    validate_submission_files()

    sequences = read_dna(dna_file)
    embeddings = load_embeddings()
    protein_embedding = embeddings[test_index]

    precomputed_columns, precomputed_weights = (
        load_precomputed_top14()
    )

    # Encode once on CPU, then move once to DEVICE.
    # All 10 V7 folds and all 10 SimBind folds reuse these device tensors.
    v7_dna, simbind_dna = encode_dna_inputs(sequences)
    v7_dna = v7_dna.to(DEVICE)
    simbind_dna = simbind_dna.to(DEVICE)
    protein_embedding = protein_embedding.to(DEVICE)

    # Equal-weight average of all 10 V7 folds.
    v7_prediction = v7_ensemble(
        v7_dna,
        protein_embedding,
    )

    # Equal-weight average of all 10 SimBind folds.
    simbind_prediction = simbind_ensemble(
        simbind_dna,
        test_index,
        precomputed_columns,
        precomputed_weights,
    )

    # Final 0.5 V7 + 0.5 SimBind ensemble.
    final_prediction = (
        V7_WEIGHT * v7_prediction
        + SIMBIND_WEIGHT * simbind_prediction
    )

    if len(final_prediction) != len(sequences):
        raise RuntimeError("Prediction count does not match DNA input count.")

    if not np.isfinite(final_prediction).all():
        raise RuntimeError("Final predictions contain NaN or Inf.")

    save_predictions(output_file, final_prediction)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())