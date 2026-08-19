#!/usr/bin/env python
# coding: utf-8

import json
import math
import os
import random
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm


# =========================================================
# Original training configuration
# =========================================================

NUM_PROTEINS = 400
NUM_DNA = 30000

N_FOLDS = 10
RANDOM_SEED = 42

BATCH_SIZE_TRAIN = 256
BATCH_SIZE_VAL = 512

MAX_EPOCHS = 40
PATIENCE = 10

LEARNING_RATE = 2.67e-4
WEIGHT_DECAY = 3.86e-4

CHECKPOINT_DIR = "checkpoints_v7"
RESULTS_DIR = "cv_results_v7"

TRAINING_DATA_ZIP = "training_data.zip"
TRAINING_SEQS_FILE = "training_seqs.txt"
PROTEIN_EMBEDDINGS_FILE = "train_protein_embeddings.pt"


# =========================================================
# Environment / input checks
# =========================================================

def print_environment():
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU count:", torch.cuda.device_count())


def check_input_files():
    for f in [
        "training_data.zip",
        "training_seqs.txt",
        "train_protein_embeddings.pt"
    ]:
        print(f, "->", os.path.exists(f))



# =========================================================
# Gated Pooling
# =========================================================

class GatedPooling1D(nn.Module):
    """
    Learns an importance score for every position and computes
    a weighted average of the sequence features.

    Input:
        x: [batch_size, channels, sequence_length]

    Output:
        pooled: [batch_size, channels]
    """

    def __init__(self, channels):
        super().__init__()

        self.gate = nn.Conv1d(
            in_channels=channels,
            out_channels=1,
            kernel_size=1
        )

    def forward(self, x):

        # Position scores: [B, 1, L]
        gate_scores = self.gate(x)

        # Normalize scores across sequence positions
        attention_weights = torch.softmax(
            gate_scores,
            dim=2
        )

        # Weighted average across positions
        pooled = torch.sum(
            x * attention_weights,
            dim=2
        )

        return pooled


# =========================================================
# Dilated Residual Convolution Block
#
# Every block preserves:
#   channels = 64
#   sequence length = 36
#
# Structure:
#   Conv1D
#   → GELU
#   → LayerNorm
#   → Dropout
#   → Residual addition
# =========================================================

class DilatedResidualConvBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        dilation,
        dropout
    ):
        super().__init__()

        # Padding that preserves sequence length
        padding = (
            dilation * (kernel_size - 1)
        ) // 2

        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )

        self.activation = nn.GELU()

        self.layer_norm = nn.LayerNorm(
            channels
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):
        """
        x:
            [batch_size, channels, sequence_length]

        returns:
            [batch_size, channels, sequence_length]
        """

        residual = x

        x = self.conv(x)
        x = self.activation(x)

        # LayerNorm operates on the final dimension,
        # so temporarily move channels to the end:
        # [B, C, L] -> [B, L, C]
        x = x.transpose(1, 2)

        x = self.layer_norm(x)

        # [B, L, C] -> [B, C, L]
        x = x.transpose(1, 2)

        x = self.dropout(x)

        # Residual connection
        x = x + residual

        return x


# =========================================================
# DNA Embedding Network
#
# Architecture:
#
# One-hot DNA [B, 4, 36]
#   → Input projection 4 -> 64
#   → Residual Conv(kernel=5,  dilation=1)
#   → Residual Conv(kernel=9,  dilation=2)
#   → Residual Conv(kernel=13, dilation=4)
#   → Extra Conv(kernel=9) with learnable residual weight
#   → Max Pooling + Gated Pooling
#   → Projection 128 -> 64
#
# Output:
#   DNA embedding [B, 64]
# =========================================================

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
        transformer_dropout=0.2653
    ):
        super().__init__()

        # -------------------------------------------------
        # Project one-hot nucleotides from 4 to 64 channels
        # -------------------------------------------------

        self.input_projection = nn.Conv1d(
            in_channels=4,
            out_channels=model_dim,
            kernel_size=1
        )

        # -------------------------------------------------
        # Learned positional embedding
        # -------------------------------------------------

        self.sequence_length = sequence_length

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                sequence_length,
                model_dim
            )
        )

        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02
        )

        # -------------------------------------------------
        # Three sequential dilated residual blocks
        # -------------------------------------------------

        self.conv_block_1 = DilatedResidualConvBlock(
            channels=model_dim,
            kernel_size=5,
            dilation=1,
            dropout=dropout
        )

        self.conv_block_2 = DilatedResidualConvBlock(
            channels=model_dim,
            kernel_size=9,
            dilation=2,
            dropout=dropout
        )

        self.conv_block_3 = DilatedResidualConvBlock(
            channels=model_dim,
            kernel_size=13,
            dilation=4,
            dropout=dropout
        )

        # -------------------------------------------------
        # Additional kernel-9 convolution from the paper
        # -------------------------------------------------

        self.extra_conv = nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=9,
            padding=4
        )

        # Learnable residual weight.
        # Starting at zero means the extra convolution is
        # introduced gradually during training.
        self.extra_conv_weight = nn.Parameter(
            torch.tensor(0.0)
        )

        # -------------------------------------------------
        # Transformer encoder
        #
        # Input/output:
        #   [batch_size, sequence_length, model_dim]
        #   [B, 36, 64]
        # -------------------------------------------------

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_feedforward_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(model_dim)
        )

        # -------------------------------------------------
        # Protein-conditioned residual FiLM
        #
        # The projected protein embedding generates one
        # scale and one shift for every DNA feature channel.
        # conditioning_strength starts at zero, so this
        # entire branch is an exact identity at initialization.
        # -------------------------------------------------

        self.protein_conditioner = nn.Sequential(
            nn.LayerNorm(protein_dim),
            nn.Linear(protein_dim, model_dim * 2)
        )

        self.conditioning_strength = nn.Parameter(
            torch.tensor(0.0)
        )

        # -------------------------------------------------
        # Max + Gated Pooling
        # -------------------------------------------------

        self.gated_pool = GatedPooling1D(
            channels=model_dim
        )

        # Max pooling gives 64 features and gated pooling
        # gives 64 features: 64 + 64 = 128
        self.output_layer_norm = nn.LayerNorm(
            model_dim * 2
        )

        self.output_dropout = nn.Dropout(
            dropout
        )

        self.output_projection = nn.Linear(
            model_dim * 2,
            64
        )

        self.output_activation = nn.GELU()

    def forward(self, x, protein_embedding):
        """
        x:
            [batch_size, 4, 36]

        returns:
            dna_embedding: [batch_size, 64]
        """

        # -------------------------------------------------
        # Input projection
        # -------------------------------------------------

        x = self.input_projection(x)
        # [B, 64, 36]

        # -------------------------------------------------
        # Add learned positional information
        # -------------------------------------------------

        sequence_length = x.size(2)

        if sequence_length > self.sequence_length:
            raise ValueError(
                f"DNA sequence length {sequence_length} exceeds "
                f"configured maximum {self.sequence_length}"
            )

        # [B, 64, 36] -> [B, 36, 64]
        x = x.transpose(1, 2)

        x = (
            x
            +
            self.position_embedding[
                :,
                :sequence_length,
                :
            ]
        )

        # [B, 36, 64] -> [B, 64, 36]
        x = x.transpose(1, 2)

        # -------------------------------------------------
        # Sequential dilated residual convolutions
        # -------------------------------------------------

        x = self.conv_block_1(x)
        # [B, 64, 36]

        x = self.conv_block_2(x)
        # [B, 64, 36]

        x = self.conv_block_3(x)
        # [B, 64, 36]

        # -------------------------------------------------
        # Additional kernel-9 residual convolution
        # -------------------------------------------------

        extra_features = self.extra_conv(x)

        x = (
            x
            +
            self.extra_conv_weight
            * extra_features
        )
        # [B, 64, 36]

        # -------------------------------------------------
        # Transformer
        # -------------------------------------------------

        # [B, 64, 36] -> [B, 36, 64]
        x = x.transpose(1, 2)

        x = self.transformer(x)
        # [B, 36, 64]

        # [B, 36, 64] -> [B, 64, 36]
        x = x.transpose(1, 2)

        # -------------------------------------------------
        # Protein-conditioned residual FiLM
        # -------------------------------------------------

        conditioning = self.protein_conditioner(
            protein_embedding
        )
        # [B, 128]

        gamma, beta = conditioning.chunk(
            2,
            dim=1
        )
        # gamma: [B, 64]
        # beta:  [B, 64]

        gamma = torch.tanh(gamma).unsqueeze(2)
        beta = torch.tanh(beta).unsqueeze(2)
        # [B, 64, 1]

        residual_update = (
            gamma * x
            + beta
        )

        x = (
            x
            + torch.tanh(
                self.conditioning_strength
            ) * residual_update
        )
        # [B, 64, 36]

        # -------------------------------------------------
        # Max + Gated Pooling
        # -------------------------------------------------

        max_pooled = torch.max(
            x,
            dim=2
        ).values
        # [B, 64]

        gated_pooled = self.gated_pool(x)
        # [B, 64]

        pooled_features = torch.cat(
            [
                max_pooled,
                gated_pooled
            ],
            dim=1
        )
        # [B, 128]

        # -------------------------------------------------
        # Final DNA embedding
        # -------------------------------------------------

        pooled_features = self.output_layer_norm(
            pooled_features
        )

        pooled_features = self.output_dropout(
            pooled_features
        )

        dna_embedding = self.output_activation(
            self.output_projection(
                pooled_features
            )
        )
        # [B, 64]

        return dna_embedding

# In[ ]:


import torch
import torch.nn as nn



# =========================================================
# Protein Projection
# ESM-C embedding: 960 -> task-specific protein embedding: 256
# =========================================================

class ProteinProjection(nn.Module):
    def __init__(self, input_dim=960, output_dim=256, dropout=0.15):
        super().__init__()

        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, protein_embedding):
        """
        protein_embedding: [batch_size, 960]

        returns:
            projected_protein: [batch_size, 256]
        """

        x = self.layer_norm(protein_embedding)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x




# =========================================================
# Gated Low-Rank Cosine Bilinear Interaction
#
# Protein: 256 -> 64
# DNA:      64 -> 64
#
# Based on the NucProNet interaction:
#
# protein = normalize(
#     U(protein) * (1 + lambda * tanh(G(protein)))
# )
#
# dna = normalize(V(dna))
#
# interaction = protein * dna
# =========================================================

class GatedCosineLowRankInteraction(nn.Module):
    def __init__(
        self,
        protein_dim=256,
        dna_dim=64,
        rank=64,
        initial_gating_strength=1.0
    ):
        super().__init__()

        # Low-rank projections
        self.protein_proj = nn.Linear(
            protein_dim,
            rank,
            bias=False
        )

        self.dna_proj = nn.Linear(
            dna_dim,
            rank,
            bias=False
        )

        # Protein-dependent gate
        self.protein_gate = nn.Linear(
            protein_dim,
            rank,
            bias=False
        )

        # Learnable gating-strength scalar
        self.gating_strength = nn.Parameter(
            torch.tensor(
                float(initial_gating_strength)
            )
        )

    def forward(
        self,
        protein_embedding,
        dna_embedding
    ):
        """
        protein_embedding: [batch_size, 256]
        dna_embedding:     [batch_size, 64]

        returns:
            interaction_vector: [batch_size, 64]
        """

        # Basic protein projection
        protein_lowrank = self.protein_proj(
            protein_embedding
        )

        # Protein-dependent multiplicative gate
        gate = (
            1.0
            +
            self.gating_strength
            * torch.tanh(
                self.protein_gate(
                    protein_embedding
                )
            )
        )

        gated_protein = (
            protein_lowrank * gate
        )

        # DNA projection
        dna_lowrank = self.dna_proj(
            dna_embedding
        )

        # L2 normalization creates cosine-style interaction
        gated_protein = F.normalize(
            gated_protein,
            p=2,
            dim=-1,
            eps=1e-8
        )

        dna_lowrank = F.normalize(
            dna_lowrank,
            p=2,
            dim=-1,
            eps=1e-8
        )

        # Keep the full interaction vector for the prediction head
        interaction_vector = (
            gated_protein * dna_lowrank
        )

        return interaction_vector


class PredictionHead(nn.Module):
    def __init__(
        self,
        protein_dim=256,
        dna_dim=64,
        interaction_dim=64
    ):
        super().__init__()

        combined_dim = (
            protein_dim
            + dna_dim
            + interaction_dim
        )

        self.net = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(128, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(
        self,
        protein_embedding,
        dna_embedding,
        interaction_vector
    ):
        """
        protein_embedding:  [batch_size, 256]
        dna_embedding:      [batch_size, 64]
        interaction_vector: [batch_size, 64]

        returns:
            predictions: [batch_size]
        """

        x = torch.cat(
            [
                protein_embedding,
                dna_embedding,
                interaction_vector
            ],
            dim=1
        )
        # x: [batch_size, 384]

        prediction = self.net(x)

        return prediction.squeeze(-1)


# In[ ]:


import torch
import torch.nn as nn



class ProteinDNABindingModel(nn.Module):
    def __init__(
        self,
        model_dim=64,
        protein_dim=256,
        dna_dropout=0.2653,
        transformer_layers=2,
        transformer_heads=4,
        transformer_feedforward_dim=256,
        transformer_dropout=0.2653,
        protein_dropout=0.3889,
        interaction_rank=64,
        initial_gating_strength=1.0
    ):
        super().__init__()

        # DNA: [B, 4, 36] -> [B, 64]
        self.dna_encoder = DNAEmbeddingCNN(
            model_dim=model_dim,
            protein_dim=protein_dim,
            dropout=dna_dropout,
            sequence_length=36,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_feedforward_dim=transformer_feedforward_dim,
            transformer_dropout=transformer_dropout
        )

        # Protein ESM-C embedding: [B, 960] -> [B, 256]
        self.protein_projection = ProteinProjection(
            input_dim=960,
            output_dim=protein_dim,
            dropout=protein_dropout
        )

        # Protein 256 + DNA 64 -> gated cosine interaction 64
        self.bilinear = GatedCosineLowRankInteraction(
            protein_dim=protein_dim,
            dna_dim=64,
            rank=interaction_rank,
            initial_gating_strength=initial_gating_strength
        )

        # [Protein 256 + DNA 64 + Interaction 64] = 384
        self.prediction_head = PredictionHead(
            protein_dim=protein_dim,
            dna_dim=64,
            interaction_dim=interaction_rank
        )

    def forward(self, dna_onehot, protein_embedding):
        """
        dna_onehot:
            [batch_size, 4, 36]

        protein_embedding:
            [batch_size, 960]

        returns:
            predictions: [batch_size]
        """

        # 1. Protein representation
        protein_embedding = self.protein_projection(protein_embedding)
        # [B, 256]

        # 2. Protein-conditioned DNA representation
        dna_embedding = self.dna_encoder(
            dna_onehot,
            protein_embedding
        )
        # [B, 64]

        # 3. Protein-DNA interaction
        interaction_vector = self.bilinear(
            protein_embedding,
            dna_embedding
        )
        # [B, 64]

        # 4. Final binding prediction
        prediction = self.prediction_head(
            protein_embedding,
            dna_embedding,
            interaction_vector
        )
        # [B]

        return prediction


# =========================================================
# 1. Load binding intensity matrix
# Shape: [30000 DNA, 400 proteins]
# =========================================================

def load_binding_matrix(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("training_data.txt") as f:
            binding_matrix = np.loadtxt(f, dtype=np.float32)

    print("Binding matrix shape:", binding_matrix.shape)

    return binding_matrix


# =========================================================
# 2. Load DNA sequences
# =========================================================

def load_dna_sequences(file_path):
    sequences = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if line and not line.startswith(">"):
                sequences.append(line)

    print("Number of DNA sequences:", len(sequences))

    return sequences


# =========================================================
# 3. DNA one-hot encoding
# A,C,G,T -> [4, sequence_length]
# =========================================================

DNA_TO_INDEX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}


def one_hot_encode_dna(sequence):
    one_hot = torch.zeros(
        4,
        len(sequence),
        dtype=torch.float32
    )

    for position, nucleotide in enumerate(sequence):
        if nucleotide in DNA_TO_INDEX:
            one_hot[DNA_TO_INDEX[nucleotide], position] = 1.0

    return one_hot


# =========================================================
# 4. Dataset
# =========================================================

class ProteinDNADataset(Dataset):

    def __init__(
        self,
        dna_sequences,
        protein_embeddings,
        binding_matrix,
        dna_indices,
        protein_indices
    ):
        """
        dna_sequences:
            list of all 30,000 DNA sequences

        protein_embeddings:
            saved ESM-C embeddings for all 400 proteins

        binding_matrix:
            [30000, 400]

        dna_indices:
            DNA indices belonging to this split

        protein_indices:
            protein indices belonging to this split
        """

        self.dna_sequences = dna_sequences
        self.protein_embeddings = protein_embeddings
        self.binding_matrix = binding_matrix

        self.dna_indices = np.asarray(dna_indices)
        self.protein_indices = np.asarray(protein_indices)

        self.num_dna = len(self.dna_indices)
        self.num_proteins = len(self.protein_indices)

    def __len__(self):
        # Number of DNA-protein pairs in this block
        return self.num_dna * self.num_proteins

    def __getitem__(self, index):

        # -------------------------------------------------
        # Convert one linear index into:
        # DNA index + protein index
        # -------------------------------------------------

        dna_position = index // self.num_proteins
        protein_position = index % self.num_proteins

        dna_idx = self.dna_indices[dna_position]
        protein_idx = self.protein_indices[protein_position]

        # -------------------------------------------------
        # DNA
        # -------------------------------------------------

        dna_sequence = self.dna_sequences[dna_idx]

        dna_onehot = one_hot_encode_dna(dna_sequence)
        # [4, 36]

        # -------------------------------------------------
        # Protein
        # -------------------------------------------------

        protein_embedding = self.protein_embeddings[int(protein_idx)].float()
        # [960]

        # -------------------------------------------------
        # Target binding intensity
        # -------------------------------------------------

        target = torch.tensor(
            self.binding_matrix[dna_idx, protein_idx],
            dtype=torch.float32
        )

        return dna_onehot, protein_embedding, target



class StructuredProteinDNABatchSampler(Sampler):
    """
    Creates structured training batches:

        up to 8 proteins × up to 64 DNA sequences

    Every protein in the batch is paired with the SAME
    DNA sequences.

    Across one epoch, every protein-DNA pair in the
    training split is used exactly once.
    """

    def __init__(
        self,
        dataset,
        proteins_per_batch=8,
        dna_per_protein=64,
        shuffle=True
    ):
        self.dataset = dataset

        self.proteins_per_batch = proteins_per_batch
        self.dna_per_protein = dna_per_protein
        self.shuffle = shuffle

        self.num_proteins = dataset.num_proteins
        self.num_dna = dataset.num_dna

    def __iter__(self):

        # -----------------------------------------------
        # Shuffle proteins and DNA independently
        # at the beginning of every epoch
        # -----------------------------------------------

        protein_positions = list(
            range(self.num_proteins)
        )

        dna_positions = list(
            range(self.num_dna)
        )

        if self.shuffle:
            random.shuffle(protein_positions)
            random.shuffle(dna_positions)

        # -----------------------------------------------
        # Split proteins into groups of 8
        # and DNA into groups of 64
        # -----------------------------------------------

        protein_groups = [
            protein_positions[i:i + self.proteins_per_batch]
            for i in range(
                0,
                self.num_proteins,
                self.proteins_per_batch
            )
        ]

        dna_groups = [
            dna_positions[i:i + self.dna_per_protein]
            for i in range(
                0,
                self.num_dna,
                self.dna_per_protein
            )
        ]

        # If the last group is smaller than 64,
        # pad it using DNA sequences from the beginning
        # so that no DNA sequence is dropped.
        if len(dna_groups[-1]) < self.dna_per_protein:

            missing = (
                self.dna_per_protein
                - len(dna_groups[-1])
            )

            dna_groups[-1].extend(
                dna_positions[:missing]
            )

        # Shuffle the order of the structured batches
        batch_combinations = [
            (protein_group, dna_group)
            for protein_group in protein_groups
            for dna_group in dna_groups
        ]

        if self.shuffle:
            random.shuffle(batch_combinations)

        # -----------------------------------------------
        # Build each batch
        #
        # IMPORTANT:
        # protein-major ordering:
        #
        # P1 × all selected DNA
        # P2 × all selected DNA
        # ...
        # -----------------------------------------------

        for protein_group, dna_group in batch_combinations:

            batch_indices = []

            for protein_position in protein_group:

                for dna_position in dna_group:

                    # This matches ProteinDNADataset.__getitem__
                    dataset_index = (
                        dna_position * self.num_proteins
                        + protein_position
                    )

                    batch_indices.append(
                        dataset_index
                    )

            yield batch_indices

    def __len__(self):

        num_protein_groups = math.ceil(
            self.num_proteins
            / self.proteins_per_batch
        )

        num_dna_groups = math.ceil(
            self.num_dna
            / self.dna_per_protein
        )

        return (
            num_protein_groups
            * num_dna_groups
        )

# =========================================================
# Loss: Huber + Pearson
#
# NucProNet hyperparameters:
#   Huber weight:   0.3478
#   Pearson weight: 0.6522
#   Huber delta:    1.0
# =========================================================

class StructuredHuberPearsonLoss(nn.Module):
    def __init__(
        self,
        proteins_per_batch=8,
        dna_per_protein=64,
        huber_weight=0.3478,
        pearson_weight=0.6522,
        huber_delta=1.0,
        eps=1e-8
    ):
        super().__init__()

        self.proteins_per_batch = proteins_per_batch
        self.dna_per_protein = dna_per_protein

        self.huber_weight = huber_weight
        self.pearson_weight = pearson_weight

        self.eps = eps

        self.huber = nn.HuberLoss(
            delta=huber_delta
        )

    def forward(self, predictions, targets):

        # -------------------------------------------------
        # Huber loss over all protein-DNA pairs
        # -------------------------------------------------

        huber_loss = self.huber(
            predictions,
            targets
        )

        # -------------------------------------------------
        # Determine actual batch structure
        # -------------------------------------------------

        batch_size = predictions.shape[0]

        current_dna_per_protein = min(
            self.dna_per_protein,
            batch_size
        )

        current_num_proteins = (
            batch_size // current_dna_per_protein
        )

        # [num_proteins, dna_per_protein]
        predictions_matrix = predictions.reshape(
            current_num_proteins,
            current_dna_per_protein
        )

        targets_matrix = targets.reshape(
            current_num_proteins,
            current_dna_per_protein
        )

        # -------------------------------------------------
        # Pearson separately for every protein
        # -------------------------------------------------

        pearsons = []

        for protein_idx in range(
            current_num_proteins
        ):

            protein_predictions = predictions_matrix[
                protein_idx
            ]

            protein_targets = targets_matrix[
                protein_idx
            ]

            pred_centered = (
                protein_predictions
                - protein_predictions.mean()
            )

            target_centered = (
                protein_targets
                - protein_targets.mean()
            )

            numerator = torch.sum(
                pred_centered * target_centered
            )

            denominator = (
                torch.sqrt(
                    torch.sum(
                        pred_centered ** 2
                    ) + self.eps
                )
                *
                torch.sqrt(
                    torch.sum(
                        target_centered ** 2
                    ) + self.eps
                )
            )

            pearson = numerator / denominator

            pearsons.append(
                pearson
            )

        mean_pearson = torch.stack(
            pearsons
        ).mean()

        pearson_loss = (
            1.0 - mean_pearson
        )

        # -------------------------------------------------
        # Final loss
        # -------------------------------------------------

        total_loss = (
            self.huber_weight * huber_loss
            +
            self.pearson_weight * pearson_loss
        )

        return total_loss

# =========================================================
# Pearson helper
# IMPORTANT:
# Pearson is computed on the FULL validation set,
# not batch-by-batch
# =========================================================

def compute_pearson(predictions, targets, eps=1e-8):

    predictions = predictions.float()
    targets = targets.float()

    pred_centered = predictions - predictions.mean()
    target_centered = targets - targets.mean()

    numerator = torch.sum(
        pred_centered * target_centered
    )

    denominator = (
        torch.sqrt(torch.sum(pred_centered ** 2) + eps)
        *
        torch.sqrt(torch.sum(target_centered ** 2) + eps)
    )

    pearson = numerator / denominator

    return pearson.item()


# =========================================================
# Train one epoch
# =========================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    num_samples = 0

    start_time = time.time()

    progress_bar = tqdm(
        train_loader,
        desc="Training",
        leave=False
    )

    for dna_batch, protein_batch, targets in progress_bar:

        dna_batch = dna_batch.to(
            device,
            non_blocking=True
        )

        protein_batch = protein_batch.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        # Clear gradients
        optimizer.zero_grad()

        # Forward
        predictions = model(
            dna_batch,
            protein_batch
        )

        # Loss
        loss = criterion(
            predictions,
            targets
        )

        # Backward
        loss.backward()

        # Update weights
        optimizer.step()

        batch_size = targets.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        num_samples += batch_size

        current_loss = (
            running_loss / num_samples
        )

        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}"
        )

    epoch_loss = (
        running_loss / num_samples
    )

    elapsed_time = time.time() - start_time

    return epoch_loss, elapsed_time


# =========================================================
# Validation
# =========================================================

def validate(
    model,
    val_loader,
    criterion,
    device
):

    model.eval()

    running_huber = 0.0
    num_samples = 0

    all_predictions = []
    all_targets = []

    start_time = time.time()

    # Validation regression loss is reported as plain Huber.
    # Model selection is still based on full validation Pearson.
    huber_criterion = torch.nn.HuberLoss(
        delta=1.0
    )

    progress_bar = tqdm(
        val_loader,
        desc="Validation",
        leave=False
    )

    with torch.no_grad():

        for dna_batch, protein_batch, targets in progress_bar:

            dna_batch = dna_batch.to(
                device,
                non_blocking=True
            )

            protein_batch = protein_batch.to(
                device,
                non_blocking=True
            )

            targets = targets.to(
                device,
                non_blocking=True
            )

            predictions = model(
                dna_batch,
                protein_batch
            )

            # Validation loss = Huber only
            huber_loss = huber_criterion(
                predictions,
                targets
            )

            batch_size = targets.size(0)

            running_huber += (
                huber_loss.item() * batch_size
            )

            num_samples += batch_size

            all_predictions.append(
                predictions.detach().cpu()
            )

            all_targets.append(
                targets.detach().cpu()
            )

            current_huber = (
                running_huber / num_samples
            )

            progress_bar.set_postfix(
                huber=f"{current_huber:.4f}"
            )

    # -----------------------------------------------------
    # Validation Huber
    # -----------------------------------------------------

    val_loss = (
        running_huber / num_samples
    )

    all_predictions = torch.cat(
        all_predictions
    )

    all_targets = torch.cat(
        all_targets
    )

    # -----------------------------------------------------
    # Competition-style validation Pearson:
    # Pearson separately for every protein
    # across ALL validation DNA probes
    # -----------------------------------------------------

    num_val_proteins = (
        val_loader.dataset.num_proteins
    )

    predictions_matrix = all_predictions.reshape(
        -1,
        num_val_proteins
    )

    targets_matrix = all_targets.reshape(
        -1,
        num_val_proteins
    )

    protein_pearsons = []

    for protein_idx in range(
        num_val_proteins
    ):

        protein_predictions = predictions_matrix[
            :,
            protein_idx
        ]

        protein_targets = targets_matrix[
            :,
            protein_idx
        ]

        pearson = compute_pearson(
            protein_predictions,
            protein_targets
        )

        protein_pearsons.append(
            pearson
        )

    # Mean score across validation proteins
    val_pearson = float(
        np.mean(protein_pearsons)
    )

    elapsed_time = (
        time.time() - start_time
    )

    return (
        val_loss,
        val_pearson,
        elapsed_time
    )



# =========================================================
# Original small shape test, now as a function
# =========================================================

def run_interaction_shape_test():
    batch_size = 8

    # Simulates saved ESM-C embeddings
    protein_esm_embedding = torch.randn(batch_size, 960)

    # Simulates output of the DNA CNN
    dna_embedding = torch.randn(batch_size, 64)

    protein_projection = ProteinProjection(
        input_dim=960,
        output_dim=256,
        dropout=0.15
    )

    bilinear = GatedCosineLowRankInteraction(
        protein_dim=256,
        dna_dim=64,
        rank=64,
        initial_gating_strength=1.0
    )

    # 960 -> 256
    protein_embedding = protein_projection(protein_esm_embedding)

    # Protein 256 + DNA 64 -> interaction 64
    interaction = bilinear(protein_embedding, dna_embedding)

    print("ESM-C protein embedding:", protein_esm_embedding.shape)
    print("Projected protein embedding:", protein_embedding.shape)
    print("DNA embedding:", dna_embedding.shape)
    print("Interaction vector:", interaction.shape)


# =========================================================
# Original full-model shape test, now as a function
# =========================================================

def run_model_shape_test():
    batch_size = 8

    dummy_dna = torch.randn(batch_size, 4, 36)
    dummy_protein = torch.randn(batch_size, 960)

    model = ProteinDNABindingModel()

    with torch.no_grad():
        predictions = model(dummy_dna, dummy_protein)

    print("DNA input:", dummy_dna.shape)
    print("Protein input:", dummy_protein.shape)
    print("Predictions:", predictions.shape)


# =========================================================
# 2D fold construction
# Original logic preserved
# =========================================================

def build_folds():
    protein_indices = np.arange(NUM_PROTEINS)
    dna_indices = np.arange(NUM_DNA)

    protein_kfold = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    dna_kfold = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    protein_folds = list(
        protein_kfold.split(protein_indices)
    )

    dna_folds = list(
        dna_kfold.split(dna_indices)
    )

    folds = []

    for fold_idx in range(N_FOLDS):
        protein_train_idx, protein_val_idx = protein_folds[fold_idx]
        dna_train_idx, dna_val_idx = dna_folds[fold_idx]

        fold = {
            "protein_train": protein_train_idx,
            "protein_val": protein_val_idx,

            "dna_train": dna_train_idx,
            "dna_val": dna_val_idx
        }

        folds.append(fold)

    return folds


# =========================================================
# Original split verification
# =========================================================

def verify_folds(folds):
    for fold_idx, fold in enumerate(folds):
        protein_overlap = np.intersect1d(
            fold["protein_train"],
            fold["protein_val"]
        )

        dna_overlap = np.intersect1d(
            fold["dna_train"],
            fold["dna_val"]
        )

        print(f"\nFold {fold_idx + 1}")
        print("-" * 40)

        print(
            "Proteins:",
            len(fold["protein_train"]),
            "train |",
            len(fold["protein_val"]),
            "validation"
        )

        print(
            "DNA:",
            len(fold["dna_train"]),
            "train |",
            len(fold["dna_val"]),
            "validation"
        )

        print(
            "Protein overlap:",
            len(protein_overlap)
        )

        print(
            "DNA overlap:",
            len(dna_overlap)
        )


# =========================================================
# Original Fold 1 access example
# =========================================================

def print_fold_1_example(folds):
    fold_1 = folds[0]

    train_proteins = fold_1["protein_train"]
    val_proteins = fold_1["protein_val"]

    train_dna = fold_1["dna_train"]
    val_dna = fold_1["dna_val"]

    print("\nExample - Fold 1")
    print("Train proteins:", train_proteins.shape)
    print("Validation proteins:", val_proteins.shape)
    print("Train DNA:", train_dna.shape)
    print("Validation DNA:", val_dna.shape)


# =========================================================
# Original data loading, now as a function
# =========================================================

def load_all_data():
    binding_matrix = load_binding_matrix(
        TRAINING_DATA_ZIP
    )

    dna_sequences = load_dna_sequences(
        TRAINING_SEQS_FILE
    )

    protein_embeddings = torch.load(
        PROTEIN_EMBEDDINGS_FILE,
        map_location="cpu"
    )

    assert binding_matrix.shape == (30000, 400)
    assert len(dna_sequences) == 30000
    assert len(protein_embeddings) == 400

    print("All data loaded successfully.")

    return (
        binding_matrix,
        dna_sequences,
        protein_embeddings
    )


# =========================================================
# Original Fold 1 Dataset/DataLoader test
# =========================================================

def build_fold_1_test_loaders(
    folds,
    dna_sequences,
    protein_embeddings,
    binding_matrix
):
    fold = folds[0]

    train_dataset = ProteinDNADataset(
        dna_sequences=dna_sequences,
        protein_embeddings=protein_embeddings,
        binding_matrix=binding_matrix,
        dna_indices=fold["dna_train"],
        protein_indices=fold["protein_train"]
    )

    val_dataset = ProteinDNADataset(
        dna_sequences=dna_sequences,
        protein_embeddings=protein_embeddings,
        binding_matrix=binding_matrix,
        dna_indices=fold["dna_val"],
        protein_indices=fold["protein_val"]
    )

    print("Train pairs:", len(train_dataset))
    print("Validation pairs:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader


# =========================================================
# Original one-batch loss test
# =========================================================

def run_one_batch_test(train_loader, device):
    model = ProteinDNABindingModel().to(device)

    criterion = StructuredHuberPearsonLoss(
        proteins_per_batch=8,
        dna_per_protein=64,
        huber_weight=0.3478,
        pearson_weight=0.6522,
        huber_delta=1.0
    )

    # Original smoke-test optimizer settings preserved.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    dna_batch, protein_batch, targets = next(
        iter(train_loader)
    )

    dna_batch = dna_batch.to(device)
    protein_batch = protein_batch.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        predictions = model(
            dna_batch,
            protein_batch
        )

        loss = criterion(
            predictions,
            targets
        )

    print("DNA batch:", dna_batch.shape)
    print("Protein batch:", protein_batch.shape)
    print("Targets:", targets.shape)
    print("Predictions:", predictions.shape)

    print("Test loss:", loss.item())

    return model, criterion, optimizer


def print_target_statistics(binding_matrix):
    print("Target min:", binding_matrix.min())
    print("Target max:", binding_matrix.max())
    print("Target mean:", binding_matrix.mean())
    print("Target std:", binding_matrix.std())


def print_device_status(device):
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None
    )


# =========================================================
# Original FOLD_TO_RUN environment-variable behavior
# =========================================================

def get_fold_to_run():
    fold_to_run = os.environ.get("FOLD_TO_RUN")

    if fold_to_run is not None:
        fold_to_run = int(fold_to_run)

        if not 1 <= fold_to_run <= N_FOLDS:
            raise ValueError(
                f"FOLD_TO_RUN must be between 1 and {N_FOLDS}"
            )

    return fold_to_run


# =========================================================
# Fold-specific target normalization
# Original logic preserved exactly
# =========================================================

def normalize_targets_for_fold(
    binding_matrix,
    fold
):
    train_dna_idx = fold["dna_train"]

    fold_target_means = binding_matrix[
        train_dna_idx, :
    ].mean(
        axis=0,
        keepdims=True
    )

    fold_target_stds = binding_matrix[
        train_dna_idx, :
    ].std(
        axis=0,
        keepdims=True
    )

    fold_target_stds = np.where(
        fold_target_stds < 1e-8,
        1.0,
        fold_target_stds
    )

    fold_binding_matrix = (
        binding_matrix
        - fold_target_means
    ) / fold_target_stds

    fold_binding_matrix = (
        fold_binding_matrix.astype(
            np.float32
        )
    )

    return fold_binding_matrix


# =========================================================
# Fold DataLoaders
# Original structured sampler behavior preserved
# =========================================================

def build_training_loaders(
    fold,
    dna_sequences,
    protein_embeddings,
    fold_binding_matrix
):
    train_dataset = ProteinDNADataset(
        dna_sequences=dna_sequences,
        protein_embeddings=protein_embeddings,
        binding_matrix=fold_binding_matrix,
        dna_indices=fold["dna_train"],
        protein_indices=fold["protein_train"]
    )

    val_dataset = ProteinDNADataset(
        dna_sequences=dna_sequences,
        protein_embeddings=protein_embeddings,
        binding_matrix=fold_binding_matrix,
        dna_indices=fold["dna_val"],
        protein_indices=fold["protein_val"]
    )

    train_batch_sampler = StructuredProteinDNABatchSampler(
        train_dataset,
        proteins_per_batch=8,
        dna_per_protein=64,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader


# =========================================================
# Fresh model/loss/optimizer for every fold
# =========================================================

def build_model(
    model_dim=64,
    protein_dim=256,
    dna_dropout=0.2653,
    transformer_layers=2,
    transformer_heads=4,
    transformer_feedforward_dim=256,
    transformer_dropout=0.2653,
    protein_dropout=0.3889,
    interaction_rank=64,
    initial_gating_strength=1.0
):
    return ProteinDNABindingModel(
        model_dim=model_dim,
        protein_dim=protein_dim,
        dna_dropout=dna_dropout,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_feedforward_dim=transformer_feedforward_dim,
        transformer_dropout=transformer_dropout,
        protein_dropout=protein_dropout,
        interaction_rank=interaction_rank,
        initial_gating_strength=initial_gating_strength
    )


def build_criterion(
    huber_weight=0.3478,
    pearson_weight=0.6522,
    huber_delta=1.0
):
    return StructuredHuberPearsonLoss(
        proteins_per_batch=8,
        dna_per_protein=64,
        huber_weight=huber_weight,
        pearson_weight=pearson_weight,
        huber_delta=huber_delta
    )


def build_optimizer(
    model,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
):
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

def build_training_objects(device):
    model = build_model().to(device)
    criterion = build_criterion()
    optimizer = build_optimizer(model)

    return model, criterion, optimizer


# =========================================================
# Original complete fold training, now as a function
# =========================================================

def run_fold(
    fold_idx,
    fold,
    binding_matrix,
    dna_sequences,
    protein_embeddings,
    device,
    model_dim=64,
    protein_dim=256,
    dna_dropout=0.2653,
    transformer_layers=2,
    transformer_heads=4,
    transformer_feedforward_dim=256,
    transformer_dropout=0.2653,
    protein_dropout=0.3889,
    interaction_rank=64,
    initial_gating_strength=1.0,
    huber_weight=0.3478,
    pearson_weight=0.6522,
    huber_delta=1.0,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    save_checkpoint=True
):
    fold_start_time = time.time()

    print("\n")
    print("=" * 70)
    print(
        f"FOLD {fold_idx + 1}/{N_FOLDS}"
    )
    print("=" * 70)

    fold_binding_matrix = normalize_targets_for_fold(
        binding_matrix,
        fold
    )

    print(
        "Target normalization: "
        "statistics computed from training DNA only"
    )

    train_loader, val_loader = build_training_loaders(
        fold,
        dna_sequences,
        protein_embeddings,
        fold_binding_matrix
    )

    print(
        "Train pairs:",
        len(train_loader.dataset)
    )

    print(
        "Validation pairs:",
        len(val_loader.dataset)
    )

    model = build_model(
        model_dim=model_dim,
        protein_dim=protein_dim,
        dna_dropout=dna_dropout,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_feedforward_dim=transformer_feedforward_dim,
        transformer_dropout=transformer_dropout,
        protein_dropout=protein_dropout,
        interaction_rank=interaction_rank,
        initial_gating_strength=initial_gating_strength
    ).to(device)

    criterion = build_criterion(
        huber_weight=huber_weight,
        pearson_weight=pearson_weight,
        huber_delta=huber_delta
    )

    optimizer = build_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    best_val_pearson = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"best_model_fold_{fold_idx + 1}.pt"
    )

    for epoch in range(
        1,
        max_epochs + 1
    ):
        print(
            f"\nFold {fold_idx + 1} "
            f"| Epoch {epoch}/{max_epochs}"
        )

        train_loss, train_time = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_pearson, val_time = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Val Huber:  {val_loss:.4f}"
        )

        print(
            f"Val Pearson: {val_pearson:.4f}"
        )

        print(
            f"Train time: {train_time / 60:.2f} min"
        )

        print(
            f"Val time:   {val_time / 60:.2f} min"
        )

        print(
            f"Epoch total: "
            f"{(train_time + val_time) / 60:.2f} min"
        )

        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            best_epoch = epoch

            epochs_without_improvement = 0

            if save_checkpoint:
                torch.save(
                    {
                        "fold": fold_idx + 1,
                        "epoch": epoch,
                        "model_state_dict":
                            model.state_dict(),
                        "optimizer_state_dict":
                            optimizer.state_dict(),
                        "val_pearson":
                            val_pearson,
                        "val_loss":
                            val_loss
                    },
                    checkpoint_path
                )

                print(
                    f"New best model saved "
                    f"(Pearson={val_pearson:.4f})"
                )

        else:
            epochs_without_improvement += 1

            print(
                f"No improvement "
                f"({epochs_without_improvement}"
                f"/{patience})"
            )

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                f"Early stopping "
                f"at epoch {epoch}"
            )

            break

    fold_elapsed = (
        time.time()
        - fold_start_time
    )

    fold_result_data = {
        "fold": fold_idx + 1,
        "best_epoch": best_epoch,
        "best_val_pearson": best_val_pearson,
        "checkpoint": checkpoint_path if save_checkpoint else None,
        "runtime_seconds": fold_elapsed
    }

    if save_checkpoint:
        fold_result_path = os.path.join(
            RESULTS_DIR,
            f"fold_{fold_idx + 1}_result.json"
        )

        with open(fold_result_path, "w") as f:
            json.dump(
                fold_result_data,
                f,
                indent=2
            )

        print(
            f"Fold result saved to: "
            f"{fold_result_path}"
        )

    print("\n")
    print("-" * 70)

    print(
        f"Fold {fold_idx + 1} finished"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation Pearson: "
        f"{best_val_pearson:.4f}"
    )

    print(
        f"Fold runtime: "
        f"{fold_elapsed / 60:.2f} min"
    )

    print("-" * 70)

    return fold_result_data


# =========================================================
# Original saved-results summary
# =========================================================

def load_all_completed_fold_results():
    all_fold_results = []

    for fold_number in range(1, N_FOLDS + 1):
        result_path = os.path.join(
            RESULTS_DIR,
            f"fold_{fold_number}_result.json"
        )

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result = json.load(f)

            all_fold_results.append(result)

    return sorted(
        all_fold_results,
        key=lambda x: x["fold"]
    )


def print_cross_validation_summary():
    all_fold_results = load_all_completed_fold_results()

    print("\n")
    print("=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    for result in all_fold_results:
        print(
            f"Fold {result['fold']}: "
            f"Pearson={result['best_val_pearson']:.4f} "
            f"| Best epoch={result['best_epoch']} "
            f"| Runtime={result['runtime_seconds'] / 60:.2f} min"
        )

    pearsons = [
        result["best_val_pearson"]
        for result in all_fold_results
    ]

    print(
        "\nCompleted folds:",
        f"{len(all_fold_results)}/{N_FOLDS}"
    )

    if len(pearsons) > 0:
        print(
            "Mean Pearson:",
            f"{np.mean(pearsons):.4f}"
        )

        print(
            "Std Pearson:",
            f"{np.std(pearsons):.4f}"
        )

    if len(all_fold_results) == N_FOLDS:
        print("\n")
        print("=" * 70)
        print("FULL CROSS-VALIDATION COMPLETE")
        print("=" * 70)

        print(
            "Mean Pearson:",
            f"{np.mean(pearsons):.4f}"
        )

        print(
            "Std Pearson:",
            f"{np.std(pearsons):.4f}"
        )

        print("=" * 70)

    else:
        missing_folds = [
            fold_number
            for fold_number in range(1, N_FOLDS + 1)
            if not os.path.exists(
                os.path.join(
                    RESULTS_DIR,
                    f"fold_{fold_number}_result.json"
                )
            )
        ]

        print(
            "Still waiting for folds:",
            missing_folds
        )


# =========================================================
# Original cross-validation behavior
# - no FOLD_TO_RUN: run all folds sequentially
# - FOLD_TO_RUN=N: run only that fold
# This preserves the possibility of launching several
# separate processes in parallel with different FOLD_TO_RUN values.
# =========================================================

def run_cross_validation(
    folds,
    binding_matrix,
    dna_sequences,
    protein_embeddings,
    device,
    fold_to_run
):
    fold_results = []

    if fold_to_run is None:
        fold_indices = range(len(folds))
    else:
        fold_indices = [fold_to_run - 1]

    for fold_idx in fold_indices:
        fold = folds[fold_idx]

        result = run_fold(
            fold_idx=fold_idx,
            fold=fold,
            binding_matrix=binding_matrix,
            dna_sequences=dna_sequences,
            protein_embeddings=protein_embeddings,
            device=device
        )

        fold_results.append(result)

    return fold_results


# =========================================================
# Main
# =========================================================

def main():
    # Same folders as original.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Original startup checks.
    print_environment()
    check_input_files()

    # Original shape checks.
    run_interaction_shape_test()
    run_model_shape_test()

    # Original folds construction and checks.
    folds = build_folds()
    verify_folds(folds)
    print_fold_1_example(folds)

    # Original data loading.
    (
        binding_matrix,
        dna_sequences,
        protein_embeddings
    ) = load_all_data()

    # Original Fold 1 Dataset/DataLoader smoke test.
    train_loader, _ = build_fold_1_test_loaders(
        folds=folds,
        dna_sequences=dna_sequences,
        protein_embeddings=protein_embeddings,
        binding_matrix=binding_matrix
    )

    # Original first device setup and one-batch loss test.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    run_one_batch_test(
        train_loader=train_loader,
        device=device
    )

    # Original statistics checks.
    print_target_statistics(binding_matrix)
    print_device_status(device)

    # Original optional single-fold behavior.
    fold_to_run = get_fold_to_run()

    # Original full CV timer.
    full_run_start = time.time()

    run_cross_validation(
        folds=folds,
        binding_matrix=binding_matrix,
        dna_sequences=dna_sequences,
        protein_embeddings=protein_embeddings,
        device=device,
        fold_to_run=fold_to_run
    )

    # Original summary scans all saved fold result files,
    # including folds that may have been produced by parallel processes.
    print_cross_validation_summary()

    full_run_elapsed = time.time() - full_run_start

    print("\n")
    print("=" * 70)
    print(
        f"TOTAL CV RUNTIME: "
        f"{full_run_elapsed / 60:.2f} minutes"
    )
    print(
        f"TOTAL CV RUNTIME: "
        f"{full_run_elapsed / 3600:.2f} hours"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()