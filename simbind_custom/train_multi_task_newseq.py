import os
import zipfile
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.model_selection import KFold
from tqdm import tqdm


# =========================================================
# Paths
# =========================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SCRIPT_DIR
)

TRAINING_DATA_ZIP = os.path.join(
    PROJECT_ROOT,
    "training_data.zip"
)

TRAINING_SEQS_FILE = os.path.join(
    PROJECT_ROOT,
    "training_seqs.txt"
)

OUTPUT_ROOT = os.path.join(
    SCRIPT_DIR,
    "models"
)

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)


# =========================================================
# Dataset constants
# Must exactly match main_v7.py
# =========================================================

NUM_PROTEINS = 400
NUM_DNA = 30000

N_FOLDS = 10
RANDOM_SEED = 42


# =========================================================
# MultiDBP parameters
#
# Keep the original SimBind / MultiDBP architecture and
# training hyperparameters.
# =========================================================

params_dict = {
    "dropout": 0.362233801349954,
    "epochs": 72,
    "batch": 512,
    "regu": 0.0,
    "hidden1": 6029,
    "hidden2": 1168,
    "filters1": 2376,
    "hidden_sec": 152,
    "filters_sec": 151,
    "leaky_alpha": 0.23149394545024274,
    "filters_long_length": 24,
    "filters_long": 51
}

params_dict["merge_2"] = (
    params_dict["filters1"] * 4
    + params_dict["filters_long"]
)

params_dict["output_layer"] = (
    params_dict["hidden_sec"]
    + params_dict["hidden2"]
    + params_dict["hidden1"]
    + params_dict["merge_2"]
)


# =========================================================
# Device
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# Stable protein IDs
#
# IMPORTANT:
# training_DBPs.txt has no FASTA headers.
#
# Therefore protein index i is always named:
#
#     protein_0000
#     protein_0001
#     ...
#     protein_0399
#
# The same naming convention will later be used by
# protein_domain.py / pair_wise.py.
# =========================================================

def protein_id(index):
    return f"protein_{int(index):04d}"


# =========================================================
# Load binding matrix
#
# Original shape:
#     [30000 DNA, 400 proteins]
# =========================================================

def load_binding_matrix(zip_path):

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as z:

        with z.open(
            "training_data.txt"
        ) as f:

            matrix = np.loadtxt(
                f,
                dtype=np.float32
            )

    print(
        "Binding matrix shape:",
        matrix.shape
    )

    if matrix.shape != (
        NUM_DNA,
        NUM_PROTEINS
    ):
        raise ValueError(
            "Unexpected binding matrix shape: "
            f"{matrix.shape}. "
            f"Expected {(NUM_DNA, NUM_PROTEINS)}."
        )

    return matrix


# =========================================================
# Load DNA sequences
# =========================================================

def load_dna_sequences(
    file_path
):

    sequences = []

    with open(
        file_path,
        "r"
    ) as f:

        for line in f:

            line = line.strip()

            if (
                line
                and not line.startswith(">")
            ):
                sequences.append(
                    line
                )

    print(
        "Number of DNA sequences:",
        len(sequences)
    )

    if len(sequences) != NUM_DNA:
        raise ValueError(
            f"Expected {NUM_DNA} DNA sequences, "
            f"found {len(sequences)}."
        )

    return sequences


# =========================================================
# DNA one-hot encoding
#
# MultiDBP expects:
#
#     [sequence_length, 4]
#
# Our DNA probes are length 36.
#
# The original implementation used a default padded length
# of 41. We preserve that behavior here.
# =========================================================

NUCLEIC_ACIDS = "ACGT"

NN_TO_IX = {
    nucleotide: i
    for i, nucleotide
    in enumerate(
        NUCLEIC_ACIDS
    )
}


def encode_sequence(
    seq,
    seq_length=41
):

    seq = seq.upper()

    tensor = np.zeros(
        (
            seq_length,
            4
        ),
        dtype=np.float32
    )

    for i, char in enumerate(
        seq
    ):

        if i >= seq_length:
            break

        if char not in NN_TO_IX:
            raise ValueError(
                f"Unknown DNA nucleotide '{char}' "
                f"in sequence '{seq}'."
            )

        tensor[
            i,
            NN_TO_IX[char]
        ] = 1.0

    return tensor


# =========================================================
# Multi-task dataset
#
# One item =
#
#     DNA sequence
#     +
#     vector of binding values for ALL training proteins
#
# So for Fold X:
#
#     x: [41, 4]
#     y: [360]
# =========================================================

class MultiDBPDataset(
    Dataset
):

    def __init__(
        self,
        dna_sequences,
        target_matrix,
        dna_indices
    ):

        self.dna_sequences = (
            dna_sequences
        )

        self.target_matrix = (
            target_matrix
        )

        self.dna_indices = np.asarray(
            dna_indices
        )

    def __len__(self):

        return len(
            self.dna_indices
        )

    def __getitem__(
        self,
        idx
    ):

        dna_idx = int(
            self.dna_indices[
                idx
            ]
        )

        sequence = (
            self.dna_sequences[
                dna_idx
            ]
        )

        x = encode_sequence(
            sequence,
            seq_length=41
        )

        y = (
            self.target_matrix[
                dna_idx
            ]
        )

        return (
            torch.tensor(
                x,
                dtype=torch.float32
            ),
            torch.tensor(
                y,
                dtype=torch.float32
            )
        )


# =========================================================
# Original MultiDBP architecture
# =========================================================

class MultiTaskModel(
    nn.Module
):

    def __init__(
        self,
        params,
        input_dim,
        output_dim
    ):

        super().__init__()

        self.conv_kernel_long = (
            nn.Conv1d(
                input_dim,
                params[
                    "filters_long"
                ],
                params[
                    "filters_long_length"
                ]
            )
        )

        self.conv_kernel_11 = (
            nn.Conv1d(
                input_dim,
                params[
                    "filters1"
                ],
                11
            )
        )

        self.conv_kernel_9 = (
            nn.Conv1d(
                input_dim,
                params[
                    "filters1"
                ],
                9
            )
        )

        self.conv_kernel_7 = (
            nn.Conv1d(
                input_dim,
                params[
                    "filters1"
                ],
                7
            )
        )

        self.conv_kernel_5 = (
            nn.Conv1d(
                input_dim,
                params[
                    "filters1"
                ],
                5
            )
        )

        self.conv_kernel_5_sec = (
            nn.Conv1d(
                input_dim,
                params[
                    "filters_sec"
                ],
                5
            )
        )

        self.hidden_dense_relu = (
            nn.Linear(
                params[
                    "merge_2"
                ],
                params[
                    "hidden1"
                ]
            )
        )

        self.hidden_dense_relu1 = (
            nn.Linear(
                params[
                    "hidden1"
                ],
                params[
                    "hidden2"
                ]
            )
        )

        self.hidden_dense_sec = (
            nn.Linear(
                params[
                    "filters_sec"
                ],
                params[
                    "hidden_sec"
                ]
            )
        )

        self.output_layer = (
            nn.Linear(
                params[
                    "output_layer"
                ],
                output_dim
            )
        )

        self.dropout = nn.Dropout(
            params[
                "dropout"
            ]
        )

        self.leaky_relu = (
            nn.LeakyReLU(
                negative_slope=params[
                    "leaky_alpha"
                ]
            )
        )

    def forward(
        self,
        x
    ):

        # [B, 41, 4]
        # ->
        # [B, 4, 41]

        x = x.permute(
            0,
            2,
            1
        )

        c_long = F.relu(
            self.conv_kernel_long(
                x
            )
        )

        c_11 = F.relu(
            self.conv_kernel_11(
                x
            )
        )

        c_9 = F.relu(
            self.conv_kernel_9(
                x
            )
        )

        c_7 = F.relu(
            self.conv_kernel_7(
                x
            )
        )

        c_5 = F.relu(
            self.conv_kernel_5(
                x
            )
        )

        c_5_sec = F.relu(
            self.conv_kernel_5_sec(
                x
            )
        )

        p_long = (
            F.max_pool1d(
                c_long,
                c_long.size(-1)
            )
            .flatten(1)
        )

        p_11 = (
            F.max_pool1d(
                c_11,
                c_11.size(-1)
            )
            .flatten(1)
        )

        p_9 = (
            F.max_pool1d(
                c_9,
                c_9.size(-1)
            )
            .flatten(1)
        )

        p_7 = (
            F.max_pool1d(
                c_7,
                c_7.size(-1)
            )
            .flatten(1)
        )

        p_5 = (
            F.max_pool1d(
                c_5,
                c_5.size(-1)
            )
            .flatten(1)
        )

        p_5_sec = (
            F.max_pool1d(
                c_5_sec,
                c_5_sec.size(-1)
            )
            .flatten(1)
        )

        merge_a = torch.cat(
            [
                p_11,
                p_7,
                p_long,
                p_9,
                p_5
            ],
            dim=1
        )

        merge_a_drop = (
            self.dropout(
                merge_a
            )
        )

        dense_a = F.relu(
            self.hidden_dense_relu(
                merge_a_drop
            )
        )

        dense_a_drop = (
            self.dropout(
                dense_a
            )
        )

        dense_b = F.relu(
            self.hidden_dense_relu1(
                dense_a_drop
            )
        )

        dense_sec = F.relu(
            self.hidden_dense_sec(
                self.dropout(
                    p_5_sec
                )
            )
        )

        final_features = (
            torch.cat(
                [
                    dense_sec,
                    dense_b,
                    merge_a_drop,
                    dense_a
                ],
                dim=1
            )
        )

        output = (
            self.output_layer(
                final_features
            )
        )

        return self.leaky_relu(
            output
        )


# =========================================================
# Original LogCosh loss
# =========================================================

class LogCoshLoss(
    nn.Module
):

    def forward(
        self,
        y_pred,
        y_true
    ):

        diff = torch.clamp(
            y_pred - y_true,
            min=-80,
            max=80
        )

        return torch.mean(
            torch.log(
                torch.cosh(
                    diff
                )
            )
        )


# =========================================================
# Train one epoch
# =========================================================

def train_one_epoch(
    dataloader,
    model,
    loss_fn,
    optimizer
):

    model.train()

    total_loss = 0.0
    num_batches = 0

    progress = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for x, y in progress:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        pred = model(
            x
        )

        loss = loss_fn(
            pred,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
        )

        num_batches += 1

        progress.set_postfix(
            loss=(
                f"{total_loss / num_batches:.4f}"
            )
        )

    return (
        total_loss
        / max(
            num_batches,
            1
        )
    )


# =========================================================
# Evaluate on unseen DNA
#
# IMPORTANT:
# This is ONLY diagnostic.
#
# We DO NOT use this score for:
#     - early stopping
#     - model selection
#     - hyperparameter tuning
#
# Therefore the 3000 DNA validation probes remain held out
# from training decisions.
# =========================================================

def evaluate(
    dataloader,
    model
):

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():

        for x, y in tqdm(
            dataloader,
            desc="Evaluating",
            leave=False
        ):

            x = x.to(
                DEVICE,
                non_blocking=True
            )

            pred = model(
                x
            )

            all_preds.append(
                pred.cpu()
            )

            all_targets.append(
                y.cpu()
            )

    preds = torch.cat(
        all_preds,
        dim=0
    ).numpy()

    targets = torch.cat(
        all_targets,
        dim=0
    ).numpy()

    correlations = []

    for protein_idx in range(
        preds.shape[1]
    ):

        correlation = (
            stats.pearsonr(
                targets[
                    :,
                    protein_idx
                ],
                preds[
                    :,
                    protein_idx
                ]
            )[0]
        )

        correlations.append(
            correlation
        )

    mean_pearson = float(
        np.nanmean(
            correlations
        )
    )

    return (
        preds,
        targets,
        mean_pearson
    )


# =========================================================
# Build exactly the same 10-fold split as main_v7.py
# =========================================================

def build_folds():

    protein_indices = np.arange(
        NUM_PROTEINS
    )

    dna_indices = np.arange(
        NUM_DNA
    )

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
        protein_kfold.split(
            protein_indices
        )
    )

    dna_folds = list(
        dna_kfold.split(
            dna_indices
        )
    )

    folds = []

    for fold_idx in range(
        N_FOLDS
    ):

        (
            protein_train_idx,
            protein_val_idx
        ) = protein_folds[
            fold_idx
        ]

        (
            dna_train_idx,
            dna_val_idx
        ) = dna_folds[
            fold_idx
        ]

        folds.append(
            {
                "protein_train":
                    protein_train_idx,

                "protein_val":
                    protein_val_idx,

                "dna_train":
                    dna_train_idx,

                "dna_val":
                    dna_val_idx
            }
        )

    return folds


# =========================================================
# Train one fold
# =========================================================

def train_fold(
    fold_number,
    binding_matrix,
    dna_sequences,
    folds
):

    fold_idx = (
        fold_number - 1
    )

    fold = folds[
        fold_idx
    ]

    protein_train_idx = (
        fold[
            "protein_train"
        ]
    )

    protein_val_idx = (
        fold[
            "protein_val"
        ]
    )

    dna_train_idx = (
        fold[
            "dna_train"
        ]
    )

    dna_val_idx = (
        fold[
            "dna_val"
        ]
    )

    print()
    print("=" * 70)

    print(
        f"SIMBIND MULTIDBP "
        f"FOLD {fold_number}/{N_FOLDS}"
    )

    print("=" * 70)

    print(
        "Training proteins:",
        len(
            protein_train_idx
        )
    )

    print(
        "Validation proteins:",
        len(
            protein_val_idx
        )
    )

    print(
        "Training DNA:",
        len(
            dna_train_idx
        )
    )

    print(
        "Validation DNA:",
        len(
            dna_val_idx
        )
    )


    # -----------------------------------------------------
    # Leakage-safe normalization
    #
    # Statistics are calculated ONLY from training DNA.
    #
    # We need statistics only for the 360 training proteins
    # because those are the tasks learned by MultiDBP.
    # -----------------------------------------------------

    train_targets_raw = (
        binding_matrix[
            :,
            protein_train_idx
        ]
    )

    target_means = (
        train_targets_raw[
            dna_train_idx,
            :
        ]
        .mean(
            axis=0,
            keepdims=True
        )
    )

    target_stds = (
        train_targets_raw[
            dna_train_idx,
            :
        ]
        .std(
            axis=0,
            keepdims=True
        )
    )

    target_stds = np.where(
        target_stds < 1e-8,
        1.0,
        target_stds
    )

    normalized_targets = (
        (
            train_targets_raw
            - target_means
        )
        / target_stds
    ).astype(
        np.float32
    )


    # -----------------------------------------------------
    # Datasets
    # -----------------------------------------------------

    train_dataset = (
        MultiDBPDataset(
            dna_sequences=dna_sequences,
            target_matrix=normalized_targets,
            dna_indices=dna_train_idx
        )
    )

    validation_dna_dataset = (
        MultiDBPDataset(
            dna_sequences=dna_sequences,
            target_matrix=normalized_targets,
            dna_indices=dna_val_idx
        )
    )


    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=params_dict[
                "batch"
            ],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    )

    validation_dna_loader = (
        DataLoader(
            validation_dna_dataset,
            batch_size=4096,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    )


    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    output_dim = len(
        protein_train_idx
    )

    if output_dim != 360:
        raise ValueError(
            f"Expected 360 training proteins, "
            f"got {output_dim}."
        )

    model = MultiTaskModel(
        params=params_dict,
        input_dim=4,
        output_dim=output_dim
    ).to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_fn = (
        LogCoshLoss()
    )


    # -----------------------------------------------------
    # Fixed 72 epochs
    #
    # Deliberately NO early stopping on dna_val.
    # -----------------------------------------------------

    print()
    print(
        f"Training for "
        f"{params_dict['epochs']} "
        f"fixed epochs..."
    )

    for epoch in range(
        1,
        params_dict[
            "epochs"
        ] + 1
    ):

        epoch_loss = (
            train_one_epoch(
                dataloader=train_loader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer
            )
        )

        print(
            f"Fold {fold_number} "
            f"| Epoch "
            f"{epoch}/"
            f"{params_dict['epochs']} "
            f"| Loss={epoch_loss:.4f}"
        )


    # -----------------------------------------------------
    # Diagnostic NewSeq evaluation
    #
    # This asks:
    #
    # Can MultiDBP predict unseen DNA for the 360 proteins
    # it DID train on?
    #
    # This score is not used for training decisions.
    # -----------------------------------------------------

    (
        _,
        _,
        newseq_pearson
    ) = evaluate(
        validation_dna_loader,
        model
    )

    print()
    print(
        "Diagnostic MultiDBP "
        "NewSeq Pearson "
        f"on held-out DNA: "
        f"{newseq_pearson:.4f}"
    )


    # -----------------------------------------------------
    # Fold output directory
    # -----------------------------------------------------

    fold_output_dir = os.path.join(
        OUTPUT_ROOT,
        f"fold_{fold_number}"
    )

    os.makedirs(
        fold_output_dir,
        exist_ok=True
    )


    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    model_path = os.path.join(
        fold_output_dir,
        f"MultiDBP_fold_{fold_number}.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )


    # -----------------------------------------------------
    # Save exact output protein order
    #
    # Column j of the model output corresponds exactly to
    # protein_order[j].
    # -----------------------------------------------------

    protein_order = [
        protein_id(index)
        for index
        in protein_train_idx
    ]

    protein_order_path = (
        os.path.join(
            fold_output_dir,
            (
                f"MultiDBP_fold_"
                f"{fold_number}_"
                f"order_protein.pkl"
            )
        )
    )

    with open(
        protein_order_path,
        "wb"
    ) as f:

        pickle.dump(
            protein_order,
            f
        )


    # -----------------------------------------------------
    # Also save indices explicitly.
    #
    # This makes later leakage checks trivial.
    # -----------------------------------------------------

    split_path = os.path.join(
        fold_output_dir,
        f"fold_{fold_number}_split.npz"
    )

    np.savez(
        split_path,

        protein_train=(
            protein_train_idx
        ),

        protein_val=(
            protein_val_idx
        ),

        dna_train=(
            dna_train_idx
        ),

        dna_val=(
            dna_val_idx
        )
    )


    # -----------------------------------------------------
    # Save normalization statistics.
    #
    # These are not strictly required by SimBind because
    # Pearson is scale invariant, but keeping them makes the
    # experiment fully reproducible.
    # -----------------------------------------------------

    normalization_path = (
        os.path.join(
            fold_output_dir,
            (
                f"fold_{fold_number}_"
                f"target_normalization.npz"
            )
        )
    )

    np.savez(
        normalization_path,
        mean=target_means,
        std=target_stds
    )


    print()
    print(
        "Saved model:",
        model_path
    )

    print(
        "Saved protein order:",
        protein_order_path
    )

    print(
        "Saved split:",
        split_path
    )

    print(
        "Saved normalization:",
        normalization_path
    )

    print("=" * 70)


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train leakage-safe "
            "SimBind MultiDBP models "
            "using the same 10-fold split "
            "as main_v7.py"
        )
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help=(
            "Fold to run (1-10). "
            "If omitted, all 10 folds "
            "are trained sequentially."
        )
    )

    args = parser.parse_args()


    if (
        args.fold is not None
        and not (
            1
            <= args.fold
            <= N_FOLDS
        )
    ):
        raise ValueError(
            "--fold must be "
            "between 1 and 10."
        )


    print(
        "Device:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )


    binding_matrix = (
        load_binding_matrix(
            TRAINING_DATA_ZIP
        )
    )

    dna_sequences = (
        load_dna_sequences(
            TRAINING_SEQS_FILE
        )
    )

    folds = build_folds()


    # -----------------------------------------------------
    # Sanity check identical CV structure to main_v7
    # -----------------------------------------------------

    for idx, fold in enumerate(
        folds,
        start=1
    ):

        assert (
            len(
                fold[
                    "protein_train"
                ]
            )
            == 360
        )

        assert (
            len(
                fold[
                    "protein_val"
                ]
            )
            == 40
        )

        assert (
            len(
                fold[
                    "dna_train"
                ]
            )
            == 27000
        )

        assert (
            len(
                fold[
                    "dna_val"
                ]
            )
            == 3000
        )

    print(
        "10-fold split verified."
    )


    if args.fold is None:

        fold_numbers = range(
            1,
            N_FOLDS + 1
        )

    else:

        fold_numbers = [
            args.fold
        ]


    for fold_number in fold_numbers:

        train_fold(
            fold_number=fold_number,
            binding_matrix=binding_matrix,
            dna_sequences=dna_sequences,
            folds=folds
        )


if __name__ == "__main__":
    main()