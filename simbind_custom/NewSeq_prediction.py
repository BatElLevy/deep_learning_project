import os
import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
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

TRAINING_SEQS_FILE = os.path.join(
    PROJECT_ROOT,
    "training_seqs.txt"
)

MODEL_ROOT = os.path.join(
    SCRIPT_DIR,
    "models"
)

OUTPUT_ROOT = os.path.join(
    SCRIPT_DIR,
    "newseq_predictions"
)

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)


# =========================================================
# Dataset / CV constants
#
# Must match main_v7.py exactly.
# =========================================================

NUM_DNA = 30000
NUM_PROTEINS = 400

N_FOLDS = 10
RANDOM_SEED = 42


# =========================================================
# MultiDBP parameters
#
# Must match train_multi_task_newseq.py exactly.
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
# DNA loading
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

    if len(sequences) != NUM_DNA:

        raise ValueError(
            f"Expected {NUM_DNA} DNA sequences, "
            f"found {len(sequences)}."
        )

    return sequences


# =========================================================
# DNA one-hot encoding
#
# Must match training encoding exactly:
# 36 nt sequence padded to length 41.
# =========================================================

DNA_TO_INDEX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}


def encode_sequence(
    sequence,
    seq_length=41
):

    sequence = sequence.upper()

    tensor = np.zeros(
        (
            seq_length,
            4
        ),
        dtype=np.float32
    )

    for position, nucleotide in enumerate(
        sequence
    ):

        if position >= seq_length:
            break

        if nucleotide not in DNA_TO_INDEX:

            raise ValueError(
                f"Unknown DNA nucleotide "
                f"'{nucleotide}' "
                f"in sequence '{sequence}'."
            )

        tensor[
            position,
            DNA_TO_INDEX[nucleotide]
        ] = 1.0

    return tensor


# =========================================================
# Validation DNA dataset
# =========================================================

class ValidationDNADataset(
    Dataset
):

    def __init__(
        self,
        dna_sequences,
        dna_indices
    ):

        self.dna_sequences = (
            dna_sequences
        )

        self.dna_indices = (
            np.asarray(
                dna_indices
            )
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

        return (
            torch.tensor(
                x,
                dtype=torch.float32
            ),
            dna_idx
        )


# =========================================================
# MultiDBP architecture
#
# Must match train_multi_task_newseq.py exactly.
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
# Build exactly same DNA folds as main_v7.py
# =========================================================

def build_dna_folds():

    dna_indices = np.arange(
        NUM_DNA
    )

    dna_kfold = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    return list(
        dna_kfold.split(
            dna_indices
        )
    )


# =========================================================
# Predict one fold
# =========================================================

def predict_fold(
    fold_number,
    dna_sequences,
    dna_folds
):

    fold_idx = (
        fold_number - 1
    )

    (
        dna_train_idx,
        dna_val_idx
    ) = dna_folds[
        fold_idx
    ]


    if len(dna_train_idx) != 27000:

        raise RuntimeError(
            f"Expected 27000 training DNA, "
            f"got {len(dna_train_idx)}."
        )


    if len(dna_val_idx) != 3000:

        raise RuntimeError(
            f"Expected 3000 validation DNA, "
            f"got {len(dna_val_idx)}."
        )


    # -----------------------------------------------------
    # Fold model paths
    # -----------------------------------------------------

    fold_model_dir = os.path.join(
        MODEL_ROOT,
        f"fold_{fold_number}"
    )


    model_path = os.path.join(
        fold_model_dir,
        f"MultiDBP_fold_{fold_number}.pt"
    )


    protein_order_path = os.path.join(
        fold_model_dir,
        (
            f"MultiDBP_fold_"
            f"{fold_number}_"
            f"order_protein.pkl"
        )
    )


    split_path = os.path.join(
        fold_model_dir,
        f"fold_{fold_number}_split.npz"
    )


    for required_path in [
        model_path,
        protein_order_path,
        split_path
    ]:

        if not os.path.isfile(
            required_path
        ):

            raise FileNotFoundError(
                f"Missing required file:\n"
                f"{required_path}"
            )


    # -----------------------------------------------------
    # Check stored split against reconstructed split
    # -----------------------------------------------------

    saved_split = np.load(
        split_path
    )


    if not np.array_equal(
        saved_split[
            "dna_val"
        ],
        dna_val_idx
    ):

        raise RuntimeError(
            "DNA fold mismatch between "
            "training and prediction."
        )


    with open(
        protein_order_path,
        "rb"
    ) as f:

        protein_order = pickle.load(
            f
        )


    output_dim = len(
        protein_order
    )


    if output_dim != 360:

        raise RuntimeError(
            f"Expected 360 model outputs, "
            f"got {output_dim}."
        )


    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = MultiTaskModel(
        params=params_dict,
        input_dim=4,
        output_dim=output_dim
    ).to(
        DEVICE
    )


    state_dict = torch.load(
        model_path,
        map_location=DEVICE
    )


    model.load_state_dict(
        state_dict
    )

    model.eval()


    # -----------------------------------------------------
    # Validation DNA
    # -----------------------------------------------------

    dataset = ValidationDNADataset(
        dna_sequences=dna_sequences,
        dna_indices=dna_val_idx
    )


    loader = DataLoader(
        dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    print()
    print("=" * 70)

    print(
        f"NEWSEQ PREDICTION "
        f"FOLD {fold_number}/{N_FOLDS}"
    )

    print("=" * 70)

    print(
        "Validation DNA:",
        len(dataset)
    )

    print(
        "Training protein outputs:",
        output_dim
    )

    print(
        "Expected prediction shape:",
        (
            len(dataset),
            output_dim
        )
    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    all_predictions = []
    all_dna_indices = []


    with torch.no_grad():

        for x, dna_indices_batch in tqdm(
            loader,
            desc=f"Fold {fold_number}"
        ):

            x = x.to(
                DEVICE,
                non_blocking=True
            )

            predictions = model(
                x
            )

            all_predictions.append(
                predictions.cpu()
            )

            all_dna_indices.append(
                dna_indices_batch.numpy()
            )


    prediction_matrix = (
        torch.cat(
            all_predictions,
            dim=0
        )
        .numpy()
    )


    output_dna_indices = np.concatenate(
        all_dna_indices
    )


    # -----------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------

    expected_shape = (
        3000,
        360
    )


    if prediction_matrix.shape != expected_shape:

        raise RuntimeError(
            f"Unexpected prediction shape "
            f"{prediction_matrix.shape}; "
            f"expected {expected_shape}."
        )


    if not np.array_equal(
        output_dna_indices,
        dna_val_idx
    ):

        raise RuntimeError(
            "Validation DNA order changed "
            "during prediction."
        )


    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    fold_output_dir = os.path.join(
        OUTPUT_ROOT,
        f"fold_{fold_number}"
    )

    os.makedirs(
        fold_output_dir,
        exist_ok=True
    )


    prediction_path = os.path.join(
        fold_output_dir,
        (
            f"fold_{fold_number}_"
            f"newseq_predictions.npy"
        )
    )


    dna_indices_path = os.path.join(
        fold_output_dir,
        (
            f"fold_{fold_number}_"
            f"dna_val_indices.npy"
        )
    )


    protein_order_output_path = os.path.join(
        fold_output_dir,
        (
            f"fold_{fold_number}_"
            f"protein_order.pkl"
        )
    )


    np.save(
        prediction_path,
        prediction_matrix
    )


    np.save(
        dna_indices_path,
        output_dna_indices
    )


    with open(
        protein_order_output_path,
        "wb"
    ) as f:

        pickle.dump(
            protein_order,
            f
        )


    print()
    print(
        "Prediction shape:",
        prediction_matrix.shape
    )

    print(
        "Saved predictions:",
        prediction_path
    )

    print(
        "Saved DNA order:",
        dna_indices_path
    )

    print(
        "Saved protein order:",
        protein_order_output_path
    )

    print("=" * 70)


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate leakage-safe "
            "MultiDBP NewSeq predictions "
            "for held-out DNA using "
            "the same folds as main_v7.py."
        )
    )


    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help=(
            "Fold to run (1-10). "
            "If omitted, all folds "
            "are processed."
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


    dna_sequences = (
        load_dna_sequences(
            TRAINING_SEQS_FILE
        )
    )


    dna_folds = (
        build_dna_folds()
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

        predict_fold(
            fold_number=fold_number,
            dna_sequences=dna_sequences,
            dna_folds=dna_folds
        )


if __name__ == "__main__":
    main()