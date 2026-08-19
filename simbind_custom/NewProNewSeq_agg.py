import os
import csv
import pickle
import argparse
import zipfile

import numpy as np
import pandas as pd

from scipy.special import softmax
from scipy import stats
from sklearn.model_selection import KFold


# =========================================================
# Paths
# =========================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SCRIPT_DIR
)

SIMILARITY_ROOT = os.path.join(
    SCRIPT_DIR,
    "similarities"
)

NEWSEQ_ROOT = os.path.join(
    SCRIPT_DIR,
    "newseq_predictions"
)

MODEL_ROOT = os.path.join(
    SCRIPT_DIR,
    "models"
)

OUTPUT_ROOT = os.path.join(
    SCRIPT_DIR,
    "simbind_predictions"
)

TRAINING_DATA_ZIP = os.path.join(
    PROJECT_ROOT,
    "training_data.zip"
)

os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)


# =========================================================
# CV constants
#
# Must match main_v7.py exactly.
# =========================================================

NUM_PROTEINS = 400
NUM_DNA = 30000

N_FOLDS = 10
RANDOM_SEED = 42


# =========================================================
# SimBind hyperparameters from the paper
# =========================================================

TOP_K = 14
SOFTMAX_SCALE = 8.0


# =========================================================
# Protein IDs
# =========================================================

def protein_id(index):
    return f"protein_{int(index):04d}"


# =========================================================
# Load original binding matrix
# Shape:
#     [30000 DNA, 400 proteins]
# =========================================================

def load_binding_matrix(
    zip_path
):

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

    if matrix.shape != (
        NUM_DNA,
        NUM_PROTEINS
    ):

        raise ValueError(
            f"Unexpected binding matrix shape "
            f"{matrix.shape}."
        )

    return matrix


# =========================================================
# Build protein folds exactly as main_v7.py
# =========================================================

def build_protein_folds():

    protein_indices = np.arange(
        NUM_PROTEINS
    )

    protein_kfold = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    return list(
        protein_kfold.split(
            protein_indices
        )
    )


# =========================================================
# Read pairwise similarities
#
# Output:
#
# {
#   query_protein:
#       [
#           (train_protein, similarity),
#           ...
#       ]
# }
#
# sorted descending by similarity
# =========================================================

def load_similarity_scores(
    csv_file
):

    pairwise_dict = {}


    with open(
        csv_file,
        "r",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            query_protein = (
                row[
                    "query_protein"
                ]
            )

            train_protein = (
                row[
                    "train_protein"
                ]
            )

            similarity = float(
                row[
                    "similarity_score"
                ]
            )


            pairwise_dict.setdefault(
                query_protein,
                []
            )

            pairwise_dict[
                query_protein
            ].append(
                (
                    train_protein,
                    similarity
                )
            )


    for query_protein in pairwise_dict:

        pairwise_dict[
            query_protein
        ].sort(
            key=lambda x:
                x[1],
            reverse=True
        )


    return pairwise_dict


# =========================================================
# SimBind aggregation
#
# For every validation/query protein:
#
# 1. take top 14 most similar TRAIN proteins
# 2. weights = softmax(similarity * 8)
# 3. weighted sum of MultiDBP predictions
#
# MultiDBP predictions:
#     [3000 DNA, 360 train proteins]
#
# Result:
#     [40 validation proteins, 3000 DNA]
# =========================================================

def aggregate_predictions(
    similarity_dict,
    newseq_predictions,
    train_protein_order
):

    # -----------------------------------------------------
    # Fast mapping:
    #
    # protein name -> column in MultiDBP output
    # -----------------------------------------------------

    protein_to_column = {
        protein_name:
            index

        for index,
        protein_name

        in enumerate(
            train_protein_order
        )
    }


    result = {}


    for query_protein in sorted(
        similarity_dict.keys()
    ):

        similarity_scores = (
            similarity_dict[
                query_protein
            ]
        )


        selected_names = []
        selected_scores = []
        selected_predictions = []


        for (
            train_protein,
            similarity
        ) in similarity_scores:

            if (
                train_protein
                not in protein_to_column
            ):
                continue


            selected_names.append(
                train_protein
            )

            selected_scores.append(
                similarity
            )


            column_index = (
                protein_to_column[
                    train_protein
                ]
            )


            selected_predictions.append(
                newseq_predictions[
                    :,
                    column_index
                ]
            )


            if (
                len(
                    selected_scores
                )
                >= TOP_K
            ):
                break


        if len(
            selected_scores
        ) != TOP_K:

            raise RuntimeError(
                f"{query_protein} has only "
                f"{len(selected_scores)} usable "
                f"training proteins; "
                f"expected {TOP_K}."
            )


        # -------------------------------------------------
        # [14, 3000]
        # -------------------------------------------------

        selected_predictions = (
            np.stack(
                selected_predictions,
                axis=0
            )
        )


        # -------------------------------------------------
        # Paper:
        #
        # softmax(s * lambda)
        #
        # lambda = 8
        # -------------------------------------------------

        weights = softmax(
            np.asarray(
                selected_scores,
                dtype=np.float64
            )
            * SOFTMAX_SCALE
        )


        # -------------------------------------------------
        # Weighted sum over 14 proxy proteins
        #
        # [14,1] * [14,3000]
        # ->
        # [3000]
        # -------------------------------------------------

        final_prediction = np.sum(
            weights[
                :,
                np.newaxis
            ]
            * selected_predictions,
            axis=0
        )


        result[
            query_protein
        ] = {
            "predictions":
                final_prediction,

            "neighbors":
                selected_names,

            "similarities":
                selected_scores,

            "weights":
                weights.tolist()
        }


    return result


# =========================================================
# Pearson helper
# =========================================================

def safe_pearson(
    y_true,
    y_pred
):

    if (
        np.std(y_true) < 1e-12
        or
        np.std(y_pred) < 1e-12
    ):
        return np.nan


    return stats.pearsonr(
        y_true,
        y_pred
    )[0]


# =========================================================
# Process one fold
# =========================================================

def process_fold(
    fold_number,
    binding_matrix,
    protein_folds
):

    fold_idx = (
        fold_number - 1
    )


    (
        protein_train_idx,
        protein_val_idx
    ) = protein_folds[
        fold_idx
    ]


    train_ids = [
        protein_id(
            index
        )
        for index
        in protein_train_idx
    ]


    val_ids = [
        protein_id(
            index
        )
        for index
        in protein_val_idx
    ]


    # -----------------------------------------------------
    # Input paths
    # -----------------------------------------------------

    similarity_path = os.path.join(
        SIMILARITY_ROOT,
        f"fold_{fold_number}",
        (
            f"fold_{fold_number}_"
            f"pairwise_AA_SID.csv"
        )
    )


    newseq_path = os.path.join(
        NEWSEQ_ROOT,
        f"fold_{fold_number}",
        (
            f"fold_{fold_number}_"
            f"newseq_predictions.npy"
        )
    )


    dna_indices_path = os.path.join(
        NEWSEQ_ROOT,
        f"fold_{fold_number}",
        (
            f"fold_{fold_number}_"
            f"dna_val_indices.npy"
        )
    )


    protein_order_path = os.path.join(
        NEWSEQ_ROOT,
        f"fold_{fold_number}",
        (
            f"fold_{fold_number}_"
            f"protein_order.pkl"
        )
    )


    for path in [
        similarity_path,
        newseq_path,
        dna_indices_path,
        protein_order_path
    ]:

        if not os.path.isfile(
            path
        ):

            raise FileNotFoundError(
                f"Missing required file:\n"
                f"{path}"
            )


    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    similarity_dict = (
        load_similarity_scores(
            similarity_path
        )
    )


    newseq_predictions = np.load(
        newseq_path
    )


    dna_val_idx = np.load(
        dna_indices_path
    )


    with open(
        protein_order_path,
        "rb"
    ) as f:

        train_protein_order = (
            pickle.load(
                f
            )
        )


    # -----------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------

    if newseq_predictions.shape != (
        3000,
        360
    ):

        raise RuntimeError(
            "Unexpected NewSeq prediction shape: "
            f"{newseq_predictions.shape}"
        )


    if len(
        dna_val_idx
    ) != 3000:

        raise RuntimeError(
            "Expected 3000 validation DNA indices."
        )


    if len(
        train_protein_order
    ) != 360:

        raise RuntimeError(
            "Expected 360 training proteins "
            "in MultiDBP output order."
        )


    if set(
        train_protein_order
    ) != set(
        train_ids
    ):

        raise RuntimeError(
            "Training protein set mismatch "
            "between fold split and MultiDBP output."
        )


    if set(
        similarity_dict.keys()
    ) != set(
        val_ids
    ):

        raise RuntimeError(
            "Validation/query protein set mismatch "
            "in similarity file."
        )


    # =====================================================
    # Aggregate
    # =====================================================

    aggregated = (
        aggregate_predictions(
            similarity_dict=similarity_dict,
            newseq_predictions=newseq_predictions,
            train_protein_order=train_protein_order
        )
    )


    # -----------------------------------------------------
    # Build final matrix
    #
    # IMPORTANT:
    # Preserve exact validation protein order from KFold.
    #
    # Shape:
    #     [40 proteins, 3000 DNA]
    # -----------------------------------------------------

    prediction_matrix = np.stack(
        [
            aggregated[
                protein_name
            ][
                "predictions"
            ]

            for protein_name
            in val_ids
        ],
        axis=0
    )


    if prediction_matrix.shape != (
        40,
        3000
    ):

        raise RuntimeError(
            f"Unexpected final SimBind shape: "
            f"{prediction_matrix.shape}"
        )


    # =====================================================
    # Evaluate against TRUE validation block
    #
    # binding matrix shape:
    #     [DNA, protein]
    #
    # We want:
    #     [40 protein, 3000 DNA]
    # =====================================================

    true_matrix = (
        binding_matrix[
            np.ix_(
                dna_val_idx,
                protein_val_idx
            )
        ]
        .T
    )


    if true_matrix.shape != (
        40,
        3000
    ):

        raise RuntimeError(
            "Unexpected true validation shape."
        )


    protein_pearsons = []


    for protein_position in range(
        40
    ):

        pearson = safe_pearson(
            true_matrix[
                protein_position
            ],
            prediction_matrix[
                protein_position
            ]
        )

        protein_pearsons.append(
            pearson
        )


    mean_pearson = float(
        np.nanmean(
            protein_pearsons
        )
    )


    # =====================================================
    # Save outputs
    # =====================================================

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
            f"simbind_predictions.npy"
        )
    )


    truth_path = os.path.join(
        fold_output_dir,
        (
            f"fold_{fold_number}_"
            f"targets.npy"
        )
    )


    pearson_path = os.path.join(
        fold_output_dir,
        (
            f"fold_{fold_number}_"
            f"pearsons.csv"
        )
    )


    neighbors_path = os.path.join(
        fold_output_dir,
        (
            f"fold_{fold_number}_"
            f"neighbors.csv"
        )
    )


    np.save(
        prediction_path,
        prediction_matrix
    )


    np.save(
        truth_path,
        true_matrix
    )


    # -----------------------------------------------------
    # Per-protein Pearson
    # -----------------------------------------------------

    pearson_df = pd.DataFrame(
        {
            "protein_id":
                val_ids,

            "protein_index":
                protein_val_idx,

            "pearson":
                protein_pearsons
        }
    )


    pearson_df.to_csv(
        pearson_path,
        index=False
    )


    # -----------------------------------------------------
    # Save top-14 neighbors, similarities and weights
    # for complete auditability.
    # -----------------------------------------------------

    neighbor_rows = []


    for query_protein in val_ids:

        info = aggregated[
            query_protein
        ]


        for rank, (
            neighbor,
            similarity,
            weight
        ) in enumerate(
            zip(
                info[
                    "neighbors"
                ],
                info[
                    "similarities"
                ],
                info[
                    "weights"
                ]
            ),
            start=1
        ):

            neighbor_rows.append(
                {
                    "query_protein":
                        query_protein,

                    "rank":
                        rank,

                    "train_protein":
                        neighbor,

                    "similarity":
                        similarity,

                    "weight":
                        weight
                }
            )


    pd.DataFrame(
        neighbor_rows
    ).to_csv(
        neighbors_path,
        index=False
    )


    # =====================================================
    # Summary
    # =====================================================

    print()
    print("=" * 70)

    print(
        f"SIMBIND AGGREGATION "
        f"FOLD {fold_number}/{N_FOLDS}"
    )

    print("=" * 70)

    print(
        "NewSeq matrix:",
        newseq_predictions.shape
    )

    print(
        "Final SimBind matrix:",
        prediction_matrix.shape
    )

    print(
        "Target matrix:",
        true_matrix.shape
    )

    print(
        "Top similar proteins:",
        TOP_K
    )

    print(
        "Softmax scale:",
        SOFTMAX_SCALE
    )

    print()
    print(
        "Fold SimBind mean Pearson:",
        f"{mean_pearson:.4f}"
    )

    print()

    print(
        "Saved predictions:",
        prediction_path
    )

    print(
        "Saved targets:",
        truth_path
    )

    print(
        "Saved Pearson scores:",
        pearson_path
    )

    print(
        "Saved top-14 neighbors:",
        neighbors_path
    )

    print("=" * 70)


    return mean_pearson


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate MultiDBP predictions "
            "with protein similarity to produce "
            "leakage-safe SimBind predictions."
        )
    )


    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help=(
            "Fold to run (1-10). "
            "If omitted, all folds are processed."
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
            "--fold must be between 1 and 10."
        )


    binding_matrix = (
        load_binding_matrix(
            TRAINING_DATA_ZIP
        )
    )


    protein_folds = (
        build_protein_folds()
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


    fold_scores = []


    for fold_number in fold_numbers:

        score = process_fold(
            fold_number=fold_number,
            binding_matrix=binding_matrix,
            protein_folds=protein_folds
        )

        fold_scores.append(
            (
                fold_number,
                score
            )
        )


    # -----------------------------------------------------
    # Summary if multiple folds are available
    # -----------------------------------------------------

    if len(
        fold_scores
    ) > 0:

        values = [
            score
            for _,
            score
            in fold_scores
        ]

        print()
        print("=" * 70)
        print("SIMBIND SUMMARY")
        print("=" * 70)


        for (
            fold_number,
            score
        ) in fold_scores:

            print(
                f"Fold {fold_number}: "
                f"Pearson={score:.4f}"
            )


        print()
        print(
            "Mean Pearson:",
            f"{np.mean(values):.4f}"
        )

        print(
            "Std Pearson:",
            f"{np.std(values):.4f}"
        )

        print("=" * 70)


if __name__ == "__main__":
    main()