import os
import sys
import argparse
import subprocess


# =========================================================
# Paths
# =========================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PYTHON = sys.executable


# =========================================================
# Scripts
# =========================================================

PROTEIN_DOMAIN_SCRIPT = os.path.join(
    SCRIPT_DIR,
    "protein_domain.py"
)

TRAIN_MULTIDBP_SCRIPT = os.path.join(
    SCRIPT_DIR,
    "train_multi_task_newseq.py"
)

PAIRWISE_SCRIPT = os.path.join(
    SCRIPT_DIR,
    "pair_wise.py"
)

NEWSEQ_SCRIPT = os.path.join(
    SCRIPT_DIR,
    "NewSeq_prediction.py"
)

AGGREGATION_SCRIPT = os.path.join(
    SCRIPT_DIR,
    "NewProNewSeq_agg.py"
)


# =========================================================
# Expected outputs
# =========================================================

PREPARED_DOMAINS = os.path.join(
    SCRIPT_DIR,
    "prepared_data",
    "training_DBPs_domains_plus_15.fasta"
)


def fold_model_path(
    fold_number
):
    return os.path.join(
        SCRIPT_DIR,
        "models",
        f"fold_{fold_number}",
        f"MultiDBP_fold_{fold_number}.pt"
    )


def fold_similarity_path(
    fold_number
):
    return os.path.join(
        SCRIPT_DIR,
        "similarities",
        f"fold_{fold_number}",
        (
            f"fold_{fold_number}_"
            f"pairwise_AA_SID.csv"
        )
    )


def fold_newseq_path(
    fold_number
):
    return os.path.join(
        SCRIPT_DIR,
        "newseq_predictions",
        f"fold_{fold_number}",
        (
            f"fold_{fold_number}_"
            f"newseq_predictions.npy"
        )
    )


def fold_simbind_path(
    fold_number
):
    return os.path.join(
        SCRIPT_DIR,
        "simbind_predictions",
        f"fold_{fold_number}",
        (
            f"fold_{fold_number}_"
            f"simbind_predictions.npy"
        )
    )


# =========================================================
# Run command
# =========================================================

def run_command(
    command,
    description
):

    print()
    print("=" * 70)
    print(description)
    print("=" * 70)

    print(
        "Command:"
    )

    print(
        " ".join(command)
    )

    print()

    subprocess.run(
        command,
        check=True
    )


# =========================================================
# Step 1
# Protein domains
#
# This is independent of CV and therefore needs to be
# calculated only once for all 400 proteins.
# =========================================================

def prepare_domains(
    force=False
):

    if (
        os.path.isfile(
            PREPARED_DOMAINS
        )
        and not force
    ):

        print()
        print(
            "[SKIP] Protein domains already exist:"
        )

        print(
            PREPARED_DOMAINS
        )

        return


    command = [
        PYTHON,
        PROTEIN_DOMAIN_SCRIPT
    ]


    run_command(
        command,
        (
            "STEP 1/5 - "
            "HMMER DNA-binding domains"
        )
    )


    if not os.path.isfile(
        PREPARED_DOMAINS
    ):

        raise RuntimeError(
            "protein_domain.py finished "
            "but expected domain FASTA "
            "was not created:\n"
            f"{PREPARED_DOMAINS}"
        )


# =========================================================
# Step 2
# Train fold-specific MultiDBP
# =========================================================

def train_multidbp(
    fold_number,
    force=False
):

    output_path = (
        fold_model_path(
            fold_number
        )
    )


    if (
        os.path.isfile(
            output_path
        )
        and not force
    ):

        print()
        print(
            f"[SKIP] Fold {fold_number} "
            f"MultiDBP already exists:"
        )

        print(
            output_path
        )

        return


    command = [
        PYTHON,
        TRAIN_MULTIDBP_SCRIPT,
        "--fold",
        str(
            fold_number
        )
    ]


    run_command(
        command,
        (
            f"STEP 2/5 - "
            f"Train MultiDBP "
            f"for Fold {fold_number}"
        )
    )


    if not os.path.isfile(
        output_path
    ):

        raise RuntimeError(
            "MultiDBP training finished "
            "but model was not created:\n"
            f"{output_path}"
        )


# =========================================================
# Step 3
# Protein similarity
#
# Validation proteins are compared ONLY with the 360
# training proteins from the same fold.
# =========================================================

def calculate_similarity(
    fold_number,
    workers=None,
    force=False
):

    output_path = (
        fold_similarity_path(
            fold_number
        )
    )


    if (
        os.path.isfile(
            output_path
        )
        and not force
    ):

        print()
        print(
            f"[SKIP] Fold {fold_number} "
            f"similarities already exist:"
        )

        print(
            output_path
        )

        return


    command = [
        PYTHON,
        PAIRWISE_SCRIPT,
        "--fold",
        str(
            fold_number
        )
    ]


    if workers is not None:

        command.extend(
            [
                "--workers",
                str(
                    workers
                )
            ]
        )


    run_command(
        command,
        (
            f"STEP 3/5 - "
            f"Protein similarity "
            f"for Fold {fold_number}"
        )
    )


    if not os.path.isfile(
        output_path
    ):

        raise RuntimeError(
            "pair_wise.py finished "
            "but similarity file "
            "was not created:\n"
            f"{output_path}"
        )


# =========================================================
# Step 4
# MultiDBP predictions on held-out DNA
# =========================================================

def predict_newseq(
    fold_number,
    force=False
):

    output_path = (
        fold_newseq_path(
            fold_number
        )
    )


    if (
        os.path.isfile(
            output_path
        )
        and not force
    ):

        print()
        print(
            f"[SKIP] Fold {fold_number} "
            f"NewSeq predictions already exist:"
        )

        print(
            output_path
        )

        return


    command = [
        PYTHON,
        NEWSEQ_SCRIPT,
        "--fold",
        str(
            fold_number
        )
    ]


    run_command(
        command,
        (
            f"STEP 4/5 - "
            f"MultiDBP predictions "
            f"for Fold {fold_number}"
        )
    )


    if not os.path.isfile(
        output_path
    ):

        raise RuntimeError(
            "NewSeq_prediction.py finished "
            "but prediction file "
            "was not created:\n"
            f"{output_path}"
        )


# =========================================================
# Step 5
# Final SimBind aggregation
#
# top-14
# lambda = 8
# =========================================================

def aggregate_simbind(
    fold_number,
    force=False
):

    output_path = (
        fold_simbind_path(
            fold_number
        )
    )


    if (
        os.path.isfile(
            output_path
        )
        and not force
    ):

        print()
        print(
            f"[SKIP] Fold {fold_number} "
            f"SimBind predictions already exist:"
        )

        print(
            output_path
        )

        return


    command = [
        PYTHON,
        AGGREGATION_SCRIPT,
        "--fold",
        str(
            fold_number
        )
    ]


    run_command(
        command,
        (
            f"STEP 5/5 - "
            f"Final SimBind aggregation "
            f"for Fold {fold_number}"
        )
    )


    if not os.path.isfile(
        output_path
    ):

        raise RuntimeError(
            "Aggregation finished "
            "but final SimBind predictions "
            "were not created:\n"
            f"{output_path}"
        )


# =========================================================
# Run complete fold
# =========================================================

def run_fold(
    fold_number,
    workers=None,
    force=False
):

    print()
    print()
    print("#" * 70)

    print(
        f"SIMBIND FOLD "
        f"{fold_number}/10"
    )

    print("#" * 70)


    # -----------------------------------------------------
    # Fold-specific MultiDBP
    # -----------------------------------------------------

    train_multidbp(
        fold_number=fold_number,
        force=force
    )


    # -----------------------------------------------------
    # Validation-vs-training protein similarity
    # -----------------------------------------------------

    calculate_similarity(
        fold_number=fold_number,
        workers=workers,
        force=force
    )


    # -----------------------------------------------------
    # Predictions for validation DNA
    # -----------------------------------------------------

    predict_newseq(
        fold_number=fold_number,
        force=force
    )


    # -----------------------------------------------------
    # Final top-14 weighted SimBind predictions
    # -----------------------------------------------------

    aggregate_simbind(
        fold_number=fold_number,
        force=force
    )


    print()
    print("#" * 70)

    print(
        f"FOLD {fold_number} COMPLETE"
    )

    print(
        "Final predictions:"
    )

    print(
        fold_simbind_path(
            fold_number
        )
    )

    print("#" * 70)


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the adapted leakage-safe "
            "SimBind pipeline on the "
            "DNA-protein binding dataset."
        )
    )


    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help=(
            "Fold to run (1-10). "
            "If omitted, run all 10 folds "
            "sequentially."
        )
    )


    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "CPU workers for protein "
            "pairwise alignment."
        )
    )


    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute outputs even if "
            "they already exist. "
            "Use carefully because this "
            "will retrain MultiDBP."
        )
    )


    args = parser.parse_args()


    # -----------------------------------------------------
    # Validate fold
    # -----------------------------------------------------

    if (
        args.fold is not None
        and not (
            1
            <= args.fold
            <= 10
        )
    ):

        raise ValueError(
            "--fold must be "
            "between 1 and 10."
        )


    print()
    print("=" * 70)

    print(
        "CUSTOM SIMBIND PIPELINE"
    )

    print("=" * 70)

    print(
        "Python:",
        PYTHON
    )

    print(
        "Script directory:",
        SCRIPT_DIR
    )

    print(
        "Fold:",
        (
            args.fold
            if args.fold is not None
            else "ALL"
        )
    )

    print(
        "Force recompute:",
        args.force
    )

    print("=" * 70)


    # =====================================================
    # Domains are shared across all folds.
    # =====================================================

    prepare_domains(
        force=args.force
    )


    # =====================================================
    # Determine folds
    # =====================================================

    if args.fold is None:

        fold_numbers = range(
            1,
            11
        )

    else:

        fold_numbers = [
            args.fold
        ]


    # =====================================================
    # Run
    # =====================================================

    for fold_number in fold_numbers:

        run_fold(
            fold_number=fold_number,
            workers=args.workers,
            force=args.force
        )


    print()
    print("=" * 70)

    print(
        "SIMBIND PIPELINE COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()