import os
import re
import csv
import argparse
import multiprocessing as mp

import numpy as np

from Bio import SeqIO, pairwise2
from Bio.Align import substitution_matrices
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

PREPARED_DATA_DIR = os.path.join(
    SCRIPT_DIR,
    "prepared_data"
)

ALL_DOMAINS_FASTA = os.path.join(
    PREPARED_DATA_DIR,
    "training_DBPs_domains_plus_15.fasta"
)

OUTPUT_ROOT = os.path.join(
    SCRIPT_DIR,
    "similarities"
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

NUM_PROTEINS = 400
N_FOLDS = 10
RANDOM_SEED = 42


# =========================================================
# BLOSUM62
# =========================================================

BLOSUM62 = substitution_matrices.load(
    "BLOSUM62"
)


# =========================================================
# Protein IDs
#
# Must match train_multi_task_newseq.py
# and protein_domain.py exactly.
# =========================================================

def protein_id(index):
    return f"protein_{int(index):04d}"


# =========================================================
# AA sequence identity
#
# Original SimBind procedure:
#
# Needleman-Wunsch global alignment
# BLOSUM62
# gap open = -11
# gap extend = -1
#
# similarity =
# exact identical aligned residues / alignment length
# =========================================================

def compute_AA_SID(
    seq1,
    seq2
):

    alignments = (
        pairwise2.align.globalds(
            seq1,
            seq2,
            BLOSUM62,
            -11,
            -1,
            one_alignment_only=True
        )
    )

    if len(alignments) == 0:
        return 0.0

    aln1 = alignments[0].seqA
    aln2 = alignments[0].seqB

    if len(aln1) == 0:
        return 0.0

    identical = sum(
        1
        for aa1, aa2
        in zip(
            aln1,
            aln2
        )
        if aa1 == aa2
    )

    sid = (
        identical
        / len(aln1)
    )

    return sid


# =========================================================
# SimBind domain-level protein similarity
#
# Same number of domains:
#     align by order and average
#
# Different number of domains:
#     slide the shorter domain list over the longer one
#     while preserving domain order,
#     and take the maximum mean SID.
# =========================================================

def calculate_pairwise_AA_SID(
    domains1,
    domains2
):

    n1 = len(domains1)
    n2 = len(domains2)

    if (
        n1 == 0
        or n2 == 0
    ):
        raise ValueError(
            "Protein has no domains."
        )

    if n1 == n2:

        sids = [
            compute_AA_SID(
                domains1[i],
                domains2[i]
            )
            for i in range(
                n1
            )
        ]

        return float(
            np.mean(
                sids
            )
        )


    # -----------------------------------------------------
    # Different numbers of domains
    # -----------------------------------------------------

    if n1 < n2:

        shorter = domains1
        longer = domains2

    else:

        shorter = domains2
        longer = domains1


    max_mean_sid = 0.0

    max_offset = (
        len(longer)
        - len(shorter)
    )


    for offset in range(
        max_offset + 1
    ):

        sids = []

        for i in range(
            len(shorter)
        ):

            sid = compute_AA_SID(
                shorter[i],
                longer[
                    i + offset
                ]
            )

            sids.append(
                sid
            )


        mean_sid = float(
            np.mean(
                sids
            )
        )


        if mean_sid > max_mean_sid:

            max_mean_sid = (
                mean_sid
            )


    return max_mean_sid


# =========================================================
# Read domain FASTA created by protein_domain.py
#
# Example header:
#
# protein_0037__Homeobox__domain1__from12__to80__...
#
# We intentionally use "__" as delimiter so protein_0037
# stays intact.
# =========================================================

def parse_domain_fasta_file(
    fasta_file
):

    protein_domains = {}


    for record in SeqIO.parse(
        fasta_file,
        "fasta"
    ):

        header = record.id


        # -------------------------------------------------
        # Extract stable protein ID
        # -------------------------------------------------

        protein_name = (
            header.split(
                "__",
                1
            )[0]
        )


        # -------------------------------------------------
        # Extract original domain start coordinate
        # -------------------------------------------------

        match = re.search(
            r"__from(\d+)__to",
            header
        )

        if match:

            from_coord = int(
                match.group(1)
            )

        else:

            from_coord = 0


        domain_seq = str(
            record.seq
        )


        if protein_name not in protein_domains:

            protein_domains[
                protein_name
            ] = []


        protein_domains[
            protein_name
        ].append(
            (
                from_coord,
                domain_seq
            )
        )


    # -----------------------------------------------------
    # Keep domains in protein order
    # -----------------------------------------------------

    for protein_name in protein_domains:

        protein_domains[
            protein_name
        ].sort(
            key=lambda x:
                x[0]
        )

        protein_domains[
            protein_name
        ] = [
            sequence
            for _,
            sequence
            in protein_domains[
                protein_name
            ]
        ]


    return protein_domains


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

    folds = list(
        protein_kfold.split(
            protein_indices
        )
    )

    return folds


# =========================================================
# Worker
# =========================================================

def compute_sid_worker(
    args
):

    (
        query_id,
        query_domains,
        train_id,
        train_domains
    ) = args


    similarity = (
        calculate_pairwise_AA_SID(
            query_domains,
            train_domains
        )
    )


    return {
        "query_protein":
            query_id,

        "train_protein":
            train_id,

        "similarity_score":
            similarity
    }


# =========================================================
# Compute one fold
# =========================================================

def compute_fold(
    fold_number,
    all_domains,
    protein_folds,
    num_workers
):

    fold_idx = (
        fold_number - 1
    )

    (
        train_indices,
        val_indices
    ) = protein_folds[
        fold_idx
    ]


    train_ids = [
        protein_id(index)
        for index
        in train_indices
    ]

    val_ids = [
        protein_id(index)
        for index
        in val_indices
    ]


    # -----------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------

    if len(train_ids) != 360:

        raise RuntimeError(
            f"Expected 360 training proteins, "
            f"got {len(train_ids)}."
        )


    if len(val_ids) != 40:

        raise RuntimeError(
            f"Expected 40 validation proteins, "
            f"got {len(val_ids)}."
        )


    missing_train = [
        pid
        for pid in train_ids
        if pid not in all_domains
    ]

    missing_val = [
        pid
        for pid in val_ids
        if pid not in all_domains
    ]


    if missing_train:

        raise RuntimeError(
            "Missing training proteins "
            "from domain FASTA:\n"
            + "\n".join(
                missing_train
            )
        )


    if missing_val:

        raise RuntimeError(
            "Missing validation proteins "
            "from domain FASTA:\n"
            + "\n".join(
                missing_val
            )
        )


    print()
    print("=" * 70)

    print(
        f"SIMBIND PROTEIN SIMILARITY "
        f"FOLD {fold_number}/{N_FOLDS}"
    )

    print("=" * 70)

    print(
        "Train proteins:",
        len(train_ids)
    )

    print(
        "Validation/query proteins:",
        len(val_ids)
    )


    # -----------------------------------------------------
    # Every validation protein vs every training protein
    #
    # 40 × 360 = 14,400 comparisons
    # -----------------------------------------------------

    arg_list = [
        (
            query_id,
            all_domains[
                query_id
            ],
            train_id,
            all_domains[
                train_id
            ]
        )

        for query_id
        in val_ids

        for train_id
        in train_ids
    ]


    expected_pairs = (
        40 * 360
    )


    if len(arg_list) != expected_pairs:

        raise RuntimeError(
            f"Expected {expected_pairs} "
            f"protein pairs, "
            f"got {len(arg_list)}."
        )


    print(
        "Protein comparisons:",
        len(arg_list)
    )


    # -----------------------------------------------------
    # Multiprocessing
    # -----------------------------------------------------

    if num_workers is None:

        workers = max(
            1,
            mp.cpu_count() - 1
        )

    else:

        workers = max(
            1,
            num_workers
        )


    print(
        "CPU workers:",
        workers
    )


    with mp.Pool(
        workers
    ) as pool:

        results = list(
            tqdm(
                pool.imap(
                    compute_sid_worker,
                    arg_list
                ),
                total=len(
                    arg_list
                ),
                desc=(
                    f"Fold {fold_number}"
                )
            )
        )


    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    fold_output_dir = os.path.join(
        OUTPUT_ROOT,
        f"fold_{fold_number}"
    )

    os.makedirs(
        fold_output_dir,
        exist_ok=True
    )


    output_csv = os.path.join(
        fold_output_dir,
        (
            f"fold_{fold_number}_"
            f"pairwise_AA_SID.csv"
        )
    )


    with open(
        output_csv,
        "w",
        newline=""
    ) as csvfile:

        fieldnames = [
            "query_protein",
            "train_protein",
            "similarity_score"
        ]

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    # -----------------------------------------------------
    # Leakage sanity check
    # -----------------------------------------------------

    train_set = set(
        train_ids
    )

    val_set = set(
        val_ids
    )

    overlap = (
        train_set
        & val_set
    )


    if overlap:

        raise RuntimeError(
            "LEAKAGE DETECTED: "
            "protein appears in both "
            "train and validation."
        )


    # -----------------------------------------------------
    # Verify exactly 360 similarities per query protein
    # -----------------------------------------------------

    counts = {}

    for row in results:

        query = row[
            "query_protein"
        ]

        counts.setdefault(
            query,
            0
        )

        counts[
            query
        ] += 1


    bad_counts = {
        protein:
            count

        for protein,
        count

        in counts.items()

        if count != 360
    }


    if bad_counts:

        raise RuntimeError(
            "Some validation proteins "
            "do not have exactly "
            "360 similarity scores:\n"
            f"{bad_counts}"
        )


    print()
    print(
        "Leakage check: PASSED"
    )

    print(
        "Each validation protein "
        "has exactly 360 similarities."
    )

    print(
        "Saved:",
        output_csv
    )

    print("=" * 70)


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compute leakage-safe "
            "SimBind protein similarities "
            "using the same protein folds "
            "as main_v7.py."
        )
    )


    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help=(
            "Fold to run (1-10). "
            "If omitted, run all folds."
        )
    )


    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of CPU workers. "
            "Default: cpu_count - 1."
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


    if not os.path.isfile(
        ALL_DOMAINS_FASTA
    ):

        raise FileNotFoundError(
            "Domain FASTA does not exist yet:\n"
            f"{ALL_DOMAINS_FASTA}\n\n"
            "Run protein_domain.py first."
        )


    # -----------------------------------------------------
    # Load all 400 prepared proteins
    # -----------------------------------------------------

    all_domains = (
        parse_domain_fasta_file(
            ALL_DOMAINS_FASTA
        )
    )


    print(
        "Proteins loaded from "
        "domain FASTA:",
        len(all_domains)
    )


    if len(all_domains) != NUM_PROTEINS:

        raise RuntimeError(
            f"Expected {NUM_PROTEINS} proteins, "
            f"found {len(all_domains)}."
        )


    # -----------------------------------------------------
    # Same CV split as v7
    # -----------------------------------------------------

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


    for fold_number in fold_numbers:

        compute_fold(
            fold_number=fold_number,
            all_domains=all_domains,
            protein_folds=protein_folds,
            num_workers=args.workers
        )


if __name__ == "__main__":
    main()