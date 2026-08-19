import os
import sys
import shutil
import argparse
import subprocess

from Bio import SeqIO


# =========================================================
# Paths
# =========================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SCRIPT_DIR
)

TRAINING_PROTEINS_FILE = os.path.join(
    PROJECT_ROOT,
    "training_DBPs.txt"
)

# HMM models are kept untouched in the original NPBIP repo
HMM_DIR = os.path.join(
    PROJECT_ROOT,
    "NPBIP_original",
    "models",
    "SimBind",
    "HMM"
)

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "prepared_data"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# Constants
# =========================================================

NUM_PROTEINS = 400


# =========================================================
# Protein IDs
#
# Must match train_multi_task_newseq.py exactly.
# =========================================================

def protein_id(index):
    return f"protein_{int(index):04d}"


# =========================================================
# Read training_DBPs.txt
#
# The file contains one amino-acid sequence per line
# and has no FASTA headers.
# =========================================================

def load_plain_protein_sequences(
    file_path
):

    sequences = []

    with open(
        file_path,
        "r"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                sequences.append(
                    line
                )

    print(
        "Number of protein sequences:",
        len(sequences)
    )

    if len(sequences) != NUM_PROTEINS:

        raise ValueError(
            f"Expected {NUM_PROTEINS} proteins, "
            f"found {len(sequences)}."
        )

    return sequences


# =========================================================
# Convert our plain protein file to FASTA
#
# protein index i corresponds to:
#
#     protein_0000
#     protein_0001
#     ...
# =========================================================

def write_training_fasta(
    sequences,
    output_path
):

    with open(
        output_path,
        "w"
    ) as fout:

        for index, sequence in enumerate(
            sequences
        ):

            pid = protein_id(
                index
            )

            fout.write(
                f">{pid}\n"
            )

            for start in range(
                0,
                len(sequence),
                80
            ):

                fout.write(
                    sequence[
                        start:start + 80
                    ]
                    + "\n"
                )

    print(
        "Saved protein FASTA:",
        output_path
    )


# =========================================================
# Run HMMER
# =========================================================

def run_hmmscan(
    fasta_file,
    hmm_db,
    output_tbl
):

    command = [
        "hmmscan",

        "-E",
        "0.01",

        "--domE",
        "0.01",

        "--domtblout",
        output_tbl,

        hmm_db,
        fasta_file
    ]

    print()
    print(
        "Running hmmscan..."
    )

    print(
        " ".join(command)
    )

    subprocess.run(
        command,
        check=True
    )

    print()
    print(
        "hmmscan finished."
    )

    print(
        "Results:",
        output_tbl
    )


# =========================================================
# Load FASTA sequences
# =========================================================

def load_protein_sequences(
    fasta_file
):

    sequences = {}

    for record in SeqIO.parse(
        fasta_file,
        "fasta"
    ):

        sequences[
            record.id
        ] = str(
            record.seq
        )

    return sequences


# =========================================================
# Parse HMMER domtblout
#
# This preserves the original NPBIP logic:
#
# - take aligned domain coordinates
# - extend by 15 amino acids on both sides
#
# If no recognized DNA-binding domain is found,
# preserve the original fallback behavior and use
# the complete protein sequence.
# =========================================================

def parse_domtblout(
    domtblout_file,
    fasta_file,
    add_15=True
):

    protein_seqs = (
        load_protein_sequences(
            fasta_file
        )
    )

    results = {}


    with open(
        domtblout_file,
        "r"
    ) as f:

        for line in f:

            if line.startswith(
                "#"
            ):
                continue

            parts = (
                line.strip().split()
            )

            if len(parts) < 21:
                continue


            domain_name = (
                parts[0]
            )

            query_name = (
                parts[3]
            )


            try:

                cEvalue = float(
                    parts[11]
                )

                ali_from = int(
                    parts[19]
                )

                ali_to = int(
                    parts[20]
                )

            except (
                ValueError,
                IndexError
            ):

                continue


            if (
                query_name
                not in protein_seqs
            ):
                continue


            full_seq = (
                protein_seqs[
                    query_name
                ]
            )


            # ---------------------------------------------
            # Original SimBind:
            # domain + 15 amino acids on each side
            # ---------------------------------------------

            if add_15:

                start = max(
                    0,
                    ali_from - 1 - 15
                )

                end = min(
                    len(full_seq),
                    ali_to + 15
                )

            else:

                start = max(
                    0,
                    ali_from - 1
                )

                end = min(
                    len(full_seq),
                    ali_to
                )


            domain_seq = (
                full_seq[
                    start:end
                ]
            )


            hit = {
                "domain_name":
                    domain_name,

                "ali_from":
                    ali_from,

                "ali_to":
                    ali_to,

                "cEvalue":
                    cEvalue,

                "domain_seq":
                    domain_seq
            }


            if query_name not in results:

                results[
                    query_name
                ] = []


            results[
                query_name
            ].append(
                hit
            )


    # =====================================================
    # Preserve all proteins.
    #
    # Original NPBIP fallback:
    # if HMMER finds no recognized domain, use the complete
    # protein as one region.
    # =====================================================

    proteins_without_domain = []


    for pid, sequence in (
        protein_seqs.items()
    ):

        if pid not in results:

            proteins_without_domain.append(
                pid
            )

            results[
                pid
            ] = [
                {
                    "domain_name":
                        "None",

                    "ali_from":
                        1,

                    "ali_to":
                        len(sequence),

                    "cEvalue":
                        1.0,

                    "domain_seq":
                        sequence
                }
            ]


    print()
    print(
        "Proteins with recognized DNA-binding domain:",
        (
            len(protein_seqs)
            - len(proteins_without_domain)
        )
    )

    print(
        "Proteins using full-sequence fallback:",
        len(
            proteins_without_domain
        )
    )


    return results


# =========================================================
# Write domain FASTA
#
# Header format is compatible with pair_wise.py:
#
# protein_0000_<domain>_domain1_fromX_toY_...
#
# IMPORTANT:
# protein IDs contain an underscore, so pair_wise.py will
# later be adapted to parse the ID correctly.
# =========================================================

def write_domain_fasta(
    domains_dict,
    output_file
):

    with open(
        output_file,
        "w"
    ) as fout:

        for protein_name in sorted(
            domains_dict.keys()
        ):

            domain_list = (
                domains_dict[
                    protein_name
                ]
            )

            # Ensure genomic/protein order of domains
            domain_list = sorted(
                domain_list,
                key=lambda d:
                    d["ali_from"]
            )

            for idx, domain in enumerate(
                domain_list,
                start=1
            ):

                header = (
                    f">{protein_name}"
                    f"__{domain['domain_name']}"
                    f"__domain{idx}"
                    f"__from{domain['ali_from']}"
                    f"__to{domain['ali_to']}"
                    f"__cEvalue{domain['cEvalue']:.2e}"
                )

                fout.write(
                    header + "\n"
                )

                sequence = (
                    domain[
                        "domain_seq"
                    ]
                )

                for start in range(
                    0,
                    len(sequence),
                    80
                ):

                    fout.write(
                        sequence[
                            start:start + 80
                        ]
                        + "\n"
                    )


    print(
        "Saved domain FASTA:",
        output_file
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare DNA-binding domains "
            "for all 400 training proteins "
            "using the original SimBind "
            "HMMER/Pfam procedure."
        )
    )

    parser.parse_args()


    # -----------------------------------------------------
    # Validate required files
    # -----------------------------------------------------

    if not os.path.isfile(
        TRAINING_PROTEINS_FILE
    ):

        raise FileNotFoundError(
            f"Missing file: "
            f"{TRAINING_PROTEINS_FILE}"
        )


    dna_hmm = os.path.join(
        HMM_DIR,
        "combined_dna_binding.hmm"
    )


    if not os.path.isfile(
        dna_hmm
    ):

        raise FileNotFoundError(
            "Could not find SimBind DNA HMM database at:\n"
            f"{dna_hmm}"
        )


    if shutil.which(
        "hmmscan"
    ) is None:

        raise RuntimeError(
            "'hmmscan' is not installed "
            "or is not available in PATH."
        )


    # -----------------------------------------------------
    # Output files
    # -----------------------------------------------------

    all_proteins_fasta = os.path.join(
        OUTPUT_DIR,
        "training_DBPs_with_ids.fasta"
    )

    hmmscan_output = os.path.join(
        OUTPUT_DIR,
        "training_DBPs_hmmscan_DNA.tbl"
    )

    domains_output = os.path.join(
        OUTPUT_DIR,
        "training_DBPs_domains_plus_15.fasta"
    )


    # -----------------------------------------------------
    # 1. Convert training_DBPs.txt -> FASTA
    # -----------------------------------------------------

    sequences = (
        load_plain_protein_sequences(
            TRAINING_PROTEINS_FILE
        )
    )

    write_training_fasta(
        sequences,
        all_proteins_fasta
    )


    # -----------------------------------------------------
    # 2. HMMER / Pfam
    # -----------------------------------------------------

    run_hmmscan(
        fasta_file=all_proteins_fasta,
        hmm_db=dna_hmm,
        output_tbl=hmmscan_output
    )


    # -----------------------------------------------------
    # 3. Extract domains + 15-aa flanks
    # -----------------------------------------------------

    domain_hits = (
        parse_domtblout(
            domtblout_file=hmmscan_output,
            fasta_file=all_proteins_fasta,
            add_15=True
        )
    )


    # -----------------------------------------------------
    # 4. Save
    # -----------------------------------------------------

    write_domain_fasta(
        domain_hits,
        domains_output
    )


    # -----------------------------------------------------
    # Final checks
    # -----------------------------------------------------

    parsed_domains = {}

    for record in SeqIO.parse(
        domains_output,
        "fasta"
    ):

        # Header starts with:
        # protein_XXXX__
        pid = record.id.split(
            "__",
            1
        )[0]

        parsed_domains.setdefault(
            pid,
            0
        )

        parsed_domains[
            pid
        ] += 1


    if len(
        parsed_domains
    ) != NUM_PROTEINS:

        raise RuntimeError(
            f"Domain FASTA contains "
            f"{len(parsed_domains)} proteins; "
            f"expected {NUM_PROTEINS}."
        )


    print()
    print("=" * 70)

    print(
        "Protein-domain preparation complete."
    )

    print(
        "Proteins represented:",
        len(parsed_domains)
    )

    print(
        "Domain FASTA:",
        domains_output
    )

    print("=" * 70)


if __name__ == "__main__":
    main()