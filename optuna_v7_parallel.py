#!/usr/bin/env python
# coding: utf-8

"""
Parallel Optuna search for main_v7_functional.py.

The base model, preprocessing, folds, training loop, validation,
and result logic are imported from main_v7_functional.py.

This file only defines the hyperparameter search space, Optuna objective,
and parallel workers that share one persistent Optuna study.

Run:
    python optuna_v7_parallel.py
"""

import json
import multiprocessing as mp
import random

import numpy as np
import optuna
import torch

import main_v7_functional as base


# =========================================================
# OPTUNA SETTINGS
# =========================================================

RANDOM_SEED = 42

N_TRIALS = 30
N_PARALLEL_WORKERS = 15
OPTUNA_FOLDS = 2
OPTUNA_MAX_EPOCHS = 7
OPTUNA_PATIENCE = 3

GPU_ASSIGNMENTS = [
    0, 0, 0, 0, 0, 0,    # שישה על GPU 0
    1, 1,                # שניים על GPU 1
    2, 2, 2,             # שלושה על GPU 2
    3, 3, 3, 3,          # ארבעה על GPU 3
]

STUDY_NAME = "protein_dna_binding_v7_2fold_final2"
STORAGE_FILE = "optuna_v7_2fold_final2.db"
BEST_PARAMS_FILE = "optuna_v7_2fold_final2_best.json"

# =========================================================
# SEARCH SPACE
# =========================================================

LEARNING_RATE_MIN = 5e-5
LEARNING_RATE_MAX = 8e-4

WEIGHT_DECAY_MIN = 1e-6
WEIGHT_DECAY_MAX = 1e-3

DNA_DROPOUT_MIN = 0.10
DNA_DROPOUT_MAX = 0.40

PROTEIN_DROPOUT_MIN = 0.10
PROTEIN_DROPOUT_MAX = 0.45

INTERACTION_RANK_OPTIONS = [32, 64, 96]

HUBER_WEIGHT_MIN = 0.15
HUBER_WEIGHT_MAX = 0.55

TRANSFORMER_LAYER_OPTIONS = [1, 2, 3]
TRANSFORMER_FEEDFORWARD_OPTIONS = [128, 256, 512]


# =========================================================
# Helpers
# =========================================================

def compute_trial_score(fold_scores):
    return float(np.mean(fold_scores))


def suggest_trial_parameters(trial):
    params = {}

    params["learning_rate"] = trial.suggest_float(
        "learning_rate",
        LEARNING_RATE_MIN,
        LEARNING_RATE_MAX,
        log=True,
    )

    params["weight_decay"] = trial.suggest_float(
        "weight_decay",
        WEIGHT_DECAY_MIN,
        WEIGHT_DECAY_MAX,
        log=True,
    )

    params["dna_dropout"] = trial.suggest_float(
        "dna_dropout",
        DNA_DROPOUT_MIN,
        DNA_DROPOUT_MAX,
    )

    params["protein_dropout"] = trial.suggest_float(
        "protein_dropout",
        PROTEIN_DROPOUT_MIN,
        PROTEIN_DROPOUT_MAX,
    )

    params["interaction_rank"] = trial.suggest_categorical(
        "interaction_rank",
        INTERACTION_RANK_OPTIONS,
    )

    params["huber_weight"] = trial.suggest_float(
        "huber_weight",
        HUBER_WEIGHT_MIN,
        HUBER_WEIGHT_MAX,
    )

    params["pearson_weight"] = 1.0 - params["huber_weight"]

    params["transformer_layers"] = trial.suggest_categorical(
        "transformer_layers",
        TRANSFORMER_LAYER_OPTIONS,
    )

    params["transformer_feedforward_dim"] = trial.suggest_categorical(
        "transformer_feedforward_dim",
        TRANSFORMER_FEEDFORWARD_OPTIONS,
    )

    return params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_trial_fold(
    params,
    fold_idx,
    fold,
    binding_matrix,
    dna_sequences,
    protein_embeddings,
    device,
):
    return base.run_fold(
        fold_idx=fold_idx,
        fold=fold,
        binding_matrix=binding_matrix,
        dna_sequences=dna_sequences,
        protein_embeddings=protein_embeddings,
        device=device,
        model_dim=64,
        protein_dim=256,
        dna_dropout=params["dna_dropout"],
        transformer_layers=params["transformer_layers"],
        transformer_heads=4,
        transformer_feedforward_dim=params["transformer_feedforward_dim"],
        transformer_dropout=params["dna_dropout"],
        protein_dropout=params["protein_dropout"],
        interaction_rank=params["interaction_rank"],
        initial_gating_strength=1.0,
        huber_weight=params["huber_weight"],
        pearson_weight=params["pearson_weight"],
        huber_delta=1.0,
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        max_epochs=OPTUNA_MAX_EPOCHS,
        patience=OPTUNA_PATIENCE,
        save_checkpoint=False,
    )


def make_objective(
    folds,
    binding_matrix,
    dna_sequences,
    protein_embeddings,
    device,
):
    selected_fold_indices = list(range(OPTUNA_FOLDS))

    def objective(trial):
        params = suggest_trial_parameters(trial)
        trial_seed = RANDOM_SEED + trial.number * 1000
        set_seed(trial_seed)

        print("\n" + "#" * 70)
        print(f"OPTUNA TRIAL {trial.number}")
        print("#" * 70)
        print("Parameters:", params)

        fold_results = []
        fold_scores = []

        for local_step, fold_idx in enumerate(selected_fold_indices):
            set_seed(trial_seed + fold_idx)

            result = run_trial_fold(
                params=params,
                fold_idx=fold_idx,
                fold=folds[fold_idx],
                binding_matrix=binding_matrix,
                dna_sequences=dna_sequences,
                protein_embeddings=protein_embeddings,
                device=device,
            )

            fold_results.append(result)
            fold_scores.append(result["best_val_pearson"])

            running_score = compute_trial_score(fold_scores)

            print(
                f"[Trial {trial.number}] Fold {fold_idx + 1}: "
                f"Pearson={result['best_val_pearson']:.4f} "
                f"| Running score={running_score:.4f}"
            )

            trial.report(running_score, step=local_step)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_score = compute_trial_score(fold_scores)
        trial.set_user_attr("fold_results", fold_results)
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("final_score", final_score)

        return final_score

    return objective


def create_storage():
    storage_url = f"sqlite:///{STORAGE_FILE}"

    return optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={
            "connect_args": {
                "timeout": 120,
            }
        },
    )


def create_or_load_study(sampler_seed=RANDOM_SEED):
    sampler = optuna.samplers.TPESampler(
        seed=sampler_seed
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
    )

    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=create_storage(),
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def worker_main(worker_id, trials_for_worker):
    if trials_for_worker <= 0:
        return

    print(
        f"[Worker {worker_id}] starting with "
        f"{trials_for_worker} trials"
    )

    set_seed(RANDOM_SEED + worker_id * 100000)

    folds = base.build_folds()
    binding_matrix, dna_sequences, protein_embeddings = base.load_all_data()

    if torch.cuda.is_available():
        gpu_idx = GPU_ASSIGNMENTS[worker_id - 1]
        torch.cuda.set_device(gpu_idx)
        device = torch.device(f"cuda:{gpu_idx}")

        print(
            f"[Worker {worker_id}] assigned to GPU {gpu_idx}"
        )
    else:
        device = torch.device("cpu")
        print(f"[Worker {worker_id}] assigned to CPU")

    study = create_or_load_study(
    sampler_seed=RANDOM_SEED + worker_id * 100003
)

    objective = make_objective(
        folds=folds,
        binding_matrix=binding_matrix,
        dna_sequences=dna_sequences,
        protein_embeddings=protein_embeddings,
        device=device,
    )

    study.optimize(
        objective,
        n_trials=trials_for_worker,
        gc_after_trial=True,
    )

    print(f"[Worker {worker_id}] finished")


def split_trials_across_workers():
    workers = min(N_PARALLEL_WORKERS, N_TRIALS)
    base_trials = N_TRIALS // workers
    remainder = N_TRIALS % workers

    allocation = []

    for worker_idx in range(workers):
        allocation.append(
            base_trials + (1 if worker_idx < remainder else 0)
        )

    return allocation


def save_best_result():
    study = create_or_load_study()

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        print("No completed trials found.")
        return

    best_trial = study.best_trial

    output = {
        "study_name": STUDY_NAME,
        "best_trial": best_trial.number,
        "best_score": best_trial.value,
        "best_params": best_trial.params,
        "fold_scores": best_trial.user_attrs.get("fold_scores"),
        "fold_results": best_trial.user_attrs.get("fold_results"),
        "settings": {
            "n_trials": N_TRIALS,
            "parallel_workers": N_PARALLEL_WORKERS,
            "optuna_folds": OPTUNA_FOLDS,
            "max_epochs": OPTUNA_MAX_EPOCHS,
            "patience": OPTUNA_PATIENCE,
        },
    }

    with open(BEST_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print("OPTUNA SUMMARY")
    print("=" * 70)
    print("Completed trials:", len(completed_trials))
    print("Best trial:", best_trial.number)
    print("Best score:", best_trial.value)
    print("Best parameters:")

    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    print("Best result saved to:", BEST_PARAMS_FILE)


def main():
    if OPTUNA_FOLDS > base.N_FOLDS:
        raise ValueError(
            "OPTUNA_FOLDS cannot be larger than N_FOLDS."
        )

    if N_PARALLEL_WORKERS < 1:
        raise ValueError(
            "N_PARALLEL_WORKERS must be >= 1."
        )

    if len(GPU_ASSIGNMENTS) != N_PARALLEL_WORKERS:
        raise ValueError(
            "GPU_ASSIGNMENTS length must equal "
            "N_PARALLEL_WORKERS."
        )

    base.print_environment()
    base.check_input_files()

    print("\n" + "=" * 70)
    print("V7 PARALLEL OPTUNA")
    print("=" * 70)
    print("Total trials:", N_TRIALS)
    print("Parallel workers:", N_PARALLEL_WORKERS)
    print("Folds per trial:", OPTUNA_FOLDS)
    print("Max epochs per fold:", OPTUNA_MAX_EPOCHS)
    print("Patience:", OPTUNA_PATIENCE)
    print("=" * 70)

    create_or_load_study()

    allocation = split_trials_across_workers()
    print("Trial allocation:", allocation)

    context = mp.get_context("spawn")
    processes = []

    for worker_id, worker_trials in enumerate(allocation, start=1):
        process = context.Process(
            target=worker_main,
            args=(worker_id, worker_trials),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    failed_processes = [
        process.pid
        for process in processes
        if process.exitcode != 0
    ]

    if failed_processes:
        print(
            "Warning: worker processes with errors:",
            failed_processes,
        )

    save_best_result()


if __name__ == "__main__":
    main()
