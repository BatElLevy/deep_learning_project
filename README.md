# DNA-Protein Binding Prediction

This directory contains the complete submission for the DNA-protein binding prediction task, including the final prediction pipeline, trained model checkpoints, precomputed representations, input data, and the generated predictions for all 64 test DBPs.

## Git LFS

This repository uses **Git LFS (Large File Storage)** for the pretrained SimBind model checkpoints.

Please make sure Git LFS is installed before cloning the repository:

```bash
git lfs install
git clone https://github.com/BatElLevy/dna_protein_binding.git
cd dna_protein_binding
git lfs pull
```

## Running the Submission

From the `submission/` directory, run:

```bash
bash run_all.sh
```

The final predictions are written to:

```text
predictions/
```

with one file per test protein (`DBP1.txt`-`DBP64.txt`).

## Submission Structure

### Main prediction pipeline

* `main.py` - main entry point for generating the final test predictions.
* `run_all.sh` - runs the complete prediction pipeline.
* `requirements.txt` - Python dependencies required to run the code.

### Deep learning model

* `main_v7.py` - final V7 model architecture and training implementation.
* `main_v7_functional.py` - functional version of the V7 model used by the final pipeline.
* `checkpoints_v7/` - trained models from the 10 cross-validation folds.
* `optuna_v7_parallel.py` - Optuna hyperparameter optimization used to select the final V7 configuration.

### Protein representations

* `embedding.py` - generates protein embeddings.
* `train_protein_embeddings.pt` - precomputed embeddings for the training proteins.
* `test_protein_embeddings.pt` - precomputed embeddings for the test proteins.

### Similarity-based model

* `simbind_custom/` - SimBind-based protein-similarity prediction implementation.
* `simbind_top14_precomputed.npz` - precomputed protein-similarity information used by the final prediction pipeline.

### Data

* `training_DBPs.txt` - training protein sequences.
* `training_seqs.txt` - training DNA sequences.
* `training_data.zip` - training binding measurements.
* `test_DBPs.txt` - test protein sequences.
* `test_seqs.txt` - test DNA sequences.

### Predictions

* `predictions/` - final binding predictions for the 64 test proteins. Each `DBP*.txt` file contains the predictions for one test DBP across the test DNA sequences.

## Method Overview

The final approach combines two complementary prediction strategies:

1. A deep learning model that integrates DNA sequence features with pretrained protein embeddings.
2. A SimBind-based similarity model that transfers binding information from related training proteins.

Predictions from the trained cross-validation models are aggregated and combined with the similarity-based signal to produce the final prediction for each test DBP.
