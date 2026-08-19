# pip install esm@git+https://github.com/Biohub/esm.git@main
import os
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

def load_protein_sequences(file_path):
    """
    Loads sequences from a file.
    Supports both simple format (one line per protein)
    and standard FASTA format (ignores lines starting with '>').
    """
    sequences = []
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return sequences

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):
                sequences.append(line)
    print(f"Successfully loaded proteins from {file_path}: {len(sequences)}")
    return sequences

def extract_and_save_embeddings(sequences, model, tokenizer, output_path, batch_size=16):
    """
    Extracts embeddings in batches from the penultimate layer of ESMC-300M
    and saves them as a dictionary {row_index: tensor_960}.
    """
    if not sequences:
        return

    embeddings_dict = {}
    device = model.device

    # Iterate through sequences in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc=f"Processing {os.path.basename(output_path)}"):
        batch_seqs = sequences[i : i + batch_size]

        # Tokenize batch with automatic padding
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            # Request hidden states of all layers
            outputs = model(**inputs, output_hidden_states=True)

            # Extract penultimate layer (index -2 in hidden_states)
            # Shape: [Batch_Size, Max_Length_In_Batch, 960]
            penultimate_layer = outputs.hidden_states[-2]

            # Correctly mask padding
            attention_mask = inputs["attention_mask"].unsqueeze(-1) # [Batch, Length, 1]
            masked_embeddings = penultimate_layer * attention_mask

            # Compute sum along the protein length and divide by real length (Mean Pooling)
            summed_embeddings = torch.sum(masked_embeddings, dim=1)
            summed_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            global_embeddings = summed_embeddings / summed_mask # [Batch_Size, 960]

            # Move to CPU and save to dictionary
            global_embeddings = global_embeddings.cpu()
            for batch_idx, emb in enumerate(global_embeddings):
                global_idx = i + batch_idx
                embeddings_dict[global_idx] = emb

    # Save PyTorch dictionary to disk
    torch.save(embeddings_dict, output_path)
    print(f"Embeddings saved to {output_path} (File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")

def main():
    # File path settings
    TRAIN_PROTEINS_FILE = "training_DBPs.txt"
    TEST_PROTEINS_FILE = "test_DBPs.txt"

    TRAIN_EMBEDDINGS_OUT = "train_protein_embeddings.pt"
    TEST_EMBEDDINGS_OUT = "test_protein_embeddings.pt"

    # 1. Load sequences
    train_seqs = load_protein_sequences(TRAIN_PROTEINS_FILE)
    test_seqs = load_protein_sequences(TEST_PROTEINS_FILE)

    if not train_seqs and not test_seqs:
        print("No data to process. Exiting.")
        return

    # 2. Initialize ESMC-300M model
    print("Loading ESMC-300M model...")
    model_name = "biohub/ESMC-300M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True
    ).to(device).eval()

    print(f"Model successfully loaded on device: {model.device}")

    # 3. Extract and save embeddings
    extract_and_save_embeddings(train_seqs, model, tokenizer, TRAIN_EMBEDDINGS_OUT, batch_size=16)
    extract_and_save_embeddings(test_seqs, model, tokenizer, TEST_EMBEDDINGS_OUT, batch_size=16)

    print("\nDone! All proteins successfully vectorized.")

if __name__ == "__main__":
    main()