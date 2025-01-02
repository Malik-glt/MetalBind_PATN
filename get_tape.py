import os
import torch
from tape import ProteinBertModel, TAPETokenizer
from tqdm import tqdm
import numpy as np
import glob
import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-in", "--path_input", type=str, help="The path of input fasta folder")
parser.add_argument("-out", "--path_output", type=str, help="The path of output .tape folder")

def main(input_folder, out_folder, miss_txt):
    # Ensure the output folder exists
    os.makedirs(out_folder, exist_ok=True)

    input_files = glob.glob(os.path.join(input_folder, "*"))

    # Initialize the TAPE model and tokenizer
    model = ProteinBertModel.from_pretrained('bert-base')
    tokenizer = TAPETokenizer(vocab='iupac')

    for path in tqdm(input_files, desc="Processing", unit="file"):
        with open(path) as f:
            fasta = f.readlines()

        title = fasta[0][1:].strip()  # Extract title from header
        sequence = fasta[1].strip()   # Extract sequence

        out_path = os.path.join(out_folder, f"{title}.tape")  # Save with .tape extension

        try:
            # Tokenize the sequence
            token_ids = torch.tensor([tokenizer.encode(sequence)])

            # Get the model output
            with torch.no_grad():
                output = model(token_ids)

            # Extract embeddings, excluding [CLS] and [SEP] tokens
            sequence_output = output[0][:, 1:-1, :].cpu().numpy()

            # Save embeddings in text format with .tape extension
            with open(out_path, 'w') as f_out:
                for aa_embedding in sequence_output[0]:
                    line = ' '.join(map(str, aa_embedding))
                    f_out.write(line + '\n')

        except Exception as e:
            # Log missing or error files
            log_mode = 'a' if os.path.exists(miss_txt) else 'w'
            with open(miss_txt, log_mode) as tape_miss:
                tape_miss.write(f"{title}.fasta\n")
            print(f"Error processing {title}: {e}")
            continue

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.path_input, args.path_output, "tape_miss")
