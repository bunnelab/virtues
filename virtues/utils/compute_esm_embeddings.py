import os
import csv
from Bio import SeqIO
import time
import torch
from loguru import logger
import esm
import argparse

"""
A tool to compute ESM embeddings for protein sequences stored in fasta files.
Warning:
For extremly long sequences, script might run out of video memory. In this case, please run on CPU with sufficient RAM.
"""

class ESMWrapper:
    
    def __init__(self, model_name="esm2_t30_150M_UR50D", device="cpu"):
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model_name = model_name

    def embed(self, data, reduce=True):
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(self.device)

        layer_nr = {"esm2_t48_15B_UR50D": 48,
                    "esm2_t36_3B_UR50D": 36,
                    "esm2_t33_650M_UR50D": 33,
                    "esm2_t30_150M_UR50D": 30,
                    "esm2_t12_35M_UR50D": 12,
                    "esm2_t6_8M_UR50D": 6
                    }.get(self.model_name)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[layer_nr], return_contacts=True)

            token_representations = results["representations"][layer_nr]

        if reduce:
            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
            return torch.stack(sequence_representations, dim=0)
        else:
            return token_representations
        
    def embed_sequence(self, sequence, reduce=True):
        data = [("some protein", sequence)]
        return self.embed(data, reduce=reduce)[0]
    
    def embed_list_sequences(self, sequences, reduce=True):
        data = [("some protein", sequence) for sequence in sequences]
        return self.embed(data, reduce=reduce)


def load_dict(csv_file):
    with open(csv_file) as f:
        reader = csv.reader(f)
        d = {rows[0]:rows[1] for rows in reader}
    return d


# load sequence as string from fasta file
def load_fasta(input_file):
    fasta_sequences = SeqIO.parse(open(input_file),'fasta')
    for fasta in fasta_sequences:
        return str(fasta.seq)


#load all sequences from fasta file and embeds them
def main(input, output, model, device):
    files = os.listdir(input)
    files = [f for f in files if f.endswith(".fasta")]
    
    data = []
    embeddings = {}

    logger.info("Loading sequences...")
    
    for file in files:
        id = file.split(".")[0]
        seq = load_fasta(os.path.join(input, file))
        data.append((id, seq))

    logger.info("Loaded " + str(len(data)) + " sequences")
    logger.info("Starting embedding process...")
    
    os.makedirs(os.path.join(output, model), exist_ok=True)

    esm = ESMWrapper(device=device, model_name=model)

    start = time.time()
    for id, sequence in data:
        target_file_path = os.path.join(output, model, id + ".pt")

        if os.path.isfile(target_file_path):
            logger.info("Embedding for " + id + " already exists. Skipping...")
            continue

        logger.info("Sequence: " + sequence + ", Length: " + str(len(sequence)))
        with torch.no_grad():
            encoding = esm.embed_sequence(sequence)
        embeddings.update({id: encoding})
        torch.save(encoding.cpu(), target_file_path)
        del encoding
        torch.cuda.empty_cache()
    end = time.time()
    logger.info("Finished embedding process in " + str(end - start) + " seconds")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Embed protein sequences using ESM')
    parser.add_argument("--input_dir", type=str, help="Path to folder containing fasta strings")
    parser.add_argument("--output_dir", type=str, help="Path to folder to save embeddings by model")
    parser.add_argument("--model", type=str, default="esm2_t30_150M_UR50D")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.model, device=args.device)