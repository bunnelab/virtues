import pandas as pd
import os
from argparse import ArgumentParser
import requests

"""
A tool to download FASTA files with Uniprot's canonical aminoacid sequences.
"""
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder to save the fastas")
    parser.add_argument("--csv", type=str, required=True, help="CSV file with column 'protein_id' containing Uniprot IDs")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"File {args.csv} not found")
    
    df = pd.read_csv(args.csv)
    
    protein_ids = df['protein_id'].values

    for pid in protein_ids:
        if os.path.exists(os.path.join(args.output_dir, f"{pid}.fasta")):
            print(f"File {pid}.fasta already exists")
            continue

        url = f"https://www.uniprot.org/uniprot/{pid}.fasta"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Error downloading {pid}.fasta")
            continue

        with open(os.path.join(args.output_dir, f"{pid}.fasta"), "w") as f:
            f.write(response.text)
        print(f"Downloaded {pid}.fasta")

