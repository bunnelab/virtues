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
    parser.add_argument("--input", type=str, required=True, help=".csv or .parquet file with column 'protein_id' containing Uniprot IDs. Alternative column name can be specified with --id_column.")
    parser.add_argument("--id_column", type=str, default="protein_id", help="Column name in the CSV file containing Uniprot IDs")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File {args.input} not found")
    
    if args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    elif args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        raise ValueError("Input file must be .csv or .parquet")
    
    protein_ids = df[args.id_column].values

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