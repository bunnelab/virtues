# VirTues: AI-powered virtual tissues from spatial proteomics for clinical diagnostics and biomedical discovery

*[[Preprint]](https://arxiv.org/pdf/2501.06039), [[Supplement]](https://github.com/bunnelab/virtues/blob/main/.github/supplement.pdf), [[Model]](https://huggingface.co/bunnelab/virtues), 2025* 

<img src=".github/VirTues_logo.jpg" alt="VirTues Logo" width="40%" align="right" />

*Authors:* Johann Wenckstern*, Eeshaan Jain*, Benedikt von Querfurth*, Yexiang Cheng*, Kiril Vasilev, Matteo Pariset, Phil F. Cheng, Petros Liakopoulos, Olivier Michielin, Andreas Wicki, Gabriele Gut, Charlotte Bunne

Spatial proteomics technologies have transformed our understanding of complex tissue architecture in cancer but present unique challenges for computational analysis. Each study uses a different marker panel and protocol, and most methods are tailored to single cohorts, which limits knowledge transfer and robust biomarker discovery. Here we present Virtual Tissues (VirTues), a general-purpose foundation model for spatial proteomics that learns marker-aware, multi-scale representations of proteins, cells, niches and tissues directly from multiplex imaging data. From a single pretrained backbone, VirTues supports marker reconstruction, cell typing and niche annotation, spatial biomarker discovery, and patient stratification, including zero-shot annotation across heterogeneous panels and datasets. In triple-negative breast cancer, VirTues-derived biomarkers predict anti-PD-L1 chemo-immunotherapy response and stratify disease-free survival in an independent cohort, outperforming state-of-the-art biomarkers derived from the same datasets and current clinical stratification schemes. 

<br>
<p align='center'>
<img src=".github/abstract_virtues.png" alt="VirTues Graphical Abstract" width="80%" />
</p>

# Installation
You can download the repository via:
```
git clone https://github.com/bunnelab/virtues.git
```
To facilitate imports, we recommend installing VirTues as a library via:
```
cd /local/path/to/repository/
pip install -e .
```

To create a new conda environment `virtues` with Python 3.12 and install all requirements run:
```
source setup.sh
```
The installation process should take less than 10 minutes. For a definitive list of package versions VirTues has been tested on, please refer to `version_requirements.txt`.


# Configuration & Setup
Before running VirTues, please ensure that your base configuration found in `configs/base_config` is properly setup for your system. 
This includes setting the following fields:
```yaml
experiments_dir: <path-to-dir> # path of directory to save all training runs

experiment.name: <run-name> # the name of your training run
experiment.wandb_mode: 'disabled' | 'online' | 'offline' # set to 'disabled' to disable wandb logging
experiment.wandb_entity: <entity-name> # your wandb entity name, leave empty for default
experiment.wandb_project: <project-name> # your project name

datasets_config: <path-to-config> # path to config file describing training datasets

marker_embeddings_dir: <path-to-dir> # path of directory containing marker embeddings saved as [UniprotID].pt files
```
Alternatively, you can set these fields from the command line when starting the training.

# Datasets

## Downloading existing datasets from spora

All training datasets used in VirTues (except where licensing restrictions apply) are accessible through **[spora](https://spora.epfl.ch/)**, a unified spatial proteomics ecosystem, developed in parallel with VirTues.

**spora** consists of three components:
1. **[spora [data]](https://spora.epfl.ch/datasets.html)** is a curated collection of over 31 spatial proteomics datasets, all following a harmonized data format using state-of-the-art file formats optimized for machine learning. The available datasets can be browsed [here](https://spora.epfl.ch/datasets.html).
2. **[spora [io]](https://github.com/bunnelab/spora-io)** is a python data loading interface for easy and efficient access to the data once downloaded.
3. **[spora [bench]](https://github.com/bunnelab/spora-bench)** is a benchmark suite for spatial proteomics foundation models.

To download datasets from **spora [data]** please follow the instructions provided in the [documentation](https://spora.epfl.ch/docs-data.html#downloading)

### Setting up a new dataset not contained in spora

VirTues can also be applied to datasets not (yet) contained in the official data corpus of spora. For this the datasets need to be locally converted to the dataset format of spora [data]. A detailed description of this format can found in the [documentation](https://spora.epfl.ch/docs-data.html#structure).

## Dataset configuration
After setting up all datasets, you need to configure a `.yaml` file that specifies the paths to each dataset in order to train VirTues.

For orientation and demonstration purposes, we provide an example dataset containing a single tissue in `assets/example_dataset`, along with corresponding configuration files located at `configs/datasets/example_config.yaml` and `configs/datasets/example_config_multiple_datasets.yaml`.

## Marker embeddings
Finally, for every measured marker across all datasets, a marker embedding must be precomputed using ESM-2 and stored in the directory `marker_embedding_dir` specified by `configs/base_config.yaml` following the naming convention `[UniprotID].pt`.

To faciliate this step, we provide two utility scripts:

You can automatically download FASTA files containing the canonical amino acid sequence from Uniprot with the script `utils/download_fastas.py` by specifying a `.csv` or `.parquet` file with a column containing the Uniprot IDs (including potential isoform suffixes). For this run: 
```
python -m virtues.utils.download_fastas --output_dir [PATH] --input [FILE] --id_column [COLUMN-NAME]
```
To generate ESM-2 embeddings of these sequences, you can use the script `virtues/utils/compute_esm_embeddings.py` via:
```
python -m virtues.utils.compute_esm_embeddings --input_dir [PATH1] --output_dir [PATH2] --device [cpu/cuda] --model [MODEL]
```
Resulting embedings will be saved at `[PATH2]/[MODEL]/[UniportID].pt`. The official published weights of VirTues were trained with the ESM-2 model `esm2_t30_150M_UR50D` (set as default). For very long protein sequences, video memory requirements might be high. In this case, we recommend running the script using `--device cpu` and sufficient RAM.

# Training 
After setting up the datasets, VirTues can be pretrained via the `virtues/train.py` script. For example, to train an instance of VirTues with a custom dataset config run:
```bash
python -m virtues.train experiment.name=[NAME] datasets_config=[PATH]
```

The training script also supports distributed training. For this we recommend using torchrun. For example, to train on a single node with 4 GPUs run:
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m virtues.train experiment.name=[NAME] datasets_config=[PATH]
```

All training results are stored in the `experiments_dir/experiment.name` directory.

# Models
As an alternative to training a VirTues model from scratch, we also provide pretrained weights via [Hugging Face Hub](https://huggingface.co/bunnelab/virtues).

Below we list all VirTues model instances available on Hugging Face Hub including their respective licence. For all academic purposes, we recommend using `virtues-sp32` trained on all 32 spatial proteomics datasets listed in the paper. For commercial use, we provide under a more permissive licence a model instance `virtues-sp31` trained on 31 datasets.

| Model Name | Training Data |  Licence of Model Weights | Repository Path | Segmentation Head Available | 
| --- | --- | --- | --- | --- | 
| `virtues-sp32` | 32 spatial proteomics datasets (diverse technologies) | CC BY-NC 4.0 | `virtues-sp32/model.safetensors` | yes | 
| `virtues-sp31` | 31 spatial proteomcis datasets (diverse technologies) | MIT Licence | `virtues-imc14/model.safetensors` | yes | 
| `virtues-imc14` | 14 IMC datasets | CC BY-NC 4.0 | `virtues-imc14/model.safetensors` | no | 

For instructions, how to download and load these model weights, please refer to the Demo notebooks below.

# Inference 

## Tutorial Notebooks

To help you start using a trained VirTues model for downstream analyses, the `notebooks` folder includes demonstrations of several common use cases:

- `1_demo_reconstruction.ipynb` – Shows how to use VirTues to reconstruct partially masked channels or inpaint fully masked ones.
- `2_demo_cell_phenotyping.ipynb`  – Demonstrates how to compute cell tokens with VirTues, which can be used for applications such as cell phenotyping and virtual biomarker discovery.
- `3_demo_segmentation.ipynb` - Shows how to use VirTues equipped with a segmentation head for joint cell instance and cell type segmentation.

These demonstration notebooks run on a standard desktop computer equipped with a modern GPU in under five minutes.

## Further Resources and Benchmarks
The benchmark library [**spora-bench**](https://github.com/bunnelab/spora-bench) contains further ready-to-run benchmarks for VirTues, in particular:
* An end-to-end linear-probe-based phenotyping pipeline — `spora_bench.benchmarks.run_cell_level_tasks`
* A pipeline for computing cell tokens across an entire dataset — `spora_bench.tools.compute_cell_tokens`
* An ABMIL-based tissue-level classification pipeline — `spora_bench.benchmarks.run_tissue_level_tasks`
* A standardized virtual staining benchmark — `spora_bench.benchmarks.run_virtual_staining_tasks`

Please refer to the repository of spora-bench for more details.

# License and Terms of Use

Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
Laboratory of Artificial Intelligence in Molecular Medicine, 2025

This repository and associated code are released under the MIT Licence. See `LICENSE.md` for details. 
Model weights are released under different licences due to restrictions of their respective training data. Please refer to the the section [Models](#models) for the respective licences.

# Reference
If you find our work useful in your research or if you use parts of this code please consider citing our [paper](https://arxiv.org/abs/2501.06039):

```
@article{wenckstern2025ai,
  title={{AI-powered virtual tissues from spatial proteomics for clinical diagnostics and biomedical discovery}},
  author={Wenckstern, Johann and Jain, Eeshaan and Cheng, Yexiang and von Querfurth, Benedikt and Vasilev, Kiril and Pariset, Matteo and Cheng, Phil F. and Liakopoulos, Petros and Michielin, Olivier and Wicki, Andreas and Gut, Gabriele and Bunne, Charlotte},
  journal={arXiv preprint arXiv:2501.06039},
  year={2025},
  url={https://arxiv.org/abs/2501.06039}, 
}
```
