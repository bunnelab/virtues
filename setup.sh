#/bin/bash

conda create -n virtues python=3.12

conda run -n virtues pip install numpy
conda run -n virtues pip install pandas
conda run -n virtues pip install einops
conda run -n virtues pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
conda run -n virtues pip install biopython
conda run -n virtues pip3 install -U scikit-learn
conda run -n virtues pip install -U matplotlib
conda run -n virtues pip install seaborn
conda run -n virtues pip install flash-attn --no-build-isolation
conda run -n virtues pip install wandb
conda run -n virtues pip install pillow
conda run -n virtues pip install umap-learn
conda run -n virtues pip install transformers
conda run -n virtues pip install safetensors
conda run -n virtues pip install huggingface_hub
conda run -n virtues pip install ipython
conda run -n virtues pip install notebook
conda run -n virtues pip install POT
conda run -n virtues pip install loguru
conda run -n virtues pip install omegaconf
conda run -n virtues pip install fair-esm