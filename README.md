# RAP-LDP: Reinforcement Adaptive Perturbation for Local Differential Privacy in Large Language Models

This repository contains the official implementation of **RAP-LDP**, a framework leveraging Reinforcement Learning to dynamically optimize privacy budgets for Local Differential Privacy (LDP) in LLMs.

## Overview

RAP-LDP reformulates the privacy-preserving fine-tuning process as a sequential decision-making problem. A Soft Actor-Critic (SAC) agent dynamically coordinates:
1.  **Adaptive Layer Fusion:** Aggregating multi-scale features from different layers.
2.  **Semantics-Aware Noise Allocation:** Injecting heterogeneous noise based on semantic density (Attention Entropy).

## Project Structure

*   `rap_ldp_components.py`: Core implementation of the RL agent (SAC), Environment state encoding, and the RAP-LDP perturbation mechanism.
*   `train_rap_ldp.py`: Main training script for fine-tuning LLMs with RAP-LDP.
*   `run_rap_ldp.sh`: Shell script with optimal hyperparameters to reproduce the paper's main results.
*   `ldp_mechanisms.py`: Utility functions for differential privacy noise generation.
*   `train_baselines.py`: Scripts to reproduce baselines (DP-SGD, DP-Forward, DP-LoRA).
*   `privacy_analysis.py`: Privacy attack evaluation suite (Membership Inference Attacks, Embedding Inversion).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Reproduce Main Results (Recommended)

We provide a script `run_rap_ldp.sh` with the optimal hyperparameters used in our paper (achieving ~90.8% accuracy on SST-2). This is the best starting point for reproduction.

```bash
bash run_rap_ldp.sh
```

### 2. Manual Training

To manually fine-tune a model (e.g., BERT-base) on SST-2 with RAP-LDP:

```bash
python train_rap_ldp.py \
    --model_name_or_path bert-base-uncased \
    --task_name sst2 \
    --target_epsilon 8.0 \
    --output_dir ./output/rap_ldp_sst2_eps8
```


### 3. Privacy Attack Evaluation

To evaluate the privacy guarantees against MIA and Embedding Inversion:

```bash
python privacy_analysis.py \
    --model_path ./output/rap_ldp_sst2_eps8 \
    --task_name sst2 \
    --epsilon 8.0
```

## Citation

If you use this code, please cite our paper:

```bibtex
@article{huang2026rapldp,
  title={RAP-LDP: Reinforcement Adaptive Perturbation for Local Differential Privacy in Large Language Models},
  author={Huang, Jiawang and Wang, Ran and Chen, Aidong and Chen, Wenwen},
  journal={Under Review},
  year={2026}
}
```
