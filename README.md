# PIFCA: Efficient Federated Clustering with Gradient Search Optimization for Medical Edge Networks

Official implementation of the paper:  
**"PIFCA: Federated Clustering Addressing Non-IID Skew With Gradient Space Disentanglement"**  
[[Paper]](link-to-your-paper) | [[Project Page]](link-to-project-page)

---

## Overview
PIFCA is a **Preliminary Iterative Federated Clustering Algorithm** designed to efficiently partition clients in federated learning under **non-IID** settings. It leverages **gradient-space search** and a **synthetic sampling dataset** to determine optimal client clusters in the early stages of training, significantly improving accuracy and communication efficiency.

**Key Features:**
- **Early clustering** using unstable gradients via synthetic sampling.
- **Gradient combination search** for optimal group allocation.
- **Scalable** to large client numbers and dynamic user participation.
- **Plug-in** capability to enhance existing FL algorithms.

---

## Datasets
We use three datasets from **MedMNIST** for medical imaging experiments:
- **DermaMNIST**
- **BloodMNIST**
- **OrganAMNIST**

Each dataset is distributed to clients using a **Dirichlet distribution** with α ∈ {0.1, 1, 100} to simulate different levels of heterogeneity.

---

## Installation

```bash
git clone https://github.com/Network-Optimization/PIFCA.git
cd PIFCA
conda create -n pifca python=3.8
conda activate pifca
pip install -r requirements.txt
```

---

## Running Experiments

### PIFCA
```bash
# Example: OrganAMNIST dataset, α=0.1
python main1.py -data OrganAMNIST_0.1 -m CNN -algo Local -gr 100 -lr 0.001 -ncl 11 -dev cuda -did 0,1
```

Alternatively, define multiple experiments in `sh1-1.py` and run:
```bash
python sh1-1.py
```

### Baseline Algorithms
```bash
python sh1-1.py  # Runs predefined baselines with stored datasets
```

### Cluster Partitioning
- `PIFCA-de.ipynb` → DermaMNIST clustering  
- `PIFCA-or+bl.ipynb` → OrganAMNIST & BloodMNIST clustering  
- Pass clustering results into `serveravg_test.py` and set:
```python
a = [1, 1, 1, 1, 1, 0, 0, 0, 0, 1]  # Same label for clients in the same cluster
```

---

## Results

### Accuracy under α=0.1 (Highly Non-IID)
| Dataset       | Best Baseline | PIFCA  | Gain   |
|---------------|--------------|--------|--------|
| DermaMNIST    | 70.42        | 74.28  | +3.86% |
| OrganAMNIST   | 61.75        | 71.55  | +9.80% |
| BloodMNIST    | 82.90        | 87.82  | +4.92% |
| CIFAR-10      | 44.71        | 51.74  | +7.03% |
| CIFAR-100     | 21.84        | 26.75  | +4.91% |

### Communication Rounds to Reach 75% of Max Accuracy
PIFCA reduces required rounds by **29.9%** on average compared to baselines.

---

## Citation
If you use this code, please cite:
```bibtex
@article{pifca2025,
  title={PIFCA: Federated Clustering Addressing Non-IID Skew With Gradient Space Disentanglement},
  author={Your Name and Others},
  journal={IEEE INFOCOM},
  year={2026}
}
```

---

## Acknowledgements
- Code framework adapted from [PFLlib](https://www.pfllib.com/docs.html)
- Dataset source: [MedMNIST](https://medmnist.com/)

