# D4Explainer: In-distribution Explanations of Graph Neural Network via Discrete Denoising Diffusion [NeurIPS 2023]
This is the Pytorch implementation of " D4Explainer: In-distribution Explanations of Graph Neural Network via Discrete Denoising Diffusion"
## Requirements

- `torch==1.10.1`
- `torch-geometric==2.0.4`
- `numpy==1.24.2`
- `pandas==1.5.3`
- `networkx==3.0`

Refer to `requirements.txt` for more details.


## Dataset

Download the datasets from [here](https://drive.google.com/drive/folders/1pwmeST3zBcSC34KbAL_Wvi-cFtufAOCE?usp=sharing) to `data/`

**Datasets Included:**

- Node classification: `BA_shapes`; `Tree_Cycle`; `Tree_Grids`; `cornell`
- Graph classification: `mutag`; `ba3`; `bbbp`; `NCI1`

## Train Base GNNs
```
cd gnns
python ba3motif_gnn.py
python bbbp_gnn.py
python mutag_gnn.py
python nci1_gnn.py
python synthetic_gnn.py --data_name Tree_Cycle
python synthetic_gnn.py --data_name BA_shapes
python tree_grids_gnn.py
python web_gnn.py
```


## Train and Evaluate D4Explainer
For example, to train D4Explainer on Mutag, run:
```
python main.py --dataset mutag
```


## Evaluation of Other Properties

- In-distribution: `python -m evaluation.ood_evaluation`
- Robustness: `python -m evaluation.robustness`


