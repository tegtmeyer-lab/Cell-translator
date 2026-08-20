# Perturbation benchmark parameter reference

| Component |Setting |
|---|---|
| Common cells | 2,106 CellOT held-out control cells |
| Genes | 18,614 |
| Perturbations | APOE, MSI2, NR1H3, LRRC4C, ANKRD12 |
| CellOT HVGs | 2,000, selected on control only |
| CellOT HVG flavor | `seurat` |
| CellOT PCA | 50 components, fit on control, `random_state=0` |
| CellOT batch size | 256 |
| CellOT test fraction | 0.20 |
| CellOT split seed | 0 |
| CellOT iterations | 100,000 |
| CellOT log frequency | 2,000 |
| CellOT eval frequency | 10,000 |
| CellOT evaluation | `iid`, `data_space` |
| CellOT full-gene reconstruction | inverse PCA on HVGs; non-HVGs filled with control mean |
| scGen epochs | 50 |
| scGen version | 2.1.1 |
| scvi-tools version | 1.3.3 |
| scGen Python | 3.10.19 |
| scGPT version | 0.2.4 |
| scGPT PyTorch | 2.3.1+cu118 |
| scGPT torchtext | 0.18.0+cpu |
| scGPT final postprocessing | `control_mean + yhat`, clip at 0 |
