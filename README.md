# DSRTC-Net
# A DSM-Guided Multimodal Unmixing Network with Spectral-Discriminative Representation and Topographic-Consistency Constraints
The code in this toolbox implements the "A DSM-Guided Multimodal Unmixing Network with Spectral-Discriminative Representation and Topographic-Consistency Constraints".
# System-specific notes
The code was tested in the environment of `Python 3.9.16`, `torch 2.4.1` and `CUDA 11.6`.
## Repository structure
```text
DSRTC-Net/
├── DSRTCNet.py          # DSRTC-Net, DAHCA, and spatial attention modules
├── loss_function.py     # DSE-Reg, SAD, and RMSE losses
├── utility.py           # Data loading, VCA initialization, metrics, and plotting
├── DSRTCNet_main.py              # Current training/evaluation entry point
├── quick_test.py        # Synthetic forward/backward functionality test
├── requirements.txt     # Python dependencies
├── Datasets/            # User-provided .mat datasets (not redistributed)
└── result/              # Generated abundance maps, endmembers, and metrics
```

## Requirements
The code requires Python and the following packages:
- PyTorch
- torchvision
- NumPy
- SciPy
- pandas
- h5py
- Matplotlib
Install the dependencies with:
```bash
pip install -r requirements.txt
```
For GPU execution, install the PyTorch build compatible with the local CUDA version. Before publication, replace this paragraph with the **exact environment used to reproduce the reported results**, including Python, PyTorch, CUDA, and GPU versions.

## Quick test
A synthetic quick test is provided to verify that DSRTC-Net and DSE-Reg can complete one forward and backward pass without external datasets.
```bash
python quick_test.py
```
Expected output:
```text
DSRTC-Net quick test passed.
Device: cpu or cuda
Abundance shape: (1, 4, 8, 8)
Endmember shape: (16, 4, 1, 1)
Reconstruction shape: (1, 16, 8, 8)
Total loss: <finite value>
```
This quick test checks code functionality only and does not reproduce the quantitative results reported in the manuscript.

## Running an experiment
The current experiment entry point is `DSRTCNet_main.py`.

1. Open `test.py`.
2. Set the dataset identifier:
```python
dataset = "Muffle"
```
3. Run:
```bash
python test.py
```
The script automatically selects the corresponding training parameters, trains DSRTC-Net, estimates abundance maps and endmembers, and saves the outputs.

## Contact
For questions about the code, please contact:
- Yuchao Ji: `s24160008@s.upc.edu.cn`
- Mingming Xu: `xumingming@upc.edu.cn`
