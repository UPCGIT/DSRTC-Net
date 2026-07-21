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
## Contact
For questions about the code, please contact:
- Yuchao Ji: `s24160008@s.upc.edu.cn`
- Mingming Xu: `xumingming@upc.edu.cn`
