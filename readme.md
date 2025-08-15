# Comparing 3D and 2D Models for Object Detection of Bacterial Motors
This repository contains our solutions for the **BYU - Locating Bacterial Flagellar Motors 2025** Kaggle competition.
## 3D U-Net

Our first approach is based on the [1st place solution by @brendanartley](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/writeups/bartley-1st-place-3d-u-net-quantile-thresholding) and utilizes a **3D U-Net** with a **Swin3D-B encoder**. The model is trained using heavy augmentations and an auxiliary loss to effectively localize bacterial motors in noisy cryo-electron tomograms.

<h1 align="center">
<img src="3D_Unet/imgs/model.jpg" alt="Model Architecture" width="800">
</h1>
<p align="center"><em>Figure: Model architecture from the 1st place solution.</em></p>

---

## 📋 Prerequisites

Before you begin, you need to manually download the data and pre-trained models.

1.  **Competition Data:** Download the data from the [competition page](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025) and place the `train` and `test` folders inside `/argusdata4/naamagav/byu/`.
2.  **Pre-trained Encoder:** Download the Swin3D-B weights (`swin3d_b_1k-24f7c7c6.pth`) from [this link](https://download.pytorch.org/models/swin3d_b_1k-24f7c7c6.pth) and place the file in `model_zoo/`.
3.  **Folds CSV:** Download the `folds_all.csv` file from [this dataset](https://www.kaggle.com/datasets/brendanartley/solution-ds-byu-1st-place-metadata) and place it in the `processed/` directory.

Your final directory structure should look like this:

```
/argusdata4/naamagav/byu/
├── checkpoints/
├── model_zoo/
│   └── swin3d_b_1k-24f7c7c6.pth
├── processed/
│   └── folds_all.csv
├── test/
├── train/
└── sample_submission.csv
└── train_labels.csv
```

---

## ⚙️ Setup and Installation

This project uses `conda` for environment management.

1.  **Install Miniconda (if not already installed):**

Get bash script for miniconda download from [here](https://docs.conda.io/en/main/miniconda.html#linux-installers)

2.  **Create and Activate the Conda Environment:**
    ```bash
    # Set up channels for best dependency resolution
    conda config --set channel_priority flexible
    conda config --add channels conda-forge nvidia pytorch

    # Create and activate the environment
    conda create --name byu_env python=3.10
    conda activate byu_env
    ```

3.  **Install Dependencies:**
    ```bash
    # Install PyTorch and MONAI via Conda
    conda install pytorch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 pytorch-cuda==11.8 monai==1.4.0 wandb==0.19.6 tqdm==4.67.1

    # Install remaining packages via Pip
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

The workflow is divided into preprocessing, training, and inference.

### 1. Preprocessing

First, process the raw tomograms into a format suitable for training. This script will populate the `/argusdata4/naamagav/byu/processed/` directory.

```bash
python -m src.pre.run
```

### 2. Training

To train the model, you can run the provided script. The default hyperparameters are listed below. A full training run on an NVIDIA L40 takes approximately 17.5 hours for 250 epochs.

```bash
# For local execution 
bash run.sh

# For SLURM-based clusters
sbatch slurm.sh

```
#### Hyperparameters
* **Model type** `3D U-Net`
* **Encoder** `Swin3D-B`
* **Number of Parameters:** `108_913_754`
* **Optimizer:** `AdamW`
* **Learning Rate:** `1e-4`
* **Weight Decay:** `1e-4`
* **Scheduler:** `Constant`
* **Epochs:** `250`
* **Batch Size:** `12`
* **Loss Function:** `SmoothBCE (main head + deep supervision + pooled loss)`
* **ROI Size:** `(64, 672, 672)`

---

### 3. Inference

Inference is done through Kaggle using the `byu-submission.ipynb` notebook.  
[View the Kaggle submission notebook here](https://www.kaggle.com/code/naama123/byu-submission/notebook?scriptVersionId=256102251). The submission notebook is based on the submission notebook of @brendanartley: 
[View brendanartley notebook here](https://www.kaggle.com/code/brendanartley/byu-1st-place-submission)




## 📚 References

1. [1st place solution of the BYU - Locating Bacterial Flagellar Motors 2025 competition by @brendanartley](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/writeups/bartley-1st-place-3d-u-net-quantile-thresholding)
2. [Solution Metadata Dataset by @brendanartley](https://www.kaggle.com/datasets/brendanartley/solution-ds-byu-1st-place-metadata)
3. [Competition Homepage: BYU - Locating Bacterial Flagellar Motors 2025](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025)
4. [Submission notebook by @brendanartley](https://www.kaggle.com/code/brendanartley/byu-1st-place-submission)
5. [@brendanartley Github repository](https://github.com/brendanartley/BYU-competition) 