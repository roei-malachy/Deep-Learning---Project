# YOLO - Fast detection  
**YOLOv8**  

This repository contains our YOLOv8-based solution for the **BYU - Locating Bacterial Flagellar Motors 2025** Kaggle competition.

Our approach focuses on using a **2D object detection model (YOLOv8n)** to detect bacterial motors in cryo-electron tomograms.  
The pipeline includes preprocessing, training, and inference with a **3D distance-based NMS** to improve final predictions.

---

## Model Architecture  
The architecture is based on the Ultralytics YOLOv8 framework with pretrained weights (`yolov8n.pt`).  
The model was fine-tuned on a custom YOLO-formatted dataset derived from the competition data.

---

## 📋 Prerequisites  
Before you begin, you need to ensure the competition data is available in the Kaggle environment.

**Competition Data:**  
Download the competition dataset from the Kaggle page and ensure it appears in `/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/`.

**YOLO Dataset (Parsed Data):**  
A YOLO-formatted dataset must exist under `/kaggle/input/parse-data/yolo_dataset/` with the structure:
```
/kaggle/input/parse-data/yolo_dataset/
├── images/{train,val}/
└── labels/{train,val}/
```
If `val` folders are missing, the script will automatically create an 80/20 train/val split.

Your final Kaggle directory structure should look like this:
```
/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/
├── train/
├── test/
├── train_labels.csv
└── sample_submission.csv

/kaggle/input/parse-data/yolo_dataset/
├── images/{train,val}/
└── labels/{train,val}/
```

---

## ⚙️ Setup and Installation  
1. Install Miniconda (if not already installed):


---

## 🚀 Usage  

1. **Install Miniconda (if not already installed):**  
Get bash script for Miniconda download from [here](https://docs.conda.io/en/latest/miniconda.html)  

2. **Create and Activate the Conda Environment:**  
# Set up channels for best dependency resolution  
conda config --set channel_priority flexible  
conda config --add channels conda-forge nvidia pytorch  

# Create and activate the environment  
conda create --name yolo_env python=3.10  
conda activate yolo_env  

3. **Install Dependencies:**  
# Install PyTorch via Conda (with CUDA 11.8)  
conda install pytorch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 pytorch-cuda==11.8  

# Install remaining packages via Pip  
pip install ultralytics==8.3.0 numpy pandas matplotlib pillow pyyaml  


---

### 1. Preprocessing  
Checks and fixes dataset paths, and creates validation split if needed.
```python
from yolo_fast_detection import prepare_dataset
yaml_path = prepare_dataset()
```

---

### 2. Training  
Runs YOLOv8 training with the hyperparameters below.  
A full run on an NVIDIA T4 GPU with 150 epochs takes around 1–1.5 hours.

```python
from yolo_fast_detection import train_yolo_model
model, results = train_yolo_model(
    yaml_path="fixed_dataset.yaml",
    pretrained_weights_path="yolov8n.pt",
    epochs=150,
    batch_size=16,
    img_size=640
)
```

**Hyperparameters**  
- Optimizer: AdamW  
- Learning Rate: 0.001  
- Epochs: 150  
- Batch Size: 16  
- Early Stopping: Patience = 8 epochs  
- Loss Function: Box Loss, Classification Loss, Distribution Focal Loss  
- Confidence Threshold: 0.45  
- 3D NMS Distance: 60 px  

---

### 3. Inference  
Runs inference on the competition test set, applies 3D NMS, and generates the submission file.

```bash
python yolo_fast_detection.py
```
The `submission.csv` will be saved in `/kaggle/working/`.

---

## 📚 References  
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)  
- BYU - Locating Bacterial Flagellar Motors 2025 competition page  
