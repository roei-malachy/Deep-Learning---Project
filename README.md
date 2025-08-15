# YOLO - Fast detection  
**YOLOv8**  

This repository contains our YOLOv8-based solution for the **BYU - Locating Bacterial Flagellar Motors 2025** Kaggle competition.

Our approach focuses on using a **2D object detection model (YOLOv8n)** to detect bacterial motors in cryo-electron tomograms.  
The pipeline includes preprocessing of the YOLO dataset, training with early stopping, and an inference stage with a **3D distance-based NMS** to improve final predictions.

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

This project runs directly in the Kaggle Notebook environment.  
However, if running locally, follow these steps:

```bash
# Python environment
conda create --name byu_yolo_env python=3.10
conda activate byu_yolo_env

# Install YOLOv8 and dependencies
pip install ultralytics
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib pillow pyyaml
```

---

## 🚀 Usage  

The workflow is divided into **preprocessing**, **training**, and **inference**.

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
