# YOLO - Fast detection  
YOLOv8n (Ultralytics)

This repository contains our YOLOv8n-based Kaggle Notebook solution for the **BYU - Locating Bacterial Flagellar Motors 2025** competition.

Our approach uses a 2D object detection model (**YOLOv8n**, Ultralytics) to detect bacterial motors in cryo-electron tomograms.  
The pipeline includes preprocessing, training, and inference with a 3D distance-based NMS to improve final predictions.


---

## Model Architecture  
The architecture is based on the Ultralytics YOLOv8 framework with pretrained weights (`yolov8n.pt`).  
The model was fine-tuned on a custom YOLO-formatted dataset derived from the competition data.

---

## 📋 Prerequisites
Required Kaggle inputs:
- `/kaggle/input/byu-locating-bacterial-flagellar-motors-2025`
- `/kaggle/input/parse-data/yolo_dataset`

**Competition Data:**
    /kaggle/input/byu-locating-bacterial-flagellar-motors-2025/
    ├── train/
    ├── test/
    ├── train_labels.csv
    └── sample_submission.csv

**YOLO Dataset (Parsed Data):**
    /kaggle/input/parse-data/yolo_dataset/
    ├── images/{train,val}/
    └── labels/{train,val}/

> If the `val` folders are missing, the script will automatically create an **80/20 train/val split**.


---

## ⚙️ Setup and Installation
In the **first cell** of your Kaggle Notebook, run:
```
    !pip install ultralytics==8.3.0
```
Then simply click **Run All** in Kaggle — no further installation or internet connection is required.

---









## 🚀 Usage  
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
- **Optimizer**: AdamW  
- **Learning Rate**: 0.001  
- **Epochs**: 150  
- **Batch Size**: 16  
- **Early Stopping**: Patience = 8 epochs  
- **Loss Function**: Box Loss, Classification Loss, Distribution Focal Loss  
- **Confidence Threshold**: 0.45  
- **3D NMS Distance**: 60 px  

---

### 3. Inference  
Runs inference on the competition test set, applies 3D NMS, and generates the submission file.

```bash
python yolo_fast_detection.py
```
The `submission.csv` will be saved in `/kaggle/working/`.

---

## 📚 References  
1. https://www.kaggle.com/code/willongwang/byu-submission-baseline-ipynb/notebook
2. [Competition Homepage: BYU - Locating Bacterial Flagellar Motors 2025](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025)

