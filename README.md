# Object Detection Using YOLOv8

A deep learning project that performs object detection using three state-of-the-art model architectures: **YOLOv8**, **Faster R-CNN**, and **RetinaNet**. YOLOv8 is the primary model trained via the Ultralytics library, while Faster R-CNN and RetinaNet leverage Facebook's Detectron2 framework. All models are trained on a custom 52-class dataset sourced from Roboflow and evaluated using confusion matrices, precision-recall curves, and F1 curves.

---

## Project Structure

```
├── Yolov8.ipynb                  # YOLOv8 training & inference pipeline
├── RetinaNet_and_Faster_R_CNN.ipynb  # Detectron2-based pipelines
└── README.md
```

---

## Dataset

- **Source:** Custom object detection dataset downloaded from [Roboflow](https://roboflow.com/)
- **Workspace / Project:** `alexa-fiverr / alexa-tybnf`
- **Classes:** 52 object classes
- **Formats:**
  - `YOLOv8` format — used for YOLOv8 training
  - `COCO JSON` format — used for Faster R-CNN and RetinaNet (via Detectron2)
- **Splits:** `train`, `valid`, and `test` sets

---

## Model Architectures

### 1. YOLOv8 (Primary Model)
- **Library:** [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Base Model:** `yolov8s.pt` (pre-trained on COCO)
- **Training Config:**
  - Epochs: `5`
  - Image Size: `800 × 800`
  - Task: `detect`
- **Evaluation Outputs:** Confusion matrix, Precision-Recall curve, F1 curve, Results plot, Validation batch predictions

### 2. Faster R-CNN
- **Framework:** [Detectron2](https://github.com/facebookresearch/detectron2)
- **Config:** `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml`
- **Backbone:** ResNet-50 + FPN
- **Training Config:**
  - Max Iterations: `5000`
  - Base Learning Rate: `0.00025`
  - Images per Batch: `2`
  - ROI Batch Size per Image: `128`
  - Confidence Threshold (Inference): `0.4`
- **Evaluation:** COCO Evaluator on the test set

### 3. RetinaNet
- **Framework:** [Detectron2](https://github.com/facebookresearch/detectron2)
- **Config:** `COCO-Detection/retinanet_R_50_FPN_3x.yaml`
- **Backbone:** ResNet-50 + FPN
- **Training Config:**
  - Max Iterations: `1000`
  - Base Learning Rate: `0.0001`
  - Images per Batch: `2`
  - Gradient Clipping: Enabled (clip value: `1.0`)
- **Evaluation:** COCO Evaluator on the test set

---

## 📊 Results

| Model        | mAP@50 | mAP@50-95 | Training Duration     |
|--------------|--------|-----------|----------------------|
| YOLOv8s      | 0.426  | 0.299     | 5 epochs             |
| Faster R-CNN | —      | —         | 5000 iterations      |
| RetinaNet    | —      | —         | 1000 iterations      |

> Faster R-CNN and RetinaNet exhibited decreasing loss over their respective training iterations. Full COCO evaluation metrics are printed at the end of training in the notebook.

### Evaluation Visualizations (YOLOv8)
- `confusion_matrix.png` — Class-level prediction breakdown
- `PR_curve.png` — Precision-Recall curve
- `F1_curve.png` — F1 score vs confidence threshold
- `R_curve.png` — Recall curve
- `results.png` — Training/validation metrics over epochs
- `val_batch0_pred.jpg` — Sample validation batch with predictions

### Loss Curves (Faster R-CNN & RetinaNet)
Loss metrics logged per iteration and visualized interactively using **Plotly**:
- `total_loss`
- `loss_cls` (classification loss)
- `loss_box_reg` (bounding box regression loss)
- `loss_rpn_cls` (RPN classification loss)
- `loss_rpn_loc` (RPN localization loss)

---

## Tech Stack

| Category          | Libraries / Tools                          |
|-------------------|--------------------------------------------|
| Object Detection  | Ultralytics (YOLOv8), Detectron2           |
| Deep Learning     | PyTorch, TorchVision                       |
| Computer Vision   | OpenCV (`cv2`)                             |
| Dataset           | Roboflow                                   |
| Pose Estimation   | MediaPipe                                  |
| Visualization     | Matplotlib, Plotly, Seaborn                |
| Data Processing   | NumPy, Pandas                              |
| ML Utilities      | scikit-learn, XGBoost, joblib              |

---

## How to Run

> This project is designed to run in **Google Colab** with **GPU support** enabled.

### Setup

1. Open [Google Colab](https://colab.research.google.com/) and enable GPU:  
   `Runtime → Change runtime type → T4 GPU`

2. Mount your Google Drive (if saving outputs there):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

### YOLOv8 Pipeline

1. Open `Yolov8.ipynb` in Colab.
2. Install dependencies:
   ```bash
   pip install numpy==1.26.4 ultralytics mediapipe==0.10.11 roboflow==1.1.48
   ```
3. Download the Roboflow dataset in YOLOv8 format.
4. Train the model:
   ```bash
   yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=5 imgsz=800 plots=True
   ```
5. Run inference on a video using the best trained weights (`runs/detect/train/weights/best.pt`).

### Faster R-CNN & RetinaNet Pipeline

1. Open `RetinaNet_and_Faster_R_CNN.ipynb` in Colab.
2. Install dependencies:
   ```bash
   pip install -U torch torchvision torchaudio opencv-python-headless
   pip install -U git+https://github.com/facebookresearch/detectron2.git
   ```
3. Download the Roboflow dataset in COCO format.
4. Register COCO datasets and run training via `DefaultTrainer`.
5. Evaluate on the test set using `COCOEvaluator`.
6. Run inference on images or videos using `DefaultPredictor`.

---

## Inference

### YOLOv8 — Video Inference
```python
from ultralytics import YOLO

model = YOLO("/content/runs/detect/train/weights/best.pt")
results = model(video_path, save=True, save_dir="/content")
```

### Faster R-CNN / RetinaNet — Image Inference
```python
from detectron2.engine import DefaultPredictor

predictor = DefaultPredictor(cfg)
outputs = predictor(image)  # image loaded via cv2.imread()
```

---

## Notes

- The Roboflow dataset contains **52 object classes** spanning gym/fitness equipment based on the dataset project name and class count.
- For Faster R-CNN video inference, per-frame detection statistics (class counts and average confidence scores) are saved to `./output/detection_stats.json`.
- RetinaNet uses gradient clipping (`clip_value=1.0`) to prevent loss explosion during training.
- All Detectron2 models use `ResNet-50 + FPN` as the backbone, initialized with COCO pre-trained weights.
