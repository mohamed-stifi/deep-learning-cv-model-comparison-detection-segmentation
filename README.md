---

# **deep-learning-cv-model-comparison-detection-segmentation**

### 📊 Comparative analysis of deep learning models for object detection, semantic segmentation, and instance segmentation using PASCAL VOC and Penn-Fudan datasets.

---

## 📌 Overview

This project presents a comparative study of state-of-the-art deep learning models across three core computer vision tasks:

- **Object Detection**
- **Semantic Segmentation**
- **Instance Segmentation**

We evaluate models on performance, training dynamics, and optimizer effectiveness using standard datasets and metrics.

---

## 🧠 Objectives

- Evaluate and compare multiple deep learning models on different vision tasks.
- Understand the impact of optimizers (Adam, AdaGrad, RMSprop) on convergence and gradient stability.
- Use quantitative metrics (mAP, IoU, pixel accuracy, etc.) for model assessment.
- Visualize training behavior, gradient flow, and detection/segmentation outcomes.

---

## 🗂️ Project Structure

```
deep-learning-cv-model-comparison-detection-segmentation/
│
├── Notebooks
|    │
|    ├── Object detection/object detection.ipynb                   # Object detection with Faster R-CNN and SSD
|    ├── Instance segmentation/instance segmenatation.ipynb        # Instance segmentation with Mask R-CNN and Panoptic FPN
|    └── Semantic segmentation/semantic segmentation.ipynb         # Semantic segmentation  
|
└── README.md                          # Project documentation
```

---

## 📚 Tasks & Models

### **1. Object Detection**
- 📦 **Dataset**: PASCAL VOC 2012
- 🧠 **Models**:
  - Faster R-CNN (Two-stage)
  - SSD / YOLO (Single-stage)
- 📏 **Metrics**: Precision, Recall, mAP (mean Average Precision)

---

### **2. Semantic Segmentation**
- 📦 **Dataset**: PASCAL VOC 2012 (segmentation annotations)
- 🧠 **Models**:
  - DeepLabV3 (ResNet-101 backbone)
  - HRNet
- 📏 **Metrics**:
  - Pixel Accuracy (PA)
  - Mean Pixel Accuracy (MPA)
  - Mean Intersection over Union (mIoU)

---

### **3. Instance Segmentation**
- 📦 **Dataset**: Penn-Fudan Pedestrian
- 🧠 **Models**:
  - Mask R-CNN (Two-stage)
  - Panoptic FPN (Single-stage)
- 📏 **Metrics**:
  - mIoU
  - Precision
  - Recall
  - mAP

---

## ⚙️ Optimization Strategies

We compare three widely-used optimizers across all models:

- **Adam** (most stable across experiments)
- **AdaGrad**
- **RMSprop**

Gradient behavior, learning rates, and training stability were analyzed and visualized for each optimizer.

---

## 📊 Results Summary

| Task | Best Model | Best Optimizer | Highlight |
|------|------------|----------------|-----------|
| Object Detection | Faster R-CNN | Adam | Highest mAP |
| Semantic Segmentation | DeepLabV3 | Adam | Robust mIoU |
| Instance Segmentation | Mask R-CNN | Adam | Smooth convergence |

> Note: Panoptic FPN suffered from metric logging issues that need debugging.

---

## 📷 Visual Outputs

- 📈 Loss and gradient plots per epoch
- 📉 Learning rate vs. performance trends
- 🎯 Detection bounding boxes and segmentation masks (ground truth vs prediction)
- 📊 Metric heatmaps and optimizer comparisons

All results are saved in the `results/` folder.

---

## 🚀 Getting Started

### 📦 Requirements
- Python 3.8+
- PyTorch
- TorchVision
- NumPy, OpenCV, Matplotlib, Seaborn, etc.



---

## 🏁 Future Work

- Add COCO dataset evaluations
- Extend to panoptic segmentation benchmarking
- Incorporate advanced optimizers (Lookahead, Ranger)
- Experiment with model quantization and pruning

---

## 📚 References

- Ren et al. (2015), *Faster R-CNN*
- Liu et al. (2016), *SSD*
- He et al. (2017), *Mask R-CNN*
- Chen et al. (2017), *DeepLabV3*
- Kirillov et al. (2019), *Panoptic FPN*
- PASCAL VOC, Penn-Fudan datasets

---

## 👨‍💻 Authors

- **Mohamed STIFI**
- **Ayoub EL ASSIOUI**

> Supervised by **Pr. Hanaa EL AFIA** – ENSIAS, Rabat

---
