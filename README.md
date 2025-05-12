# Object Detection for Waste Recognition 🧠♻️

**Bachelor's Thesis Project by Mario Cicalese**  
Computer Science — University of Salerno, 2023

This repository contains the code and documentation for my Bachelor's thesis, which focuses on developing an object detection model aimed at identifying various categories of static and dynamic objects in low-visibility images captured at night and/or during adverse weather conditions (nighttime, fog, rain, and snow) by a dashcam installed inside a car. The system leverages deep learning, specifically a Faster R-CNN (Resnet50) architecture.

## 📖 Project Overview

The aim of this thesis project is to build a deep learning-based object detection system for real-time object recognition to support the perception system of autonomous vehicles in low-visibility and/or adverse weather conditions. The model identifies object types across 10 categories, including static (e.g., traffic signs) and dynamic (e.g., riders, pedestrians, etc.) elements, returning both the object's position and its predicted class.

The project includes:
- Dataset collection and preprocessing
- Model training with fasterrcnn_resnet50_fpn
- Evaluation and visualization of predictions
- 
---

## ⚙️ Technologies Used

- **PyTorch (model)**
- **Python 3.10**
- **OpenCV**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**
- **Google Colab**
---

## 📁 Dataset

The dataset used for training and evaluation is ACDC Dataset (Adverse Conditions Dataset - https://arxiv.org/abs/2104.13395). The images were recorded in Switzerland on both urban and suburban roads using a GoPro Hero 5 camera mounted on the vehicle’s interior windshield. The images, extracted from recorded videos, have a resolution of 1920x1080, and the objects within them are annotated using coordinates stored in a separate JSON file, used for object detection tasks. It includes:
- 21 object classes (I reduced the object classes to 10)
- 4,000+ 1920x1080 RGB annotated images (bounding boxes)
- Formats: VOC and COCO

**Suggested image for this section:**  
Insert a sample image showing a few annotated waste items (bounding boxes + class labels). You can capture it from Roboflow or your training samples.

---
