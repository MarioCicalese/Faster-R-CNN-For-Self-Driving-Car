# 🚗 Road Object Detection for Self-Driving Systems In adverse weather conditions (nighttime, fog, rain, and snow) 🌧️❄️

**Bachelor's Thesis Project by Mario Cicalese**  
Computer Science — University of Salerno, 2023

This repository contains the code and documentation for my Bachelor's thesis, which focuses on developing an object detection model aimed at identifying various categories of static and dynamic objects in low-visibility images captured at night and/or during adverse weather conditions (nighttime, fog, rain, and snow) by a dashcam installed inside a car. The system leverages deep learning, specifically a Faster R-CNN (Resnet50) architecture.

## 📖 Project Overview

The aim of this thesis project is to build a deep learning-based object detection system for real-time object recognition to support the perception system of autonomous vehicles in low-visibility and/or adverse weather conditions. The model identifies object types across 10 categories, including static (e.g., traffic signs) and dynamic (e.g., riders, pedestrians, etc.) elements, returning both the object's position and its predicted class.

The project includes:
- Dataset collection and preprocessing
- Model training with fasterrcnn_resnet50_fpn
- Evaluation and visualization of predictions
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
  
In addition to **image pre-processing** operations performed to adapt the images for the model, **data augmentation was also applied due to the limited size of the dataset**. Specifically, the augmentation techniques used were: **Horizontal Flip** and **Random Brightness/Contrast**.

**Suggested image for this section:**  
Insert a sample image showing a few annotated waste items (bounding boxes + class labels). You can capture it from Roboflow or your training samples.

---

## 🧠 Model Architecture

The system is based on **Faster R-CNN ResNet50** PyTorch Model ([PyTorch doc link:](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)). The model offers:
- **Training Input**: During training, the model expects both the input tensors and a targets (list of dictionary), containing:
  - **boxes (FloatTensor[N, 4]):** the ground-truth boxes in [x1, y1, x2, y2] format.
  - **labels (Int64Tensor[N]):** the class label for each ground-truth box
- **Training Output**: The model returns a Dict[Tensor] during training, containing the classification and regression losses for both the RPN and the R-CNN.
- **Inference**: During inference, the model requires only the input tensors, and returns the post-processed predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as follows, where N is the number of detections:
  
   - **boxes (FloatTensor[N, 4]):** the predicted boxes in [x1, y1, x2, y2] format
   - **labels (Int64Tensor[N]):** the predicted labels for each detection
   - **scores (Tensor[N]):** the scores of each detection

> Training was performed using `yolo task=detect mode=train` with various hyperparameters optimized for our dataset.
> 
---

## 🏋️ Training Process

Key training parameters:
- Epochs: 25
- Batch size: 4
- Image size: 800x800
- Optimizer: SGD
- Loss: Objectness + Classification + Bounding Box Classification
---

## 📊 Evaluation and Results
The metrics used to evaluate the model were:
- **Average Intersection over Union (Average IoU)**: 0.60
- **Average Recall (AR):** 0.53
- **Mean Average Precision (mAP):** 0.45

### ❌ Results Problem
- **High False Negative Number:** "The reasons why the model may fail to detect objects within the images could include: low visibility, objects being too small, partially cropped, or poorly distinguishable due to the surrounding context.
- **Unbalanced Classes:** There are certain classes (e.g., buses, trains, and bicycles) that have significantly fewer instances compared to other classes.
- **Object size**: in some cases, the objects to be detected are very small and hard to see, leading the model to miss them.
- **Noisy Annotation**: For the 'Traffic Sign' and 'Traffic Light' classes, the bounding boxes are not always accurately created. In some cases, the bounding boxes are larger than they should be, including irrelevant details.
---
