# Hazard View Bird - On-device Disaster Scene Parsing

This project focuses on the development of an efficient model for semantic segmentation on edge devices, specifically targeting the analysis of disaster scenes from images captured by unmanned aerial vehicles (UAVs). The key challenge is to optimize the model for accuracy, speed, and power consumption, given the computational constraints of edge computing environments. Our goal is to enable rapid and accurate understanding of disaster-stricken areas through advanced computer vision techniques, facilitating quicker response times for relief operations.

## Table of Contents

- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Results](#results)
- [Metrics](#metrics)
- [Classification Labels](#classification-labels)

## Dependencies

The project requires `Python 3.6.9` and the following libraries:

- PyTorch 1.11.1
- OpenCV 4.7.0.72
- NumPy 1.19.5
- Pillow 8.4.0

To install the dependencies, use the provided `requirements.txt` file:

```bash
python3.6 -m pip install --upgrade pip
python3.6 -m pip install -r requirements.txt
```

## Dataset

The dataset comprises 1,700 images collected by UAVs in disaster-affected areas. The images are segmented into 14 classes representing various elements and conditions found in disaster scenes. Place the dataset in the `dataset/` directory following this structure:

```plaintext
├── dataset
│   ├── train
│   │    ├── IMG
│   │    │    └──train
│   │    └── GT
│   │         └──train
│   └── val
│       ├── IMG
│       │    └──val
│       └── GT
│       │    └──val
```

## Results

Our evaluation focused on various architectures and backbones, emphasizing their performance in terms of the Dice Coefficient—a critical measure for segmentation tasks. Below is a summary of our findings:

| Architecture | Backbone           | Dice Coefficient |
|--------------|--------------------|------------------|
| FPN          | EfficientNetB0     | 0.56             |
| FPN          | MobileNetV2        | 0.52             |
| UNET         | MobileNetV2        | 0.508            |
| UNET         | EfficientNetB0     | 0.47             |
| PSPNET       | MobileNetV2        | 0.47             |


## Metrics

The models were evaluated based on two main metrics:

- **Accuracy**: Measured by the Dice Coefficient across all 14 categories. The Dice Coefficient is a statistical tool that measures the similarity between two samples. This metric is particularly useful in segmentation tasks to quantify the model's performance in predicting the overlap between the predicted segmentation and the ground truth.

- **Speed**: Assessed as the average runtime for processing one frame, expressed in seconds per frame (s/f). This metric is crucial for applications requiring real-time or near-real-time processing capabilities, especially in disaster response scenarios where timely information can significantly impact decision-making processes.

## Classification Labels

The dataset includes images classified into 14 distinct categories, representing various elements commonly observed in disaster-stricken areas:

**Background**
1. **Avalanche**
2. **Building Undamaged**
3. **Building Damaged**
4. **Cracks/Fissure/Subsidence**
5. **Debris/Mud/Rock Flow**
6. **Fire/Flare**
7. **Flood/Water/River/Sea**
8. **Ice Jam Flow**
9. **Lava Flow**
10. **Person**
11. **Pyroclastic Flow**
12. **Road/Railway/Bridge**
13. **Vehicle**


