# Alzheimer's Disease Detection using Ensemble Deep Learning Models

This repository contains the implementation of an efficient ensemble approach for Alzheimer's Disease detection using MRI images, as described in the paper:

**"An Efficient Ensemble Approach for Alzheimer's Disease Detection Using an Adaptive Synthetic Technique and Deep Learning"**  
[MDPI Diagnostics 2023](https://www.mdpi.com/2075-4418/13/15/2489)

## Project Overview

This project reproduces and implements several ensemble deep learning models for multiclass classification of Alzheimer's Disease stages from MRI images, addressing class imbalance using the ADASYN oversampling technique.

Key components:
- Implementation of 5 ensemble models combining different architectures
- Custom CNN architecture
- ADASYN oversampling for handling class imbalance
- Performance evaluation on Alzheimer's MRI dataset

## Implemented Models

### Ensemble Models
1. **EfficientNetB2 + DenseNet121**
2. **EfficientNetB2 + VGG16** (Primary model from paper)
3. **EfficientNetB2 + Xception**
4. **VGG16 + DenseNet121**
5. **Xception + DenseNet121**

### Custom Model
6. Custom CNN architecture

## Dataset

The model uses the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset with four classes:
- Mild Demented
- Moderate Demented
- Non-Demented
- Very Mild Demented

Original class distribution is highly imbalanced, addressed using ADASYN oversampling.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- imbalanced-learn (for ADASYN)
- OpenCV
- NumPy
- pandas
- scikit-learn

Install requirements:
```bash
pip install -r requirements.txt
```

## Project Structure

```
alzheimer-detection/
├── data/                   # Dataset directory
├── models/                 # Saved model weights
├── notebooks/              # Jupyter notebooks for experimentation
├── src/
│   ├── data_preprocessing.py   # Data loading and ADASYN implementation
│   ├── models.py               # Model architectures
│   ├── train.py                # Training script
│   └── utils.py                # Utility functions
├── results/                 # Evaluation metrics and plots
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Usage

1. **Data Preparation**
   - Place dataset in `data/` directory
   - Run preprocessing script:
     ```bash
     python src/data_preprocessing.py
     ```

2. **Training Models**
   - Train all ensemble models:
     ```bash
     python src/train.py --model all
     ```
   - Train specific model (e.g., EfficientNetB2+VGG16):
     ```bash
     python src/train.py --model efficientnet_vgg
     ```

3. **Evaluation**
   - Evaluation metrics are saved in `results/`
   - Confusion matrices and performance plots are generated automatically

## Results

Reproduced results from the paper (on balanced dataset):

| Model                      | Accuracy | Precision | Recall | F1 Score | AUC   |
|----------------------------|----------|-----------|--------|----------|-------|
| EfficientNetB2+DenseNet121 | 96.96%   | 97.00%    | 96.98% | 96.93%   | 99.60%|
| VGG16+EfficientNetB2       | 97.35%   | 97.32%    | 97.35% | 97.37%   | 99.64%|
| EfficientNetB2+Xception    | 96.26%   | 96.24%    | 96.50% | 96.25%   | 99.11%|
| VGG16+DenseNet121          | 95.56%   | 95.50%    | 95.23% | 95.50%   | 98.75%|
| Xception+DenseNet121       | 91.05%   | 91.50%    | 91.00% | 90.75%   | 97.78%|

