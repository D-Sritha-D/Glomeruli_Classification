# Glomeruli Classification Project

## Overview
This project implements and compares two deep learning architectures (ResNet50 and DenseNet169) for classifying glomeruli images into globally sclerotic and non-globally sclerotic categories.

## Dataset
- **Globally Sclerotic**: 1,054 images
- **Non-Globally Sclerotic**: 4,704 images
- **Class Ratio**: 1:4.46

### Dataset Division
- Training set: 60%
- Validation set: 20%
- Test set: 20%

## Architecture Comparison

### Base Architecture Differences

| Feature | ResNet50 | DenseNet169 |
|---------|----------|-------------|
| Base Layers | 50 | 169 |
| Connection Type | Skip Connections | Dense Connections |
| Parameter Efficiency | Moderate | High |
| Feature Reuse | Through Residuals | Through Dense Connections |
| Trainable Layers | Last 30 unfrozen | Last 50 unfrozen |

### Implementation Differences

#### ResNet Implementation
```python
model = Sequential([
    ResNet50Base,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(2, activation='softmax')
])
```

#### DenseNet Implementation
```python
model = Sequential([
    DenseNet169Base,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    LayerNormalization(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
    LayerNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    LayerNormalization(),
    Dropout(0.4),
    Dense(2, activation='softmax')
])
```

### Key Technical Differences

| Feature | ResNet Implementation | DenseNet Implementation |
|---------|---------------------|----------------------|
| Normalization | BatchNormalization | LayerNormalization |
| Dropout Rates | 0.4, 0.3 | 0.5, 0.4 |
| Regularization | None | L2 (0.01) |
| Learning Rate | 1e-4 | 5e-5 |
| Optimizer | Adam | AdamW with weight decay |
| Loss Function | Basic Categorical Crossentropy | Categorical Crossentropy with label smoothing |
| Data Augmentation | Basic | Enhanced with RandomRotation and RandomZoom |

## Training Configuration Comparison

### ResNet Training
- Simpler training setup
- Fixed learning rate
- Basic metrics (accuracy, AUC)
- Standard optimization

### DenseNet Training
- Advanced training configuration
- Adaptive learning rate with weight decay
- Comprehensive metrics (accuracy, precision, recall, F1, AUC)
- Custom F1 score monitoring
- Label smoothing for better generalization

## Implementation Features

### ResNet Model
```
Input → Conv → BatchNorm → MaxPool → [ResBlock×N] → GlobalAvgPool → Dense → Output
```

### DenseNet Model
```
Input → Conv → [DenseBlock → Transition]×N → GlobalAvgPool → Dense → Output
```

# Model Training Results

## Summary of Results

| Metric                | Model 1: ResNet          | Model 2: DenseNet      |
|-----------------------|---------------------|---------------------|
| **Training Accuracy**  | 94.53%              | 98.18%    |
| **Precision**         | 0.8542              | 0.9406 |
| **Recall**            | 0.8241              | 0.9548 |
| **Validation Accuracy**| 94.18%              | 98% |
| **Training Time**     | ~2 hours on M2 Mac  | ~1 hours on M2 Mac |


## Running Instructions

### Environment Setup

1. Clone this repository
2. Create a Python virtual environment:
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```
3. Install all dependencies using the requirements.txt file:
```bash
pip install -r requirements.txt
```

### Project Structure
```
final_glomeruli/
├── venv/                      # Virtual environment directory
├── data/
│   ├── globally_sclerotic_glomeruli/
│   └── non_globally_sclerotic_glomeruli/
│   └── public.csv
├── densenet/
│   ├── densenet_model_builder/
│   └── densenet_model_trainer/
├── resnet/
│   ├── resnet_model_builder/
│   └── resnet_model_trainer/
├── config.py                  # Configuration settings
├── data_analyzer.py           # Visualizes the data and shows all data information
├── data_generator.py          # Custom data generator
├── model_evaluator.py         # Evaluation metrics for the trained models
├── evaluation.py              # Evaluation script for the custom new data
├── main.ipynb                 # Main Notebook to run the code
├── requirements.txt           # Project dependencies

├── model/
│   └── checkpoints/          # Saved model weights 
└── results/                  # Training results and plots
```

### Training the Models
1. Activate the virtual environment:
```bash
# For macOS/Linux
source venv/bin/activate

# For Windows
.\venv\Scripts\activate
```
2. Run each cell in the main.ipynb notebook

## Model File
The models are available in the following folder at: [finalModels](https://www.dropbox.com/scl/fo/m762o7z0ku13yc17bklij/AIUBy1a3DUZoxh_uq7fGb8U?rlkey=m6cih85n64ksncfang7jbhpun&st=maitg1t1&dl=0)


### Running Evaluation
1. Place all the images you want to evaluate in a new folder and give that folder path in the cell under "7.Evaluation on New dataset"
2. Ensure you have the model downloaded and stored in a folder named: 'model'
3. Once you have the paths all set, change the paths accordingly in "7.Evaluation on New dataset" and run the cell
4. Your desired prediction csv file will be stored as 'evaluation.csv'

The script will:
- Process all images in the folder storing your images
- Generate 'evaluation.csv' with predictions
- CSV format: name,ground_truth (where ground_truth is the predicted class)


## References
ResNet and its application to medical image processing: Research progress and challenges [link](https://www.sciencedirect.com/science/article/pii/S0169260723003255)
Deep Learning in Image Classification using Residual Network (ResNet) Variants for Detection of Colorectal Cancer [link](https://www.sciencedirect.com/science/article/pii/S1877050921000284)
Dense Convolutional Network and Its Application in Medical Image Analysis [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC9060995/)
Optimization and fine-tuning of DenseNet model for classification of COVID-19 cases in medical imaging [link](https://www.sciencedirect.com/science/article/pii/S2667096821000136)
