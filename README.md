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
├── config.py                  # Configuration settings
├── data_generator.py          # Custom data generator
├── model_builder.py           # Model architecture definition
├── model_trainer.py           # Training pipeline
├── model_evaluator.py         # Evaluation metrics
├── evaluation.py              # Evaluation script
├── requirements.txt           # Project dependencies
├── data/
│   ├── globally_sclerotic_glomeruli/
│   └── non_globally_sclerotic_glomeruli/
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
2. Run each cell in the glomeruli.classification.ipynb notebook

## Model File
The models are available in the following folder at: [finalModels]([https://www.dropbox.com/home/Durga%20Sritha%20Dongla/Glomeruli_Classification_Model](https://www.dropbox.com/scl/fo/m762o7z0ku13yc17bklij/AIUBy1a3DUZoxh_uq7fGb8U?rlkey=m6cih85n64ksncfang7jbhpun&st=maitg1t1&dl=0))


### Running Evaluation
1. Place all the images you want to evaluate in a new folder and give that folder path in the cell under "7.Evaluation on New dataset"
2. Ensure you have the model downloaded and stored in a folder named: 'model'
3. Once you have the paths all set, change the paths accordingly in "7.Evaluation on New dataset" and run the cell
4. Your desired prediction csv file will be stored as 'evaluation.csv'

The script will:
- Process all images in the folder storing your images
- Generate 'evaluation.csv' with predictions
- CSV format: name,ground_truth (where ground_truth is the predicted class)

## Project Insights
1. **Dataset Challenges**:
   - Imbalanced class distribution
   - Varying image quality
   - Limited dataset size
2. **Model Selection**:
   - ResNet50 showed superior performance compared to other architectures
   - Transfer learning crucial for good performance
   - Custom top layers improved accuracy by ~5%
3. **Training Strategy**:
   - Early stopping prevented overfitting
   - Learning rate reduction on plateau improved convergence
   - Batch size of 32 provided optimal training stability

## Future Improvements
1. Implement more extensive data augmentation
2. Explore ensemble methods
3. Add attention mechanisms
4. Collect and annotate more training data
5. Implement cross-validation

## References
ResNet and its application to medical image processing: Research progress and challenges [link](https://www.sciencedirect.com/science/article/pii/S0169260723003255)

Deep Learning in Image Classification using Residual Network (ResNet) Variants for Detection of Colorectal Cancer [link](https://www.sciencedirect.com/science/article/pii/S1877050921000284)
