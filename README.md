# Glomeruli Classification Project

## Overview
This project implements a deep learning solution for classifying glomeruli images into two categories: globally sclerotic and non-globally sclerotic. The model uses a fine-tuned ResNet50 architecture with custom top layers for improved performance on this specific medical imaging task.

## Approach

### Machine Learning Pipeline
- **Base Architecture**: ResNet50 pre-trained on ImageNet
- **Custom Modifications**:
  - Enhanced preprocessing layer
  - Fine-tuning strategy with frozen early layers
  - Dual dense block structure with residual connections
  - Dropout layers for regularization
  - L2 regularization on dense layers
  - Batch normalization for improved training stability

### Data Preprocessing
1. **Image Processing**:
   - Resizing to 224x224 pixels
   - RGB conversion
   - Normalization (pixel values scaled to [0,1])
   - ResNet50 specific preprocessing
2. **Augmentation Techniques**:
   - Implemented via custom data generator
   - Real-time batch processing
   - Memory-efficient data handling

### Dataset Division
- Training set: 70%
- Validation set: 15%
- Test set: 15%

## Performance Metrics
- **Training Accuracy**: 81.82%
- **AUC**: 0.8863
- **Precision**: 0.8182
- **Recall**: 0.8182
- **Validation Accuracy**: 
- **Training Time**: ~2 hours on GPU


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

### Training the Model
1. Activate the virtual environment:
```bash
# For macOS/Linux
source venv/bin/activate

# For Windows
.\venv\Scripts\activate
```
2. Run each cell in the glomeruli.classification.ipynb notebook

## Model File
The trained model is available in the following folder at: [finalModel](https://www.dropbox.com/home/Durga%20Sritha%20Dongla/Glomeruli_Classification_Model)

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
[ResNet and its application to medical image processing: Research progress and challenges](https://www.sciencedirect.com/science/article/pii/S0169260723003255)
[Deep Learning in Image Classification using Residual Network (ResNet) Variants for Detection of Colorectal Cancer](https://www.sciencedirect.com/science/article/pii/S1877050921000284)
I have also referred to the following [GitHub Repository](https://github.com/JyXUU/Binary-Classification-of-Glomeruli/tree/main) of Jingyi Xu

## License
This project is licensed under the MIT License - see the LICENSE file for details.
