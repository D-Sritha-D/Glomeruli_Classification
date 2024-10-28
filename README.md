# Glomeruli Classification Project

## Overview
In this project, I implemented a deep learning solution to classify glomeruli images into two categories: globally sclerotic and non-globally sclerotic. I developed a fine-tuned ResNet50 architecture and added custom top layers to enhance the model's performance for this specific medical imaging task. I chose to work with ResNet because of it's high performance on medical image data and also it's ability to be more generalized.

## Approach

### Machine Learning Pipeline
- **Base Architecture**:
- The model is built on **ResNet50** as its foundational architecture, utilizing **pre-trained ImageNet weights** to efficiently capture essential visual features.
- **Transfer learning** is applied by:
  - **Freezing the initial layers** of ResNet50 to retain foundational features.
  - **Fine-tuning the last 30 layers** to adapt specifically to the current task.
- **Custom layers** are added on top of the base model:
  - Dense layers with **dropout** and **batch normalization** to reduce overfitting and enhance generalization.
- The model outputs a **binary prediction** using **softmax activation** for classification.
- The **Adam optimizer** is employed along with **categorical crossentropy loss** for efficient learning and convergence.

### Data Preprocessing
- Sets up a **data generator for glomeruli image classification**, feeding data batches efficiently into a deep learning model.
- **Initializes key parameters** such as:
  - **Batch size**
  - **Image dimensions**
  - **Data shuffling** preference
- **Calculates the required number of batches** based on the dataset size.
- **Loads images** from a specified directory, then:
  - **Resizes images** to the target input dimensions.
  - **Normalizes pixel values** for consistent input scaling.
- **Prepares labels in one-hot encoded format** to ensure compatibility with the model’s expected input.
- Includes functionality to **shuffle data at the end of each epoch** to enhance training robustness and variability.

### Model Training
- **Model checkpointing** saves the best-performing version of the model during training.
- **Learning rate reduction** adjusts the learning rate when performance plateaus, enabling finer adjustments.
- **Early stopping** halts training to prevent overfitting when improvements stop.
- **Training history** (accuracy and loss metrics) is recorded and saved to allow:
  - **Performance analysis** on training and validation datasets.
  - **Evaluation** of model stability and optimization.

### Dataset Division
- Training set: 60%
- Validation set: 20%
- Test set: 20%

## Performance Metrics
- **Training Accuracy**: 94.53%
- **AUC**: 0.9948
- **Precision**: 0.8542
- **Recall**: 0.8241
- **Validation Accuracy**: 94.18
- **Training Time**: ~2 hours on M2 mac chip


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
ResNet and its application to medical image processing: Research progress and challenges [link](https://www.sciencedirect.com/science/article/pii/S0169260723003255)

Deep Learning in Image Classification using Residual Network (ResNet) Variants for Detection of Colorectal Cancer [link](https://www.sciencedirect.com/science/article/pii/S1877050921000284)
