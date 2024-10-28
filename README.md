# Glomeruli Classification Project

## Overview
In this project, I implemented a deep learning solution to classify glomeruli images into two categories: globally sclerotic and non-globally sclerotic. I developed a fine-tuned ResNet50 architecture and added custom top layers to enhance the model's performance for this specific medical imaging task. I chose to work with ResNet because of it's high performance on medical image data and also it's ability to be more generalized.

## Approach

### Machine Learning Pipeline
- **Base Architecture**:
The model is built using ResNet50 as its foundational architecture, leveraging pre-trained ImageNet weights to capture essential visual features efficiently. Transfer learning is implemented by freezing the initial layers of ResNet50, ensuring that the foundational features are retained, while fine-tuning the last 30 layers to adapt specifically to the task at hand. On top of the base model, additional custom layers are incorporated, including dense layers with dropout and batch normalization, which help reduce overfitting and improve generalization. For classification, the model outputs a binary prediction with softmax activation, using the Adam optimizer with categorical crossentropy loss for effective learning and convergence.

### Data Preprocessing
This code sets up a data generator for image classification of glomeruli images, designed to feed batches of data into a deep learning model efficiently. It begins by initializing key parameters such as batch size, image dimensions, and whether to shuffle the data, then calculates the number of batches needed based on the dataset size. During data retrieval, the code loads images from a specified directory, resizes them to the target input dimensions, and normalizes their pixel values. It also prepares labels in a one-hot encoded format, ensuring the data is in a format compatible with the model's requirements. Finally, it includes functionality to shuffle the data at the end of each epoch, promoting a more robust and varied training process.

### Model Training
In training, several techniques are applied to optimize model performance and stability. Model checkpointing is used to save the best-performing version throughout the training process. To improve learning efficiency, learning rate reduction is employed whenever performance plateaus, allowing the model to make finer adjustments. Additionally, early stopping helps prevent overfitting by ceasing training once improvements level off. Training history, including both accuracy and loss metrics, is documented and saved to support analysis and evaluation of the model’s performance on training and validation datasets.

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
ResNet and its application to medical image processing: Research progress and challenges [link](https://www.sciencedirect.com/science/article/pii/S0169260723003255)

Deep Learning in Image Classification using Residual Network (ResNet) Variants for Detection of Colorectal Cancer [link](https://www.sciencedirect.com/science/article/pii/S1877050921000284)
