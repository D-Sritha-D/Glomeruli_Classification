import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from pathlib import Path
from config import DatasetConfig

class ModelPredictor:
    """Handles prediction on evaluation images and saves results"""
    
    def __init__(self, model_path, evaluation_dir, output_path='evaluation.csv'):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the saved model
            evaluation_dir: Directory containing images to evaluate
            output_path: Path where to save the CSV results
        """
        self.model = tf.keras.models.load_model(model_path)
        self.evaluation_dir = Path(evaluation_dir)
        self.output_path = output_path
        self.image_size = DatasetConfig.TRAINING['image_size']
        
    def _preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.image_size)
            img_array = np.array(img) / 255.0
            return np.expand_dims(img_array, axis=0)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
            
    def predict_images(self):
        """
        Predict all images in the evaluation directory and save results to CSV
        """
        results = []
        supported_formats = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        
        # Get all image files from the evaluation directory
        image_files = [
            f for f in self.evaluation_dir.rglob('*')
            if f.suffix.lower() in supported_formats
        ]
        
        print(f"Found {len(image_files)} images to evaluate")
        
        for image_path in image_files:
            # Get relative path for storing in CSV
            relative_path = str(image_path.relative_to(self.evaluation_dir))
            
            # Preprocess image
            processed_image = self._preprocess_image(image_path)
            if processed_image is None:
                continue
                
            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(prediction[0])
            
            # Store result
            results.append({
                'name': relative_path,
                'ground_truth': DatasetConfig.LABELS[predicted_class]
            })
            
        # Create and save DataFrame
        if results:
            df = pd.DataFrame(results)
            df.to_csv(self.output_path, index=False)
            print(f"Results saved to {self.output_path}")
        else:
            print("No results to save")
