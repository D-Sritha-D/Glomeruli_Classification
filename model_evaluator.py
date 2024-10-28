import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

class ModelEvaluator:
    """Handles model evaluation and visualization"""
    
    def __init__(self, model, test_generator, class_names):
        self.model = model
        self.test_generator = test_generator
        self.class_names = class_names
        
    def evaluate(self, history):
        """Evaluate model performance"""
        all_labels = []
        all_preds = []
        
        for i in range(len(self.test_generator)):
            batch_x, batch_y = self.test_generator[i]
            pred = self.model.predict(batch_x)
            all_labels.extend(np.argmax(batch_y, axis=1))
            all_preds.extend(np.argmax(pred, axis=1))
            
        self._calculate_metrics(all_labels, all_preds)
        self._create_visualizations(all_labels, all_preds, history)
    
    def _calculate_metrics(self, true_labels, predictions):
        """Calculate and display classification metrics"""
        metrics = {
            'Accuracy': accuracy_score(true_labels, predictions),
            'Precision': precision_score(true_labels, predictions, average='binary'),
            'Recall': recall_score(true_labels, predictions, average='binary'),
            'F1 Score': f1_score(true_labels, predictions, average='binary')
        }
        
        for metric_name, value in metrics.items():
            print(f'{metric_name}: {value:.4f}')
    
    def _create_visualizations(self, true_labels, predictions, history):
        """Create and display evaluation visualizations"""
        self._plot_confusion_matrix(true_labels, predictions)
        self._plot_roc_curve(true_labels, predictions)
        self._plot_learning_curves(history)
    
    def _plot_confusion_matrix(self, true_labels, predictions):
        """Plot confusion matrix"""
        conf_matrix = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
    
    def _plot_roc_curve(self, true_labels, predictions):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    def _plot_learning_curves(self, history):
        """Plot training and validation accuracy/loss curves"""
        plt.figure(figsize=(10, 6))
        
        # Plot accuracy curves
        plt.subplot(2, 1, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss curves
        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
