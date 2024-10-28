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
        self._plot_precision_recall_curve(true_labels, predictions)
        self._plot_learning_curves(history)
        self._plot_feature_importance(self.model, self.test_generator.feature_names)
        self._plot_misclassified_samples(true_labels, predictions, self.test_generator)
        self._plot_class_metrics(true_labels, predictions, self.class_names)
        self._plot_calibration_curve(true_labels, predictions)
    
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
    
    def _plot_precision_recall_curve(self, true_labels, predictions):
        """Plot precision-recall curve"""
        precision, recall, _ = precision_recall_curve(true_labels, predictions)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
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
    
    def _plot_misclassified_samples(self, true_labels, predictions, test_data):
        """Plot misclassified samples"""
        misclassified_indices = np.where(true_labels != predictions)[0]
        
        if len(misclassified_indices) > 0:
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            
            for i, idx in enumerate(misclassified_indices[:9]):
                row = i // 3
                col = i % 3
                axes[row, col].imshow(test_data[idx][0], cmap='gray')
                axes[row, col].set_title(f'True: {self.class_names[true_labels[idx]]}\nPred: {self.class_names[predictions[idx]]}')
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.show()
        else:
            print('No misclassified samples found.')
    
    def _plot_class_metrics(self, true_labels, predictions, class_names):
        """Plot class-specific metrics"""
        class_metrics = {
            'Precision': precision_score(true_labels, predictions, average=None),
            'Recall': recall_score(true_labels, predictions, average=None),
            'F1-Score': f1_score(true_labels, predictions, average=None)
        }
        
        plt.figure(figsize=(10, 6))
        
        for metric_name, metric_values in class_metrics.items():
            plt.bar(np.arange(len(class_names)), metric_values)
            plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
            plt.title(f'{metric_name} per Class')
            plt.xlabel('Class')
            plt.ylabel(metric_name)
            plt.tight_layout()
            plt.show()
    
    def _plot_calibration_curve(self, true_labels, predictions):
        """Plot calibration curve"""
        from sklearn.calibration import calibration_curve
        
        prob_true, prob_pred = calibration_curve(true_labels, predictions, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.plot(prob_pred, prob_true, marker='.')
        plt.title('Calibration Curve')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('True Probability')
        plt.legend()
        plt.tight_layout()
        plt.show()