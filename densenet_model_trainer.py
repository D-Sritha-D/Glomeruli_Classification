import os
import matplotlib.pyplot as plt
import tensorflow as tf
from config import DatasetConfig

class ModelTrainer:
    """Handles model training with comprehensive metric tracking"""
    
    def __init__(self, model, train_gen, val_gen, results_dir):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
    def train(self):
        """Train the model with improved monitoring and visualization"""
        # Create checkpoint directory
        checkpoint_path = os.path.join('model', 'best_model.keras')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Enhanced callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_f1_score',  # Monitor F1 score instead of accuracy
                save_best_only=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_f1_score',  # Monitor F1 score for LR reduction
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                mode='max'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_score',  # Monitor F1 score for early stopping
                patience=10,
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.CSVLogger(
                os.path.join(self.results_dir, 'training_log.csv'),
                separator=',',
                append=False
            )
        ]
        
        # Train with class weights if specified in config
        class_weights = None
        if hasattr(DatasetConfig.TRAINING, 'class_weights'):
            class_weights = DatasetConfig.TRAINING['class_weights']
        
        history = self.model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=DatasetConfig.TRAINING['epochs'],
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        self._save_training_plots(history)
        return history
    
    def _save_training_plots(self, history):
        """Save comprehensive training history plots"""
        # Define metrics to plot
        metrics = {
            'accuracy': 'Training and Validation Accuracy',
            'loss': 'Training and Validation Loss',
            'precision': 'Training and Validation Precision',
            'recall': 'Training and Validation Recall',
            'f1_score': 'Training and Validation F1 Score',
            'auc': 'Training and Validation AUC'
        }
        
        # Create plots for each metric
        for metric, title in metrics.items():
            plt.figure(figsize=(10, 6))
            
            # Plot training metric
            if metric in history.history:
                plt.plot(history.history[metric], label=f'Training {metric}')
            
            # Plot validation metric
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                plt.plot(history.history[val_metric], label=f'Validation {metric}')
            
            plt.title(title)
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plt.savefig(os.path.join(self.results_dir, f'training_validation_{metric}.png'))
            plt.close()
        
        # Save learning rate plot
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate over Training')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, 'learning_rate.png'))
        plt.close()
        
    def save_model_summary(self):
        """Save model architecture summary to a text file"""
        with open(os.path.join(self.results_dir, 'model_summary.txt'), 'w') as f:
            # Redirect stdout to file to capture model summary
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            
            # Add training configuration
            f.write('\nTraining Configuration:\n')
            f.write(f"Epochs: {DatasetConfig.TRAINING['epochs']}\n")
            f.write(f"Initial Learning Rate: {self.model.optimizer.learning_rate.numpy()}\n")
            if hasattr(DatasetConfig.TRAINING, 'class_weights'):
                f.write(f"Class Weights: {DatasetConfig.TRAINING['class_weights']}\n")