import os
import matplotlib.pyplot as plt
import tensorflow as tf
from config import DatasetConfig

class ModelTrainer:
    """Handles model training"""
    
    def __init__(self, model, train_gen, val_gen, results_dir):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.results_dir = results_dir
        
    def train(self):
        """Train the model"""
        checkpoint_path = os.path.join('model', 'best_model.keras')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=3,
                min_lr=1e-6
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=18,
                restore_best_weights=True
            )
        ]
        
        history = self.model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=DatasetConfig.TRAINING['epochs'],
            callbacks=callbacks
        )
        
        self._save_training_plots(history)
        return history
    
    def _save_training_plots(self, history):
        """Save training history plots"""
        metrics = {
            'accuracy': 'Training and Validation Accuracy',
            'loss': 'Training and Validation Loss'
        }
        
        for metric, title in metrics.items():
            plt.figure(figsize=(10, 6))
            plt.plot(history.history[metric], label=f'Training {metric}')
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(title)
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.savefig(os.path.join(self.results_dir, f'training_validation_{metric}.png'))
            plt.close()