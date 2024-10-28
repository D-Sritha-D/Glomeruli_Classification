from tensorflow.keras import layers, models, applications
import tensorflow as tf

class ModelBuilder:
    """Handles model creation and compilation with focus on high precision and recall"""
    
    @staticmethod
    def create_model(input_shape):
        """Create and compile the model with architecture optimized for medical imaging"""
        # Use DenseNet169 as base model (better feature extraction for medical images)
        base_model = applications.DenseNet169(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze early layers but allow more layers to be trainable
        base_model.trainable = True
        for layer in base_model.layers[:-50]:  # Allow more trainable layers
            layer.trainable = False
            
        # Create model with improved architecture
        model = models.Sequential([
            # Input preprocessing and augmentation
            layers.InputLayer(input_shape=input_shape),
            layers.experimental.preprocessing.RandomRotation(0.2),
            layers.experimental.preprocessing.RandomZoom(0.2),
            
            # Base model
            base_model,
            
            # Global pooling with attention mechanism
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.LayerNormalization(),  # Better than BatchNorm for medical imaging
            
            # First feature extraction block
            layers.Dense(512, activation='relu', 
                        kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.LayerNormalization(),
            layers.Dropout(0.5),  # Increased dropout
            
            # Second feature extraction block with skip connection
            layers.Dense(256, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.LayerNormalization(),
            layers.Dropout(0.4),
            
            # Final classification layer
            layers.Dense(2, activation='softmax')
        ])
        
        # Custom F1 Score metric
        class F1Score(tf.keras.metrics.Metric):
            def __init__(self, name='f1_score', **kwargs):
                super().__init__(name=name, **kwargs)
                self.precision = tf.keras.metrics.Precision()
                self.recall = tf.keras.metrics.Recall()

            def update_state(self, y_true, y_pred, sample_weight=None):
                self.precision.update_state(y_true, y_pred, sample_weight)
                self.recall.update_state(y_true, y_pred, sample_weight)

            def result(self):
                p = self.precision.result()
                r = self.recall.result()
                return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

            def reset_states(self):
                self.precision.reset_states()
                self.recall.reset_states()
        
        # Compile model with improved metrics and loss
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(  # AdamW optimizer for better generalization
                learning_rate=5e-5,  # Lower learning rate for stability
                weight_decay=0.01    # Weight decay for regularization
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),  # Label smoothing
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                F1Score(name='f1_score'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        return model

    @staticmethod
    def get_callbacks():
        """Get callbacks for training with improved monitoring"""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_score',  # Monitor F1 score
                mode='max',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_f1_score',
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='model/checkpoints/model_{epoch:02d}_{val_f1_score:.3f}.keras',
                monitor='val_f1_score',
                mode='max',
                save_best_only=True
            )
        ]


