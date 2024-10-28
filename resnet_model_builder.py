from tensorflow.keras import layers, models, applications
import tensorflow as tf

class ModelBuilder:
    """Handles model creation and compilation"""
    
    @staticmethod
    def create_model(input_shape):
        """Create and compile the model with improved architecture"""
        # Create base model
        base_model = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze most layers but unfreeze last 30
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Create model with improved architecture
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(2, activation='softmax')
        ])
        
        # Compile model with fixed learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model