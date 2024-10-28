import numpy as np
import tensorflow as tf
from PIL import Image
import os
from config import DatasetConfig

class GlomeruliDataGenerator(tf.keras.utils.Sequence):
    """Data generator for glomeruli images"""
    
    def __init__(self, annotations_df, root_dir, batch_size=32, image_size=(224, 224), shuffle=True):
        self.annotations = annotations_df
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.annotations))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.annotations) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.zeros((len(batch_indexes), *self.image_size, 3))
        batch_y = np.zeros((len(batch_indexes), 2))
        
        for i, idx in enumerate(batch_indexes):
            img_label = self.annotations.iloc[idx, 1]
            subdir = DatasetConfig.LABELS[img_label]
            img_path = os.path.join(self.root_dir, subdir, self.annotations.iloc[idx, 0])
            
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.image_size)
                img_array = np.array(img) / 255.0
                batch_x[i] = img_array
                batch_y[i] = tf.keras.utils.to_categorical(img_label, num_classes=2)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

