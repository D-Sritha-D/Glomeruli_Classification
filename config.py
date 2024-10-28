import os

class DatasetConfig:
    """Configuration settings for the dataset and training"""
    PATHS = {
        'base': 'data',
        'results': 'results',
        'csv': os.path.join('data', 'public.csv')
    }
    
    TRAINING = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 25,
        'image_size': (224, 224)
    }
    
    LABELS = {
        0: "non_globally_sclerotic_glomeruli",
        1: "globally_sclerotic_glomeruli"
    }