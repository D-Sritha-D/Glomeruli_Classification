import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
from IPython.display import display, HTML

class DataAnalyzer:
    """Interactive data analyzer for glomeruli dataset with notebook display"""
    
    def __init__(self, annotations_df, results_dir):
        """
        Initialize data analyzer
        
        Args:
            annotations_df: DataFrame containing image annotations
            results_dir: Directory to save analysis results
        """
        self.df = annotations_df
        self.results_dir = results_dir
        self.data_dir = 'data'  # Base directory for images
        self.gs_dir = os.path.join(self.data_dir, 'globally_sclerotic_glomeruli')
        self.non_gs_dir = os.path.join(self.data_dir, 'non_globally_sclerotic_glomeruli')
        
        # Collect image paths
        self.gs_paths = [os.path.join(self.gs_dir, f) for f in os.listdir(self.gs_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.non_gs_paths = [os.path.join(self.non_gs_dir, f) for f in os.listdir(self.non_gs_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def analyze_data(self):
        """Perform comprehensive data analysis with interactive display"""
        self._display_section_header("Dataset Overview")
        self._display_basic_stats()
        
        self._display_section_header("Class Distribution Analysis")
        self._plot_class_distribution()
        
        self._display_section_header("Image Properties Analysis")
        self._analyze_image_properties()
        
        self._display_section_header("Sample Images")
        self._plot_sample_images()
        
        self._display_section_header("Pixel Intensity Analysis")
        self._analyze_pixel_intensities()
    
    def _display_section_header(self, title):
        """Display formatted section header"""
        display(HTML(f"<h2>{title}</h2>"))
    
    def _display_basic_stats(self):
        """Display basic dataset statistics"""
        display(HTML("<h3>Dataset Statistics</h3>"))
        display(self.df.describe())
        
        display(HTML("<h3>Category Distribution</h3>"))
        display(self.df['ground truth'].value_counts())
    
    def _plot_class_distribution(self):
        """Plot class distribution"""
        plt.figure(figsize=(10, 6))
        counts = [len(self.non_gs_paths), len(self.gs_paths)]
        plt.bar(['Non-GS', 'GS'], counts, color=['skyblue', 'salmon'])
        plt.title('Class Distribution in Dataset')
        plt.ylabel('Number of Images')
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            plt.text(i, count, str(count), ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Display statistics as formatted text
        display(HTML(f"""
        <div style='margin: 20px'>
            <h4>Class Distribution Statistics:</h4>
            <ul>
                <li>Globally Sclerotic: {len(self.gs_paths)} images</li>
                <li>Non-Globally Sclerotic: {len(self.non_gs_paths)} images</li>
                <li>Ratio (GS:Non-GS): 1:{len(self.non_gs_paths)/len(self.gs_paths):.2f}</li>
            </ul>
        </div>
        """))
    
    def _analyze_image_properties(self):
        """Analyze image properties with interactive display"""
        properties = []
        
        for path_list, class_name in [(self.gs_paths, 'GS'), (self.non_gs_paths, 'Non-GS')]:
            for path in path_list:
                img = Image.open(path)
                properties.append({
                    'class': class_name,
                    'width': img.size[0],
                    'height': img.size[1],
                    'channels': len(img.getbands()),
                    'format': img.format,
                    'size_kb': os.path.getsize(path) / 1024
                })
        
        df = pd.DataFrame(properties)
        
        # Plot size distributions
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(x='class', y='width', data=df)
        plt.title('Image Width Distribution')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x='class', y='height', data=df)
        plt.title('Image Height Distribution')
        
        plt.tight_layout()
        plt.show()
        
        # Display summary statistics
        summary = df.groupby('class').agg({
            'width': ['mean', 'min', 'max'],
            'height': ['mean', 'min', 'max'],
            'size_kb': ['mean', 'min', 'max'],
            'channels': 'first'
        })
        
        display(HTML("<h4>Image Properties Summary:</h4>"))
        display(summary.style.format("{:.2f}"))
    
    def _plot_sample_images(self, num_samples=5):
        """Display sample images from each class"""
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        
        # Plot GS samples
        for i, path in enumerate(np.random.choice(self.gs_paths, num_samples)):
            img = plt.imread(path)
            axes[0, i].imshow(img)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Globally Sclerotic\nSamples', pad=10)
        
        # Plot Non-GS samples
        for i, path in enumerate(np.random.choice(self.non_gs_paths, num_samples)):
            img = plt.imread(path)
            axes[1, i].imshow(img)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Non-Globally Sclerotic\nSamples', pad=10)
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_pixel_intensities(self, num_samples=100):
        """Analyze pixel intensity distributions with interactive display"""
        # Sample and calculate intensities
        gs_intensities = []
        non_gs_intensities = []
        
        for path in np.random.choice(self.gs_paths, num_samples):
            img = np.array(Image.open(path).convert('RGB'))
            gs_intensities.append(img.mean())
        
        for path in np.random.choice(self.non_gs_paths, num_samples):
            img = np.array(Image.open(path).convert('RGB'))
            non_gs_intensities.append(img.mean())
        
        # Plot distributions
        plt.figure(figsize=(10, 6))
        plt.hist(gs_intensities, alpha=0.5, label='GS', bins=30)
        plt.hist(non_gs_intensities, alpha=0.5, label='Non-GS', bins=30)
        plt.title('Distribution of Mean Pixel Intensities')
        plt.xlabel('Mean Pixel Intensity')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Display statistics
        display(HTML(f"""
        <div style='margin: 20px'>
            <h4>Pixel Intensity Statistics:</h4>
            <ul>
                <li>GS Mean: {np.mean(gs_intensities):.2f} ± {np.std(gs_intensities):.2f}</li>
                <li>Non-GS Mean: {np.mean(non_gs_intensities):.2f} ± {np.std(non_gs_intensities):.2f}</li>
            </ul>
        </div>
        """))
