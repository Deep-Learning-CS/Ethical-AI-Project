import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import umap.umap_ as umap
import torch.nn.functional as F
import os
import seaborn as sns

class ModelBasedCleaner:
    """
    A class for performing model-based data cleaning:
    1. Identifies outliers and potentially mislabeled examples
    2. Detects and handles noisy or corrupted text
    3. Applies consistency checks across the dataset
    """
    
    def __init__(self, model_name=None, device=None):
        """
        Initialize the model-based cleaner with a pretrained model.
        
        Parameters:
        -----------
        model_name: str or None
            Name of the pretrained model to use for feature extraction
        device: str or None
            Device to use ('cuda', 'cpu', etc.)
        """
        # Default to multilingual BERT for Ukrainian and Russian
        if model_name is None:
            model_name = "bert-base-multilingual-cased"
        
        # Determine device
        if device is None:
            if torch.backends.mps.is_available():  # Check for MPS (Mac M1)
                self.device = torch.device("mps")
            elif torch.cuda.is_available():  # Check for CUDA (Windows/Linux with GPU)
                self.device = torch.device("cuda")
            else:  # Fallback to CPU
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded. Using device: {self.device}")
        
        # Initialize statistics counters
        self.outliers_removed = 0
        self.mislabeled_fixed = 0
        self.low_quality_removed = 0
        
        # Set a maximum token length
        self.max_length = 512
        
    def _extract_features(self, texts, batch_size=8):
        """
        Extract semantic features from texts using the model.
        
        Parameters:
        -----------
        texts: list
            List of text strings
        batch_size: int
            Batch size for processing
            
        Returns:
        --------
        Tensor of features
        """
        features = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting features"):
            batch_texts = texts[i:i+batch_size]
            
            # Skip empty texts
            batch_texts = [t if isinstance(t, str) and t.strip() else "" for t in batch_texts]
            
            # Tokenize
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Get [CLS] token embedding from last hidden state
                last_hidden_state = outputs.hidden_states[-1]
                cls_embeddings = last_hidden_state[:, 0, :].cpu().numpy()
                
                features.append(cls_embeddings)
                
        return np.vstack(features) if features else np.array([])
    
    def identify_outliers(self, df, text_column='content', label_column=None, contamination=0.05, visualize=True):
        """
        Identify outliers in the dataset using an Isolation Forest.
        
        Parameters:
        -----------
        df: pandas DataFrame
            Dataset containing text to clean
        text_column: str
            Column name containing the text
        label_column: str or None
            Column name containing labels (if available)
        contamination: float
            Expected proportion of outliers
        visualize: bool
            Whether to generate a visualization
            
        Returns:
        --------
        DataFrame with outlier scores and flags
        """
        # Extract features
        texts = df[text_column].tolist()
        features = self._extract_features(texts)
        
        # If we have labels, process each class separately
        if label_column is not None and label_column in df.columns:
            result_df = df.copy()
            result_df['outlier_score'] = np.nan
            result_df['is_outlier'] = False
            
            # Process each class separately
            for label_value in df[label_column].unique():
                indices = df[df[label_column] == label_value].index
                if len(indices) < 10:  # Skip if too few samples
                    continue
                    
                # Get features for this class
                class_features = features[df.index.isin(indices)]
                
                # Train Isolation Forest
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                outlier_preds = iso_forest.fit_predict(class_features)
                outlier_scores = iso_forest.decision_function(class_features)
                
                # -1 indicates outlier, 1 indicates inlier in isolation forest
                result_df.loc[indices, 'outlier_score'] = outlier_scores
                result_df.loc[indices, 'is_outlier'] = (outlier_preds == -1)
        else:
            # Train Isolation Forest on all data
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_preds = iso_forest.fit_predict(features)
            outlier_scores = iso_forest.decision_function(features)
            
            # Add results to dataframe
            result_df = df.copy()
            result_df['outlier_score'] = outlier_scores
            result_df['is_outlier'] = (outlier_preds == -1)
            
        # Always create visualization when requested
        if visualize:
            outliers = result_df['is_outlier'].values
            labels = result_df[label_column].values if label_column in result_df.columns else None
            
            # Force recreation of visualization even if it exists
            save_path = "outlier_visualization.png"
            if os.path.exists(save_path):
                os.remove(save_path)
                
            # Create enhanced visualization
            self.create_enhanced_visualization(features, labels, outliers, save_path=save_path)
            
        return result_df, features
    
    def detect_mislabeled(self, df, text_column='content', label_column='label', confidence_threshold=0.8):
        """
        Detect potentially mislabeled examples by checking model confidence.
        
        Parameters:
        -----------
        df: pandas DataFrame
            Dataset to check
        text_column: str
            Column containing the text
        label_column: str
            Column containing the labels (0 or 1)
        confidence_threshold: float
            Confidence threshold for flagging samples
            
        Returns:
        --------
        DataFrame with confidence scores and mislabel flags
        """
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataframe")
            
        result_df = df.copy()
        result_df['model_confidence'] = np.nan
        result_df['potentially_mislabeled'] = False
        
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        
        batch_size = 8
        for i in tqdm(range(0, len(texts), batch_size), desc="Checking labels"):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # Skip empty texts
            batch_texts = [t if isinstance(t, str) and t.strip() else "" for t in batch_texts]
            
            # Tokenize
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1).cpu().numpy()
                
                # Calculate confidence for the actual label
                for j, (prob, label) in enumerate(zip(probs, batch_labels)):
                    idx = i + j
                    if idx < len(result_df):
                        # Get confidence for the assigned label
                        label_idx = int(label) if isinstance(label, (int, float)) else 0
                        confidence = prob[label_idx]
                        
                        result_df.loc[idx, 'model_confidence'] = confidence
                        result_df.loc[idx, 'potentially_mislabeled'] = (confidence < confidence_threshold)
        
        return result_df
    
    def check_text_quality(self, df, text_column='content', min_words=3, max_noise_ratio=0.3):
        """
        Check text quality by detecting very short or noisy text.
        
        Parameters:
        -----------
        df: pandas DataFrame
            Dataset to check
        text_column: str
            Column containing the text
        min_words: int
            Minimum number of words required
        max_noise_ratio: float
            Maximum allowed ratio of special chars to total chars
            
        Returns:
        --------
        DataFrame with quality flags
        """
        result_df = df.copy()
        result_df['word_count'] = 0
        result_df['noise_ratio'] = 0.0
        result_df['low_quality_text'] = False
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking text quality"):
            text = row[text_column]
            
            if not isinstance(text, str) or not text.strip():
                result_df.loc[idx, 'low_quality_text'] = True
                continue
                
            # Count words
            words = text.split()
            word_count = len(words)
            result_df.loc[idx, 'word_count'] = word_count
            
            # Calculate noise ratio (special chars, numbers)
            total_chars = len(text)
            noise_chars = sum(1 for c in text if not c.isalpha() and not c.isspace())
            noise_ratio = noise_chars / total_chars if total_chars > 0 else 1.0
            result_df.loc[idx, 'noise_ratio'] = noise_ratio
            
            # Flag low quality text
            if word_count < min_words or noise_ratio > max_noise_ratio:
                result_df.loc[idx, 'low_quality_text'] = True
                
        return result_df
    
    def create_enhanced_visualization(self, features, labels=None, outliers=None, save_path=None):
        """
        Create an enhanced UMAP visualization of the dataset showing outliers.
        
        Parameters:
        -----------
        features: ndarray
            Feature embeddings
        labels: ndarray or None
            Class labels
        outliers: ndarray or None
            Boolean array indicating outliers
        save_path: str or None
            Path to save the plot
        """
        print("Creating UMAP visualization...")
        plt.figure(figsize=(12, 10))
        
        # Set a styling theme
        sns.set_style("whitegrid")
        
        # Reduce to 2D for visualization - use a higher n_neighbors for better global structure
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
        embedding = reducer.fit_transform(features)
        
        # Plot main scatter with transparent points
        if labels is not None:
            # Create a colormap for labels
            unique_labels = np.unique(labels)
            n_labels = len(unique_labels)
            cmap = plt.cm.get_cmap('tab10', n_labels)
            
            # Plot by class with a legend
            for i, label in enumerate(unique_labels):
                idx = (labels == label)
                plt.scatter(
                    embedding[idx, 0], 
                    embedding[idx, 1], 
                    s=60, 
                    c=[cmap(i)], 
                    label=f'Class {label}', 
                    alpha=0.7,
                    edgecolors='none'
                )
        else:
            # Just plot all points
            plt.scatter(
                embedding[:, 0], 
                embedding[:, 1], 
                s=60, 
                c='#1f77b4', 
                alpha=0.7,
                edgecolors='none'
            )
            
        # Highlight outliers with red circles
        if outliers is not None and np.any(outliers):
            plt.scatter(
                embedding[outliers, 0], 
                embedding[outliers, 1], 
                s=100, 
                facecolors='none', 
                edgecolors='red',
                linewidths=2,
                label='Outliers'
            )
            
            # Add a text annotation showing how many outliers
            outlier_count = np.sum(outliers)
            plt.text(
                0.02, 0.02, 
                f"Found {outlier_count} outliers ({outlier_count/len(outliers):.1%})",
                transform=plt.gca().transAxes,
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
            )
                
        plt.title('UMAP Visualization of Dataset with Outliers', fontsize=16)
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.legend(fontsize=10, loc='upper right')
        
        # Add grid and tight layout
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        if save_path:
            print(f"Saving visualization to {save_path}")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # Don't close the figure so it can be displayed in the notebook/interactive environment
        else:
            plt.show()
    
    def visualize_dataset(self, features, labels=None, outliers=None, save_path=None):
        """
        Visualize the dataset in 2D using UMAP for dimensionality reduction.
        This is kept for backward compatibility.
        
        Parameters:
        -----------
        features: ndarray
            Feature embeddings
        labels: ndarray or None
            Class labels
        outliers: ndarray or None
            Boolean array indicating outliers
        save_path: str or None
            Path to save the plot
        """
        # Use the enhanced visualization instead
        return self.create_enhanced_visualization(features, labels, outliers, save_path)
    
    def clean_dataset(self, df, text_column='content', label_column=None,
                     remove_outliers=True, fix_mislabeled=True, filter_low_quality=True,
                     visualize_outliers=True):
        """
        Clean the dataset by removing or fixing problematic samples.
        
        Parameters:
        -----------
        df: pandas DataFrame
            Dataset to clean
        text_column: str
            Column containing the text
        label_column: str or None
            Column containing the labels
        remove_outliers: bool
            Whether to remove outliers
        fix_mislabeled: bool
            Whether to fix mislabeled examples
        filter_low_quality: bool
            Whether to filter low quality text
        visualize_outliers: bool
            Whether to generate an outlier visualization
            
        Returns:
        --------
        Cleaned DataFrame
        """
        # Start with a copy of the original data
        clean_df = df.copy()
        
        # Step 1: Identify outliers
        if remove_outliers:
            print("Identifying outliers...")
            outlier_df, features = self.identify_outliers(
                clean_df, 
                text_column, 
                label_column,
                visualize=visualize_outliers
            )
            
            # Remove outliers
            old_count = len(clean_df)
            clean_df = outlier_df[~outlier_df['is_outlier']].copy()
            outliers_count = old_count - len(clean_df)
            self.outliers_removed += outliers_count
            print(f"Removed {outliers_count} outliers")
        
        # Step 2: Check for mislabeled examples
        if fix_mislabeled and label_column is not None:
            print("Checking for mislabeled examples...")
            mislabel_df = self.detect_mislabeled(clean_df, text_column, label_column)
            
            # Either remove or correct mislabeled examples
            mislabeled_count = mislabel_df['potentially_mislabeled'].sum()
            
            if mislabeled_count > 0:
                # Option 1: Remove them
                old_count = len(clean_df)
                clean_df = mislabel_df[~mislabel_df['potentially_mislabeled']].copy()
                actual_removed = old_count - len(clean_df)
                self.mislabeled_fixed += actual_removed
                print(f"Removed {actual_removed} potentially mislabeled examples")
                
                # Option 2: Correct them (would need a more sophisticated approach in practice)
                # This is just a placeholder for what could be done
                # mislabel_df.loc[mislabel_df['potentially_mislabeled'], label_column] = 1 - mislabel_df.loc[mislabel_df['potentially_mislabeled'], label_column]
        
        # Step 3: Filter low quality text
        if filter_low_quality:
            print("Checking text quality...")
            quality_df = self.check_text_quality(clean_df, text_column)
            
            # Remove low quality samples
            old_count = len(clean_df)
            clean_df = quality_df[~quality_df['low_quality_text']].copy()
            low_quality_count = old_count - len(clean_df)
            self.low_quality_removed += low_quality_count
            print(f"Removed {low_quality_count} low quality samples")
            
        # Generate report
        original_count = len(df)
        cleaned_count = len(clean_df)
        removed_count = original_count - cleaned_count
        
        print(f"\nCleaning Summary:")
        print(f"Original samples: {original_count}")
        print(f"Cleaned samples: {cleaned_count}")
        print(f"Removed samples: {removed_count} ({removed_count/original_count:.1%})")
        
        return clean_df

# Example usage
if __name__ == "__main__":
    # Example usage
    from data_read import return_textcl
    
    # Get sample data
    df = return_textcl()
    
    # Initialize the cleaner
    cleaner = ModelBasedCleaner()
    
    # Clean the dataset
    cleaned_df = cleaner.clean_dataset(df)
    
    # Save the cleaned dataset
    cleaned_df.to_parquet("cleaned_dataset.parquet")