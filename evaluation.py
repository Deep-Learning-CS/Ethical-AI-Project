import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(predictions, labels):
    """
    Compute evaluation metrics for multilabel classification
    
    Args:
        predictions: Model predictions (after sigmoid, thresholded at 0.5)
        labels: True labels
    
    Returns:
        Dictionary with metrics
    """
    # Convert tensors to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Binary predictions (threshold at 0.5)
    binary_preds = (predictions > 0.5).astype(int)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, binary_preds, average='macro'
    )
    
    # Calculate per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        labels, binary_preds, average=None
    )
    
    # Calculate accuracy - less meaningful for multilabel but still reported
    accuracy = accuracy_score(labels, binary_preds)
    
    # Calculate Hamming loss (fraction of labels incorrectly predicted)
    hamming_loss = np.mean(binary_preds != labels)
    
    # Example-based metrics
    exact_match = np.all(binary_preds == labels, axis=1).mean()
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hamming_loss': hamming_loss,
        'exact_match': exact_match,
        'per_class_f1': per_class_f1.tolist()
    }
    
    return metrics