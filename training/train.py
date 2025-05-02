import logging
import torch
import sys
import os
from torch.utils.data import DataLoader

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import your modules
from dataset.data import AugmentedTextDataset
from models.hierarchical_model import HierarchicalAttentionModel

import pandas as pd
df = pd.read_parquet("augmented_text_classification_20250327_161754.parquet")
print(df.columns.tolist())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load your augmented data
    train_dataset = AugmentedTextDataset(
        "augmented_text_classification_20250327_161754.parquet",  # Your augmented file
        text_column='content',
        label_column='manipulative'  # Change if your label column has different name
    )
    
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    logger.info(f"Loaded dataset with {len(train_dataset)} samples")

    # Model setup
    model = HierarchicalAttentionModel(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(3):  # Example: 3 epochs
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            # Extract and ensure logits have shape [batch_size, num_classes]
            logits = outputs['logits']
            print(f"Logits shape: {logits.shape}, Labels shape: {inputs['labels'].shape}")

            # For multi-label, ensure labels are float for BCEWithLogitsLoss
            labels = inputs['labels'].float()

            # Make sure logits are 2D [batch_size, num_classes]
            if logits.dim() > 2:
                logits = logits.view(logits.size(0), -1)

            # Now try the loss calculation
            labels = inputs['labels'].float()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                predictions = (outputs['logits'] > 0.5).float()
                correct_predictions = (predictions == inputs['labels']).float()
                accuracy = correct_predictions.sum() / correct_predictions.numel()
                logger.info(f"Batch accuracy: {accuracy:.4f}")
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        logger.info(f"Epoch {epoch+1} completed | Avg Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    main()