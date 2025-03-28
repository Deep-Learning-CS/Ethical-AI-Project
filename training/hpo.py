import optuna
from train import main as train_main
from dataset import PropagandaDataset
import torch
from torch.utils.data import DataLoader

def objective(trial):
    # Suggest hyperparameters
    params = {
        'lr': trial.suggest_float('lr', 1e-6, 1e-4, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
        'grad_accum_steps': trial.suggest_int('grad_accum_steps', 2, 4)
    }
    
    # Initialize minimal training for HPO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PropagandaDataset("checkpoints/text_classification/step5_context_optimization.parquet")
    train_loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    
    model = HierarchicalAttentionModel(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])
    
    # Short training run
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx > 10:  # Only use 10 batches for HPO
            break
            
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=inputs['input_ids'], 
                      attention_mask=inputs['attention_mask'])
        loss = torch.nn.functional.cross_entropy(outputs['logits'], inputs['labels'])
        
        total_loss += loss.item()
    
    return total_loss / min(10, len(train_loader))

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)  # Reduced for testing
    print("Best params:", study.best_params)