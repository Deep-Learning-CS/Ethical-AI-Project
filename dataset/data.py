import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class AugmentedTextDataset(Dataset):
    def __init__(self, parquet_path, text_column='content', label_column='label', max_length=512):
        self.data = pd.read_parquet(parquet_path)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(
            str(row[self.text_column]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels' : torch.tensor(row[self.label_column], dtype=torch.long)
        }


# Add this new class for multilabel classification
class ManipulationDataset(Dataset):
    """Dataset for manipulation technique classification"""
    
    TECHNIQUE_COLUMNS = [
        'loaded_language', 'glittering_generalities', 'euphoria', 
        'appeal_to_fear', 'fud', 'bandwagon', 'thought_terminating_cliche',
        'whataboutism', 'cherry_picking', 'straw_man'
    ]
    
    def __init__(self, file_path, max_length=512):
        """
        Initialize the dataset
        
        Args:
            file_path: Path to parquet file with processed data
            max_length: Maximum sequence length
        """
        self.df = pd.read_parquet(file_path)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['chunk_text'] if 'chunk_text' in row else row['content']
        
        # Get multilabel targets - this assumes your dataframe has columns for each technique
        labels = torch.zeros(10)
        for i, col in enumerate(self.TECHNIQUE_COLUMNS):
            if col in row and row[col] == 1:
                labels[i] = 1
        
        # Tokenize text
        encoding = self.tokenizer(
            str(text),
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding['labels'] = labels
        
        return encoding