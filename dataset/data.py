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