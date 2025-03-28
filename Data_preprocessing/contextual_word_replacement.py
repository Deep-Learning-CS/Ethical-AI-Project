import torch
import random
import pandas as pd
from tqdm import tqdm
from functools import lru_cache
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import DataLoader
from typing import List, Tuple

class CachedTokenizer:
    """Enhanced tokenizer with attribute pass-through"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Proxy important attributes
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        
    @lru_cache(maxsize=5000)
    def cached_tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)
        
    def tokenize_batch(self, texts: List[str], max_length: int) -> dict:
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )
        
    def __getattr__(self, name):
        """Pass through any undefined attributes to the base tokenizer"""
        return getattr(self.tokenizer, name)

class ContextualWordReplacer:
    def __init__(self, model_name=None, batch_size=8, device=None):
        if model_name is None:
            model_name = "distilbert-base-multilingual-cased"
            
        print(f"Loading model: {model_name}")
        self.tokenizer = CachedTokenizer(AutoTokenizer.from_pretrained(model_name))
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        # Device configuration
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else 
                "mps" if torch.backends.mps.is_available() else 
                "cpu"
            )
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Configuration
        self.batch_size = batch_size
        self.max_length = 450
        self.replace_ratio = 0.15  # Default value
        self.temperature = 1.0     # Default value
        self.num_predictions = 5
        
        print(f"Model loaded. Using device: {self.device}")

    def _get_replacements(self, tokens: List[str], input_ids: torch.Tensor, 
                    logits: torch.Tensor) -> List[Tuple[int, str]]:
        """Get contextual replacements for tokens"""
        replacements = []
        replaceable_indices = [
            i for i, token in enumerate(tokens)
            if token not in {'[CLS]', '[SEP]', '[MASK]'} 
            and not token.startswith('##')
            and len(token) > 2
        ]
        
        if not replaceable_indices:
            return replacements
            
        num_to_replace = max(1, int(len(replaceable_indices) * self.replace_ratio))
        replace_indices = random.sample(replaceable_indices, 
                                    min(num_to_replace, len(replaceable_indices)))
        
        # Apply temperature scaling
        scaled_logits = logits[replace_indices] / self.temperature
        probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
        top_preds = torch.topk(probs, k=self.num_predictions, dim=-1)
        
        for idx, (prob, pred) in zip(replace_indices, 
                                    zip(top_preds.values, top_preds.indices)):
            new_token = self.tokenizer.convert_ids_to_tokens([pred[0].item()])[0]
            if not new_token.startswith('##'):
                replacements.append((idx, new_token))
                
        return replacements

    def _process_with_sliding_window(self, text: str) -> str:
        try:
            all_tokens = self.tokenizer.cached_tokenize(text)
            if len(all_tokens) <= self.max_length - 2:
                return self._augment_tokens(all_tokens)
                
            window_size = self.max_length - 2
            stride = window_size // 2
            augmented_windows = []
            
            for start in range(0, len(all_tokens), stride):
                end = min(start + window_size, len(all_tokens))
                window_tokens = all_tokens[start:end]
                
                # Process on CPU to avoid MPS issues
                input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(window_tokens)]).to('cpu')
                with torch.no_grad():
                    logits = self.model(input_ids.to(self.device)).logits[0].cpu()
                    
                replacements = self._get_replacements(window_tokens, input_ids[0], logits)
                
                new_tokens = window_tokens.copy()
                for idx, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
                    new_tokens[idx] = replacement
                    
                augmented_windows.append(self.tokenizer.convert_tokens_to_string(new_tokens))
                
            return " ".join(augmented_windows)
        except Exception as e:
            print(f"Sliding window error: {str(e)}")
            return text

    def _augment_tokens(self, tokens: List[str]) -> str:
        """Augment a list of tokens"""
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        inputs = torch.tensor([input_ids]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs.logits[0]
            
        replacements = self._get_replacements(tokens, inputs[0], logits)
        
        # Apply replacements in reverse order
        new_tokens = tokens.copy()
        for idx, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
            new_tokens[idx] = replacement
            
        return self.tokenizer.convert_tokens_to_string(new_tokens)

    def augment_batch(self, texts: List[str], replace_ratio: float = 0.15,
                 temperature: float = 1.0) -> List[str]:
        """Updated batch processing with better length handling"""
        augmented_texts = []
        
        for text in texts:
            try:
                # Skip empty texts
                if not isinstance(text, str) or not text.strip():
                    augmented_texts.append(text)
                    continue
                    
                # Check token length first
                tokens = self.tokenizer.cached_tokenize(text)
                if len(tokens) > self.max_length - 2:  # Account for [CLS] and [SEP]
                    # Use sliding window for long texts
                    augmented = self._process_with_sliding_window(text)
                else:
                    # Process normally for short texts
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    inputs = torch.tensor([input_ids]).to('cpu')  # Force CPU for MPS compatibility
                    
                    with torch.no_grad():
                        outputs = self.model(inputs.to(self.device))
                        logits = outputs.logits[0].cpu()  # Bring back to CPU for processing
                        
                    replacements = self._get_replacements(tokens, inputs[0], logits)
                    
                    # Apply replacements
                    new_tokens = tokens.copy()
                    for idx, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
                        new_tokens[idx] = replacement
                        
                    augmented = self.tokenizer.convert_tokens_to_string(new_tokens)
                    
                augmented_texts.append(augmented)
                
            except Exception as e:
                print(f"Error processing text: {str(e)}")
                augmented_texts.append(text)  # Fallback to original text
                
        return augmented_texts


    def augment_dataframe(self, df: pd.DataFrame, text_column: str = 'content',
                     lang_column: str = None, replace_ratio: float = 0.15,
                     temperature: float = 1.0, num_augmentations: int = 1) -> pd.DataFrame:
        """Updated to include all parameters"""
        augmented_rows = []
        
        for _ in range(num_augmentations):
            texts = df[text_column].tolist()
            augmented_texts = self.augment_batch(texts, replace_ratio, temperature)
            
            for orig_row, aug_text in zip(df.to_dict('records'), augmented_texts):
                new_row = orig_row.copy()
                new_row[text_column] = aug_text
                new_row['augmentation_type'] = 'contextual_replacement'
                if lang_column:  # Preserve language if column exists
                    new_row[lang_column] = orig_row[lang_column]
                augmented_rows.append(new_row)
                
        return pd.DataFrame(augmented_rows)

    def __del__(self):
            """Cleanup CUDA cache"""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()