import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import random
from tqdm import tqdm

class ContextualWordReplacer:
    """
    A class for performing contextual word replacement using masked language models.
    This augmentation technique replaces words with contextually appropriate alternatives
    while preserving the original meaning of the text.
    """
    
    def __init__(self, model_name=None):
        """
        Initialize the ContextualWordReplacer with specified language models.
        
        Parameters:
        -----------
        model_name: str or None
            The name of the pre-trained model to use. If None, will use multilingual BERT.
        """
        # Default to multilingual BERT for Ukrainian and Russian
        if model_name is None:
            model_name = "bert-base-multilingual-cased"
            
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        device = None

        # Set device - use GPU if available
        if device is None:
            if torch.backends.mps.is_available():  # Check for MPS (Mac M1)
                self.device = torch.device("mps")
            elif torch.cuda.is_available():  # Check for CUDA (Windows/Linux with GPU)
                self.device = torch.device("cuda")
            else:  # Fallback to CPU
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Special token IDs
        self.mask_token_id = self.tokenizer.mask_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        
        # Maximum sequence length for the model (BERT is 512)
        self.max_length = 512 - 2  # -2 for [CLS] and [SEP]
        
        print(f"Model loaded. Using device: {self.device}")
        
    def _process_with_sliding_window(self, text, replace_ratio=0.15, num_predictions=5, temperature=1.0):
        """
        Process long text using a sliding window approach.
        
        Parameters:
        -----------
        text: str
            The text to augment
        replace_ratio: float
            Proportion of non-stopwords to replace
        num_predictions: int
            Number of alternatives to consider
        temperature: float
            Controls randomness (higher = more random)
            
        Returns:
        --------
        Augmented text
        """
        # Tokenize the full text
        all_tokens = self.tokenizer.tokenize(text)
        
        # If text is within the limit, process directly
        if len(all_tokens) <= self.max_length:
            return self._augment_tokens(all_tokens, replace_ratio, num_predictions, temperature)
        
        # For longer texts, use a sliding window approach
        window_size = self.max_length
        stride = window_size // 2  # 50% overlap
        
        # Process each window
        augmented_windows = []
        prev_end = 0
        
        for start in range(0, len(all_tokens), stride):
            end = min(start + window_size, len(all_tokens))
            
            # Get window tokens
            window_tokens = all_tokens[start:end]
            
            # Only augment non-overlapping or first appearance of overlapping tokens
            if start == 0:
                mask_range = list(range(len(window_tokens)))
            else:
                # Skip the overlapping part with the previous window
                overlap = start - prev_end
                if overlap < 0:
                    # Skip first -overlap tokens which were already processed
                    mask_range = list(range(-overlap, len(window_tokens)))
                else:
                    # No overlap, process all tokens
                    mask_range = list(range(len(window_tokens)))
            
            # Augment this window
            augmented_window = self._augment_tokens(
                window_tokens, 
                replace_ratio, 
                num_predictions, 
                temperature,
                maskable_indices=mask_range
            )
            
            # Add to results
            if start > 0 and prev_end > start:
                # There's an overlap, only add the non-overlapping part
                non_overlap_start = prev_end - start
                augmented_windows.append(augmented_window[non_overlap_start:])
            else:
                augmented_windows.append(augmented_window)
            
            prev_end = end
        
        # Combine augmented windows
        return self.tokenizer.convert_tokens_to_string(sum(augmented_windows, []))
    
    def _augment_tokens(self, tokens, replace_ratio=0.15, num_predictions=5, temperature=1.0, maskable_indices=None):
        """
        Augment a list of tokens by replacing some with contextual alternatives.
        
        Parameters:
        -----------
        tokens: list
            List of tokens to augment
        replace_ratio: float
            Proportion of non-stopwords to replace
        num_predictions: int
            Number of alternatives to consider
        temperature: float
            Controls randomness (higher = more random)
        maskable_indices: list or None
            Indices that can be masked. If None, all indices are considered
            
        Returns:
        --------
        List of augmented tokens
        """
        # Make a copy of tokens to avoid modifying the original
        augmented_tokens = tokens.copy()
        
        # Find replaceable tokens
        replaceable_indices = []
        for i, token in enumerate(tokens):
            # Skip indices that are not maskable
            if maskable_indices is not None and i not in maskable_indices:
                continue
                
            # Skip special tokens and very short tokens
            if (not token.startswith('[') and 
                not token.endswith(']') and 
                not token.startswith('##') and 
                len(token) > 2):
                replaceable_indices.append(i)
        
        # Determine how many tokens to replace
        num_to_replace = max(1, int(len(replaceable_indices) * replace_ratio))
        
        if num_to_replace == 0 or not replaceable_indices:
            return augmented_tokens
        
        # Randomly select indices to replace
        indices_to_replace = random.sample(replaceable_indices, min(num_to_replace, len(replaceable_indices)))
        
        # Get replacements
        replacements = self._get_replacements(augmented_tokens, indices_to_replace, num_predictions, temperature)
        
        # Apply replacements in reverse order to avoid index shifting
        for idx, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
            augmented_tokens[idx] = replacement
        
        return augmented_tokens
    
    def _get_replacements(self, tokens, mask_indices, num_predictions=5, temperature=1.0):
        """
        Generate contextual replacements for specified token indices.
        
        Parameters:
        -----------
        tokens: list
            The tokens to augment
        mask_indices: list
            List of token indices to replace
        num_predictions: int
            Number of alternatives to consider
        temperature: float
            Controls randomness (higher = more random)
            
        Returns:
        --------
        List of (index, replacement_token) tuples
        """
        replacements = []
        
        for idx in mask_indices:
            if idx >= len(tokens):
                continue
                
            # Save the original token
            original_token = tokens[idx]
            
            # Skip special tokens
            if original_token.startswith('[') and original_token.endswith(']'):
                continue
                
            # Replace with [MASK]
            tokens[idx] = self.tokenizer.mask_token
            
            # Convert to tensor
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            # Add special tokens
            input_ids = [self.cls_token_id] + input_ids + [self.sep_token_id]
            mask_idx = idx + 1  # +1 to account for [CLS]
            
            input_ids = torch.tensor([input_ids]).to(self.device)
            
            try:
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    predictions = outputs.logits
                
                # Get predicted tokens for the masked position
                mask_predictions = predictions[0, mask_idx].cpu()
                
                # Apply temperature scaling
                if temperature != 1.0:
                    mask_predictions = mask_predictions / temperature
                    
                # Get top predictions, excluding the original token
                original_id = self.tokenizer.convert_tokens_to_ids([original_token])[0]
                
                # Apply softmax
                probs = torch.nn.functional.softmax(mask_predictions, dim=-1)
                
                # Get top tokens, excluding the original
                top_ids = torch.topk(probs, num_predictions + 5).indices.tolist()
                top_ids = [token_id for token_id in top_ids if token_id != original_id][:num_predictions]
                
                # Convert to tokens
                replacement_tokens = self.tokenizer.convert_ids_to_tokens(top_ids)
                
                # Choose a random replacement
                if replacement_tokens:
                    # Filter out subword tokens (starting with ##)
                    replacement_tokens = [t for t in replacement_tokens if not t.startswith('##')]
                    if replacement_tokens:
                        replacement = random.choice(replacement_tokens)
                        replacements.append((idx, replacement))
            except Exception as e:
                print(f"Error while getting replacements for index {idx}: {str(e)}")
            
            # Restore the original token for the next iteration
            tokens[idx] = original_token
            
        return replacements
    
    def augment_text(self, text, replace_ratio=0.15, num_predictions=5, temperature=1.0):
        """
        Augment the text by replacing some words with contextual alternatives.
        
        Parameters:
        -----------
        text: str
            The text to augment
        replace_ratio: float
            Proportion of non-stopwords to replace
        num_predictions: int
            Number of alternatives to consider
        temperature: float
            Controls randomness (higher = more random)
            
        Returns:
        --------
        Augmented text
        """
        if not isinstance(text, str) or not text.strip():
            return text
        
        # Use sliding window approach for long texts
        return self._process_with_sliding_window(text, replace_ratio, num_predictions, temperature)
    
    def augment_dataframe(self, df, text_column='content', lang_column='lang', 
                         replace_ratio=0.15, temperature=1.0, num_augmentations=1):
        """
        Augment texts in a dataframe using contextual word replacement.
        
        Parameters:
        -----------
        df: pandas DataFrame
            Dataset containing texts to augment
        text_column: str
            Column name containing the text
        lang_column: str
            Column name indicating language
        replace_ratio: float
            Proportion of non-stopwords to replace
        temperature: float
            Controls randomness (higher = more random)
        num_augmentations: int
            Number of augmented versions to create per text
            
        Returns:
        --------
        DataFrame with augmented texts
        """
        augmented_rows = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting texts"):
            original_text = row[text_column]
            
            # Skip if text is empty
            if not isinstance(original_text, str) or not original_text.strip():
                continue
                
            # Generate augmentations
            for i in range(num_augmentations):
                try:
                    new_row = row.copy()
                    new_row[text_column] = self.augment_text(
                        original_text, 
                        replace_ratio=replace_ratio, 
                        temperature=temperature
                    )
                    new_row['augmentation_type'] = 'contextual_word_replacement'
                    new_row['augmentation_id'] = i + 1
                    augmented_rows.append(new_row)
                except Exception as e:
                    print(f"Error augmenting text at index {idx}: {str(e)}")
                    # Skip this augmentation but continue with others
                    continue
                
        # Create new dataframe with augmented texts
        augmented_df = pd.DataFrame(augmented_rows)
        
        return augmented_df

# Example usage
if __name__ == "__main__":
    # Example usage
    from data_read import return_textcl
    
    # Get sample data
    df = return_textcl()
    df = df.head(5)  # Just use a few samples for demonstration
    
    # Initialize the augmenter
    augmenter = ContextualWordReplacer()
    
    # Generate augmentations
    augmented_df = augmenter.augment_dataframe(df, num_augmentations=2)
    
    # Show sample results
    for i, (orig, aug) in enumerate(zip(df['content'].iloc[:2], augmented_df['content'].iloc[:4])):
        if i % 2 == 0:
            print(f"\nOriginal: {orig}")
        else:
            print(f"Augmented: {aug}")
            print("-" * 50)