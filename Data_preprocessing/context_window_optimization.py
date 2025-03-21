import pandas as pd
from transformers import AutoTokenizer
import numpy as np

# Load your dataset
def optimize_context_windows(df, text_column='content', max_length=512, overlap=100):
    """
    Optimize context windows by splitting text into overlapping chunks that fit within
    transformer model context limits.
    
    Parameters:
    -----------
    df: pandas DataFrame
        Dataset containing text to process
    text_column: str
        Column name containing the text
    max_length: int
        Maximum token length for the transformer model
    overlap: int
        Number of tokens to overlap between consecutive chunks
    
    Returns:
    --------
    DataFrame with optimized context windows
    """
    # Load a tokenizer (using multilingual BERT for Ukrainian and Russian)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    # Create a new dataframe to store the chunks
    chunks = []
    
    for idx, row in df.iterrows():
        text = row[text_column]
        
        # Skip empty texts
        if not isinstance(text, str) or text.strip() == '':
            continue
            
        # Tokenize the text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # If the text fits in one window, keep it as is
        if len(tokens) <= max_length - 2:  # -2 for [CLS] and [SEP]
            new_row = row.copy()
            new_row['chunk_id'] = 0
            new_row['chunk_text'] = text
            new_row['token_length'] = len(tokens)
            chunks.append(new_row)
        else:
            # Split into overlapping chunks
            chunk_id = 0
            start_idx = 0
            
            while start_idx < len(tokens):
                end_idx = min(start_idx + max_length - 2, len(tokens))
                
                # Get the chunk tokens and convert back to text
                chunk_tokens = tokens[start_idx:end_idx]
                chunk_text = tokenizer.decode(chunk_tokens)
                
                # Create a new row for this chunk
                new_row = row.copy()
                new_row['chunk_id'] = chunk_id
                new_row['chunk_text'] = chunk_text
                new_row['token_length'] = len(chunk_tokens)
                new_row['is_continuation'] = (start_idx > 0)
                chunks.append(new_row)
                
                # Move to next chunk with overlap
                start_idx = end_idx - overlap
                chunk_id += 1
                
                # Break if we've reached the end
                if end_idx == len(tokens):
                    break
    
    # Convert to DataFrame
    chunks_df = pd.DataFrame(chunks)
    print(f"Original texts: {len(df)}, Optimized chunks: {len(chunks_df)}")
    
    return chunks_df

# Example usage
if __name__ == "__main__":
    # Assuming df contains your preprocessed data
    from data_read import return_textcl
    df = return_textcl()
    
    # Apply context window optimization
    optimized_df = optimize_context_windows(df)
    
    # Show sample results
    print("\nSample of optimized context windows:")
    print(optimized_df[['id', 'chunk_id', 'token_length', 'chunk_text']].head(3))