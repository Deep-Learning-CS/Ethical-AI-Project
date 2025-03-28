import sys
import os
import pandas as pd
import time
import json
import numpy as np
import traceback
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_read import spandf, techclass  # Import the dataframes directly

# Import the preprocessing modules we created
from context_window_optimization import optimize_context_windows
from entity_recognition import perform_entity_recognition
from semantic_role_labeling import apply_semantic_role_labeling
from model_based_cleaning import ModelBasedCleaner
from contextual_word_replacement import ContextualWordReplacer
from transformers import AutoModel
from custom_attention import PropagandaAttention
import torch

# Helper function to convert NumPy types to Python native types
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def load_checkpoint(checkpoint_dir, step_name):
    """Load a checkpoint if it exists."""
    checkpoint_file = Path(checkpoint_dir) / f"{step_name}.parquet"
    if checkpoint_file.exists():
        try:
            return pd.read_parquet(checkpoint_file)
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_file}: {e}")
            return None
    return None

def save_checkpoint(df, checkpoint_dir, step_name):
    """Save a checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"{step_name}.parquet"
    try:
        df.to_parquet(checkpoint_file)
        print(f"Saved checkpoint for {step_name} to {checkpoint_file}")
    except Exception as e:
        print(f"Error saving checkpoint {checkpoint_file}: {e}")

def run_advanced_preprocessing_pipeline(data_source='text_classification', sample_size=None, 
                                       clean_data=True, augment_data=True, augmentations_per_text=1,
                                       checkpoint_dir="checkpoints", resume=True):
    """
    Run the complete advanced preprocessing pipeline in a single pass with checkpoint support.
    """
    print(f"Starting advanced preprocessing pipeline for {data_source} data")
    start_time = time.time()
    
    # Initialize cleaner (will be used in summary stats)
    cleaner = ModelBasedCleaner() if clean_data else None
    
    # Create checkpoint directory path
    checkpoint_dir = Path(checkpoint_dir) / data_source
    if sample_size is not None:
        checkpoint_dir = checkpoint_dir / f"sample_{sample_size}"
    
    # Load appropriate dataset
    if data_source == 'text_classification':
        df = techclass.copy()  # Use dataframe directly and make a copy
    elif data_source == 'span_detection':
        df = spandf.copy()     # Use dataframe directly and make a copy
    else:
        raise ValueError("data_source must be 'text_classification' or 'span_detection'")
    
    # Take a sample if requested
    if sample_size is not None:
        df = df.sample(min(sample_size, len(df)), random_state=42)  # Fixed random state for reproducibility
        print(f"Using a sample of {len(df)} records")
    
    # Step 1: Apply model-based cleaning if requested
    if clean_data:
        print("\nStep 1: Model-Based Cleaning")
        step_start = time.time()
        
        # Try to load checkpoint
        checkpoint_df = None
        if resume:
            checkpoint_df = load_checkpoint(checkpoint_dir, "step1_cleaning")
        
        if checkpoint_df is not None:
            df = checkpoint_df
            print("Loaded cleaned data from checkpoint")
        else:
            # Clean the dataset using our pre-initialized cleaner
            df = cleaner.clean_dataset(
                df, 
                text_column='content', 
                label_column='label' if 'label' in df.columns else None,
                remove_outliers=True, 
                fix_mislabeled=True, 
                filter_low_quality=True,
                visualize_outliers=True  # Only create visualization once
            )
            
            # Save checkpoint
            save_checkpoint(df, checkpoint_dir, "step1_cleaning")
        
        print(f"Completed in {time.time() - step_start:.2f} seconds")
    
    # Step 2: Apply entity recognition and linking
    print("\nStep 2: Entity Recognition and Linking")
    step_start = time.time()
    
    checkpoint_df = None
    if resume:
        checkpoint_df = load_checkpoint(checkpoint_dir, "step2_entity_recognition")
    
    if checkpoint_df is not None:
        df = checkpoint_df
        print("Loaded entity recognition results from checkpoint")
    else:
        df = perform_entity_recognition(df)
        save_checkpoint(df, checkpoint_dir, "step2_entity_recognition")
    
    print(f"Completed in {time.time() - step_start:.2f} seconds")
    
    # Step 3: Apply semantic role labeling
    print("\nStep 3: Semantic Role Labeling")
    step_start = time.time()
    
    checkpoint_df = None
    if resume:
        checkpoint_df = load_checkpoint(checkpoint_dir, "step3_semantic_labeling")
    
    if checkpoint_df is not None:
        df = checkpoint_df
        print("Loaded semantic role labeling results from checkpoint")
    else:
        df = apply_semantic_role_labeling(df)
        save_checkpoint(df, checkpoint_dir, "step3_semantic_labeling")
    
    print(f"Completed in {time.time() - step_start:.2f} seconds")
    
    # Step 4: Apply contextual word replacement for data augmentation if requested
    augmented_df = None
    if augment_data:
        print("\nStep 4: Contextual Word Replacement (Data Augmentation)")
        step_start = time.time()
        
        checkpoint_df = None
        if resume:
            checkpoint_df = load_checkpoint(checkpoint_dir, "step4_data_augmentation")
        
        if checkpoint_df is not None:
            augmented_df = checkpoint_df
            print("Loaded augmented data from checkpoint")
        else:
            # Initialize the augmenter with our enhanced version
            augmenter = ContextualWordReplacer()
            
            # Generate augmentations
            augmented_df = augmenter.augment_dataframe(
                df, 
                text_column='content', 
                lang_column='lang',
                replace_ratio=0.15, 
                temperature=1.0, 
                num_augmentations=augmentations_per_text
            )
            
            save_checkpoint(augmented_df, checkpoint_dir, "step4_data_augmentation")

        # Combine original and augmented data
        df = pd.concat([df, augmented_df], ignore_index=True)
        print(f"Added {len(augmented_df)} augmented texts")
        print(f"Completed in {time.time() - step_start:.2f} seconds")

    if augment_data:
        print("\nSaving augmented data...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        augmented_file = f"augmented_{data_source}_{timestamp}.parquet"
        augmented_df.to_parquet(augmented_file)
        print(f"Saved augmented data to {augmented_file}")

    print("\nInitializing model with custom attention...")
    model = AutoModel.from_pretrained(
        "distilbert-base-multilingual-cased",
        output_attentions=True  # Enable attention outputs
    )
    
    # Replace the last two layers' attention for DistilBERT
    for layer in model.transformer.layer[-2:]:
        layer.attention = PropagandaAttention(model.config.hidden_size)
    
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Model with custom attention initialized")

    # Step 5: Context window optimization
    print("\nStep 5: Context Window Optimization")
    step_start = time.time()
    
    checkpoint_df = None
    if resume:
        checkpoint_df = load_checkpoint(checkpoint_dir, "step5_context_optimization")
    
    if checkpoint_df is not None:
        optimized_df = checkpoint_df
        print("Loaded context window optimization results from checkpoint")
    else:
        optimized_df = optimize_context_windows(df)
        save_checkpoint(optimized_df, checkpoint_dir, "step5_context_optimization")
    
    print(f"Completed in {time.time() - step_start:.2f} seconds")
    
    # Save the results to a structured output file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"advanced_processed_{data_source}_{timestamp}.parquet"
    optimized_df.to_parquet(output_file)
    print(f"\nSaved advanced preprocessed data to {output_file}")
    
    # Create a summary of the processing
    most_common_types = optimized_df['entity_types'].explode().value_counts().head(5).to_dict()
    most_common_types = convert_to_serializable(most_common_types)
    
    # Create detailed summary including new preprocessing steps
    summary = {
        "original_records": int(len(df) - (0 if augmented_df is None else len(augmented_df))),
        "processed_chunks": int(len(optimized_df)),
        "cleaning_stats": {
            "cleaned_enabled": clean_data,
            "removed_outliers": int(cleaner.outliers_removed) if clean_data else 0,
            "fixed_mislabeled": int(cleaner.mislabeled_fixed) if clean_data else 0,
            "removed_low_quality": int(cleaner.low_quality_removed) if clean_data else 0
        },
        "augmentation_stats": {
            "augmentation_enabled": augment_data,
            "augmented_texts": int(0 if augmented_df is None else len(augmented_df)),
            "augmentations_per_text": augmentations_per_text
        },
        "entity_stats": {
            "total_entities": int(optimized_df['entity_count'].sum()),
            "avg_entities_per_text": float(optimized_df['entity_count'].mean()),
            "most_common_entity_types": most_common_types
        },
        "semantic_role_stats": {
            "total_predicates": int(optimized_df['semantic_predicate_count'].sum()),
            "avg_predicates_per_text": float(optimized_df['semantic_predicate_count'].mean())
        },
        "processing_time": float(time.time() - start_time),
        "checkpoints_used": resume
    }
    
    # Save the summary
    summary_file = f"preprocessing_summary_{data_source}_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"Processing summary saved to {summary_file}")
    print(f"Total processing time: {summary['processing_time']:.2f} seconds")
    
    # Save the results in a structured format (e.g., JSON or CSV)
    

        # Save the results in a structured format (e.g., JSON or CSV)
    structured_output_file = f"structured_results_{data_source}_{timestamp}.json"
    structured_results = []

    for _, row in optimized_df.iterrows():
        result = {
            "id": row['id'],
            "content": row['content'],
            "chunk_text": row['chunk_text'],
            "token_length": int(row['token_length']),  # Convert to native int
            "is_augmented": row.get('augmentation_type') == 'contextual_word_replacement' if 'augmentation_type' in row else False
        }
        
        # Add entity recognition results
        if 'entities' in row:
            result.update({
                "entities": convert_to_serializable(row['entities']),
                "entity_types": convert_to_serializable(row['entity_types']),
                "entity_count": int(row['entity_count']),
                "linked_entities": convert_to_serializable(row['linked_entities'])
            })
            
        # Add semantic role labeling results
        if 'semantic_predicates' in row:
            result.update({
                "semantic_predicates": convert_to_serializable(row['semantic_predicates']),
                "semantic_predicate_count": int(row['semantic_predicate_count'])
            })
            
        # Add model-based cleaning results if available
        if clean_data and 'outlier_score' in row:
            result.update({
                "outlier_score": float(row.get('outlier_score')) if pd.notna(row.get('outlier_score')) else None,
                "model_confidence": float(row.get('model_confidence')) if pd.notna(row.get('model_confidence')) else None,
                "word_count": int(row.get('word_count')) if pd.notna(row.get('word_count')) else None,
                "noise_ratio": float(row.get('noise_ratio')) if pd.notna(row.get('noise_ratio')) else None
            })
            
        structured_results.append(convert_to_serializable(result))

    #with open(structured_output_file, 'w', encoding='utf-8') as f:
    #    json.dump(structured_results, f, ensure_ascii=False, indent=2)
    
    print(f"Structured results saved to {structured_output_file}")
    
    return optimized_df, model

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Run advanced preprocessing pipeline with checkpoints')
    parser.add_argument('--data_source', type=str, default='text_classification',
                        choices=['text_classification', 'span_detection'],
                        help='Data source to process')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of records to sample (None for all)')
    parser.add_argument('--clean_data', action='store_true', default=True,
                        help='Apply model-based cleaning')
    parser.add_argument('--no_clean_data', action='store_false', dest='clean_data',
                        help='Skip model-based cleaning')
    parser.add_argument('--augment_data', action='store_true', default=True,
                        help='Apply contextual word replacement augmentation')
    parser.add_argument('--no_augment_data', action='store_false', dest='augment_data',
                        help='Skip data augmentation')
    parser.add_argument('--augmentations_per_text', type=int, default=1,
                        help='Number of augmented versions to create per text')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                        help='Directory to store/load checkpoints')
    parser.add_argument('--no_resume', action='store_false', dest='resume',
                        help='Disable resuming from checkpoints')
    
    args = parser.parse_args()
    
    # Run the pipeline
    processed_df, model = run_advanced_preprocessing_pipeline(
        data_source=args.data_source,
        sample_size=args.sample_size,
        clean_data=args.clean_data,
        augment_data=args.augment_data,
        augmentations_per_text=args.augmentations_per_text,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume
    )
    
    # Display sample of results
    print("\nSample of final processed data:")
    print(processed_df.head(2))