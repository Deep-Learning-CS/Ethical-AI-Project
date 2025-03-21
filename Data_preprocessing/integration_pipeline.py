import sys
import os
import pandas as pd
import time
import json
import numpy as np
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_read import spandf, techclass  # Import the dataframes directly

# Import the preprocessing modules we created
from context_window_optimization import optimize_context_windows
from entity_recognition import perform_entity_recognition
from semantic_role_labeling import apply_semantic_role_labeling
from model_based_cleaning import ModelBasedCleaner
from contextual_word_replacement import ContextualWordReplacer

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

def run_advanced_preprocessing_pipeline(data_source='text_classification', sample_size=None, 
                                       clean_data=True, augment_data=True, augmentations_per_text=1):
    """
    Run the complete advanced preprocessing pipeline in a single pass.
    
    Parameters:
    -----------
    data_source: str
        Either 'text_classification' or 'span_detection'
    sample_size: int or None
        If specified, only process a sample of the data
    clean_data: bool
        Whether to apply model-based cleaning
    augment_data: bool
        Whether to apply contextual word replacement augmentation
    augmentations_per_text: int
        Number of augmented versions to create per text
        
    Returns:
    --------
    Processed DataFrame
    """
    print(f"Starting advanced preprocessing pipeline for {data_source} data")
    start_time = time.time()
    
    # Load appropriate dataset
    if data_source == 'text_classification':
        df = techclass.copy()  # Use dataframe directly and make a copy
    elif data_source == 'span_detection':
        df = spandf.copy()     # Use dataframe directly and make a copy
    else:
        raise ValueError("data_source must be 'text_classification' or 'span_detection'")
    
    # Take a sample if requested
    if sample_size is not None:
        df = df.sample(min(sample_size, len(df)))
        print(f"Using a sample of {len(df)} records")
    
    # Step 1: Apply model-based cleaning if requested
    if clean_data:
        print("\nStep 1: Model-Based Cleaning")
        step_start = time.time()
        
        # Initialize the cleaner
        cleaner = ModelBasedCleaner()
        
        # We want to create only one visualization per run
        # The first (and only) time we run the cleaner
        
        # Clean the dataset
        df = cleaner.clean_dataset(
            df, 
            text_column='content', 
            label_column='label' if 'label' in df.columns else None,
            remove_outliers=True, 
            fix_mislabeled=True, 
            filter_low_quality=True,
            visualize_outliers=True  # Only create visualization once
        )
        
        print(f"Completed in {time.time() - step_start:.2f} seconds")
    
    # Step 2: Apply entity recognition and linking
    print("\nStep 2: Entity Recognition and Linking")
    step_start = time.time()
    df = perform_entity_recognition(df)
    print(f"Completed in {time.time() - step_start:.2f} seconds")
    
    # Step 3: Apply semantic role labeling
    print("\nStep 3: Semantic Role Labeling")
    step_start = time.time()
    df = apply_semantic_role_labeling(df)
    print(f"Completed in {time.time() - step_start:.2f} seconds")
    
    # Step 4: Apply contextual word replacement for data augmentation if requested
    augmented_df = None
    if augment_data:
        print("\nStep 4: Contextual Word Replacement (Data Augmentation)")
        step_start = time.time()
        
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
        
        # Combine original and augmented data
        df = pd.concat([df, augmented_df], ignore_index=True)
        print(f"Added {len(augmented_df)} augmented texts")
        print(f"Completed in {time.time() - step_start:.2f} seconds")
    
    # Step 5: Context window optimization
    print("\nStep 5: Context Window Optimization")
    step_start = time.time()
    optimized_df = optimize_context_windows(df)
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
        "processing_time": float(time.time() - start_time)
    }
    
    # Save the summary
    summary_file = f"preprocessing_summary_{data_source}_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"Processing summary saved to {summary_file}")
    print(f"Total processing time: {summary['processing_time']:.2f} seconds")
    
    # Save the results in a structured format (e.g., JSON or CSV)
    structured_output_file = f"structured_results_{data_source}_{timestamp}.json"
    structured_results = []

    for _, row in optimized_df.iterrows():
        result = {
            "id": row['id'],
            "content": row['content'],
            "chunk_text": row['chunk_text'],
            "token_length": row['token_length'],
            "is_augmented": row.get('augmentation_type') == 'contextual_word_replacement' if 'augmentation_type' in row else False
        }
        
        # Add entity recognition results
        if 'entities' in row:
            result.update({
                "entities": row['entities'],
                "entity_types": row['entity_types'],
                "entity_count": row['entity_count'],
                "linked_entities": row['linked_entities']
            })
            
        # Add semantic role labeling results
        if 'semantic_predicates' in row:
            result.update({
                "semantic_predicates": row['semantic_predicates'],
                "semantic_predicate_count": row['semantic_predicate_count']
            })
            
        # Add model-based cleaning results if available
        if clean_data and 'outlier_score' in row:
            result.update({
                "outlier_score": row.get('outlier_score'),
                "model_confidence": row.get('model_confidence'),
                "word_count": row.get('word_count'),
                "noise_ratio": row.get('noise_ratio')
            })
            
        structured_results.append(result)

    with open(structured_output_file, 'w', encoding='utf-8') as f:
        json.dump(structured_results, f, ensure_ascii=False, indent=2)
    
    print(f"Structured results saved to {structured_output_file}")
    
    return optimized_df

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Run advanced preprocessing pipeline')
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
    
    args = parser.parse_args()
    
    # Run the pipeline
    processed_df = run_advanced_preprocessing_pipeline(
        data_source=args.data_source,
        sample_size=args.sample_size,
        clean_data=args.clean_data,
        augment_data=args.augment_data,
        augmentations_per_text=args.augmentations_per_text
    )
    
    # Display sample of results
    print("\nSample of final processed data:")
    print(processed_df.head(2))