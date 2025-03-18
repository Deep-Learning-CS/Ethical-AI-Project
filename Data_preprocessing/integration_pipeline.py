import sys
import os
import pandas as pd
import time
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_read import spandf, techclass  # Import the dataframes directly

# Import the preprocessing modules we created
from context_window_optimization import optimize_context_windows
from entity_recognition import perform_entity_recognition
from semantic_role_labeling import apply_semantic_role_labeling

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

def run_advanced_preprocessing_pipeline(data_source='text_classification', sample_size=None):
    """
    Run the complete advanced preprocessing pipeline.
    
    Parameters:
    -----------
    data_source: str
        Either 'text_classification' or 'span_detection'
    sample_size: int or None
        If specified, only process a sample of the data
        
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
    
    # Step 1: Apply entity recognition and linking
    print("\nStep 1: Entity Recognition and Linking")
    step_start = time.time()
    df = perform_entity_recognition(df)
    print(f"Completed in {time.time() - step_start:.2f} seconds")
    
    # Step 2: Apply semantic role labeling
    print("\nStep 2: Semantic Role Labeling")
    step_start = time.time()
    df = apply_semantic_role_labeling(df)
    print(f"Completed in {time.time() - step_start:.2f} seconds")
    
    # Step 3: Context window optimization
    print("\nStep 3: Context Window Optimization")
    step_start = time.time()
    optimized_df = optimize_context_windows(df)
    print(f"Completed in {time.time() - step_start:.2f} seconds")
    
    # Save the results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"advanced_processed_{data_source}_{timestamp}.parquet"
    optimized_df.to_parquet(output_file)
    print(f"\nSaved advanced preprocessed data to {output_file}")
    
    # Create a summary of the processing
    # Convert all values to serializable Python types
    most_common_types = optimized_df['entity_types'].explode().value_counts().head(5).to_dict()
    most_common_types = convert_to_serializable(most_common_types)
    
    summary = {
        "original_records": int(len(df)),
        "processed_chunks": int(len(optimized_df)),
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
    
    args = parser.parse_args()
    
    # Run the pipeline
    processed_df = run_advanced_preprocessing_pipeline(
        data_source=args.data_source,
        sample_size=args.sample_size
    )
    
    # Display sample of results
    print("\nSample of final processed data:")
    print(processed_df.head(2))