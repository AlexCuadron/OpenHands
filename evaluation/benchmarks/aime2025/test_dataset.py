#!/usr/bin/env python3
"""
Script to test the AIME2025 dataset loading and answer extraction.
"""

import os
import pandas as pd
from datasets import load_dataset

def load_aime2025_dataset():
    """Load the AIME2025 dataset."""
    print("Loading AIME2025 dataset...")
    try:
        # Try loading from Hugging Face
        dataset_i = load_dataset('opencompass/AIME2025', 'AIME2025-I')
        dataset_ii = load_dataset('opencompass/AIME2025', 'AIME2025-II')
        
        # Convert to pandas DataFrames
        aime_i_df = dataset_i['test'].to_pandas()
        aime_ii_df = dataset_ii['test'].to_pandas()
        
        # Add source information to distinguish between I and II
        aime_i_df['source'] = 'AIME2025-I'
        aime_ii_df['source'] = 'AIME2025-II'
        
        # Combine the datasets
        aime_df = pd.concat([aime_i_df, aime_ii_df], ignore_index=True)
        
        print(f"Successfully loaded AIME2025 dataset from Hugging Face with {len(aime_df)} problems")
    except Exception as e:
        print(f"Error loading AIME2025 dataset from Hugging Face: {e}")
        # As a fallback, try loading from the local directory
        print("Trying to load from local directory...")
        try:
            # Load from the local AIME2025 directory
            aime_i_path = "/workspace/OpenHands/AIME2025/aime2025-I.jsonl"
            aime_ii_path = "/workspace/OpenHands/AIME2025/aime2025-II.jsonl"
            
            aime_i_df = pd.read_json(aime_i_path, lines=True)
            aime_ii_df = pd.read_json(aime_ii_path, lines=True)
            
            # Add source information
            aime_i_df['source'] = 'AIME2025-I'
            aime_ii_df['source'] = 'AIME2025-II'
            
            # Combine the datasets
            aime_df = pd.concat([aime_i_df, aime_ii_df], ignore_index=True)
            
            print(f"Successfully loaded AIME2025 dataset from local files with {len(aime_df)} problems")
        except Exception as e2:
            print(f"Error loading from local directory: {e2}")
            raise ValueError("Failed to load AIME2025 dataset")
    
    # Add instance_id if not present
    if 'instance_id' not in aime_df.columns:
        aime_df['instance_id'] = aime_df.index.map(lambda x: f'aime2025_{x}')
    
    return aime_df

def main():
    """Main function."""
    # Load the dataset
    aime_df = load_aime2025_dataset()
    
    # Print dataset information
    print(f"Dataset columns: {aime_df.columns.tolist()}")
    print(f"Dataset shape: {aime_df.shape}")
    
    # Print the first 5 problems
    print("\nFirst 5 problems:")
    for i, row in aime_df.head(5).iterrows():
        print(f"\nProblem {i+1}:")
        print(f"ID: {row['instance_id']}")
        print(f"Question: {row['question']}")
        print(f"Answer: {row['answer']}")
        print(f"Source: {row['source']}")
    
    # Create a directory to save the dataset
    os.makedirs("aime2025_data", exist_ok=True)
    
    # Save the dataset to a CSV file
    aime_df.to_csv("aime2025_data/aime2025_dataset.csv", index=False)
    print("\nDataset saved to aime2025_data/aime2025_dataset.csv")

if __name__ == "__main__":
    main()