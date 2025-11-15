#!/usr/bin/env python3
"""
Examine the structure of the dataset_folds.pickle file
"""

import pickle
import numpy as np

def examine_dataset():
    print("Examining dataset_folds.pickle...")
    
    with open('data/dataset/dataset_folds.pickle', 'rb') as f:
        dataset_folds = pickle.load(f)
    
    print(f"Type: {type(dataset_folds)}")
    print(f"Length/Shape: {len(dataset_folds) if hasattr(dataset_folds, '__len__') else 'N/A'}")
    
    if isinstance(dataset_folds, (list, tuple)):
        print("\nExamining list/tuple structure:")
        for i, item in enumerate(dataset_folds):
            print(f"Item {i}: Type={type(item)}, Length={len(item) if hasattr(item, '__len__') else 'N/A'}")
            if hasattr(item, '__len__') and len(item) > 0:
                if hasattr(item, '__getitem__'):
                    print(f"  First element type: {type(item[0])}")
                    if isinstance(item[0], (list, tuple)) and len(item[0]) > 0:
                        print(f"  First element length: {len(item[0])}")
                        print(f"  Sample: {str(item[0][:3])[:100]}...")
    
    print("\n" + "="*50)
    print("Also examining dataset_full.pickle for comparison...")
    
    with open('data/dataset/dataset_full.pickle', 'rb') as f:
        dataset_full = pickle.load(f)
    
    print(f"dataset_full type: {type(dataset_full)}")
    print(f"dataset_full length: {len(dataset_full) if hasattr(dataset_full, '__len__') else 'N/A'}")
    
    if isinstance(dataset_full, (list, tuple)) and len(dataset_full) >= 2:
        texts, labels = dataset_full[0], dataset_full[1]
        print(f"Texts length: {len(texts)}")
        print(f"Labels length: {len(labels)}")
        
        # Check label distribution
        from collections import Counter
        label_counts = Counter(labels)
        print(f"Label distribution: {dict(label_counts)}")
        
        print(f"Sample text: {texts[0][:100]}...")
        print(f"Sample labels: {labels[:10]}")

if __name__ == "__main__":
    examine_dataset()