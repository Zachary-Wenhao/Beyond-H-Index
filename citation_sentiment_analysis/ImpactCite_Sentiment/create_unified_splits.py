#!/usr/bin/env python3
"""
Convert ImpactCite CSC-Clean fold data to unified TSV format for consistent evaluation.
Creates train.tsv, val.tsv, test.tsv for each fold with format: text<TAB>label
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse

def create_splits_directory():
    """Create the splits directory structure"""
    splits_dir = Path("splits")
    splits_dir.mkdir(exist_ok=True)
    
    for k in range(1, 11):
        fold_dir = splits_dir / f"fold_{k}"
        fold_dir.mkdir(exist_ok=True)
    
    return splits_dir

def load_dataset():
    """Load the dataset from pickle files"""
    print("Loading dataset...")
    
    # Load the folds data
    with open('data/dataset/dataset_folds.pickle', 'rb') as f:
        dataset_folds = pickle.load(f)
    
    # dataset_folds is [texts_by_fold, labels_by_fold]
    texts_by_fold = dataset_folds[0]  # List of 10 folds, each containing texts
    labels_by_fold = dataset_folds[1] # List of 10 folds, each containing labels
    
    print(f"Loaded {len(texts_by_fold)} folds")
    print(f"Fold sizes: {[len(fold) for fold in texts_by_fold]}")
    
    return texts_by_fold, labels_by_fold

def create_fold_splits(texts_by_fold, labels_by_fold, splits_dir, validation_ratio=0.15, random_state=42):
    """Create train/val/test splits for each fold"""
    
    label_map = {'o': 0, 'p': 1, 'n': 2}  # neutral, positive, negative
    
    for k in range(10):  # 0-9 indexing in data, but we'll name folders 1-10
        fold_num = k + 1
        fold_dir = splits_dir / f"fold_{fold_num}"
        
        print(f"\nProcessing fold {fold_num}...")
        
        # Get test data (kth fold)
        test_texts = texts_by_fold[k]
        test_labels = labels_by_fold[k]
        
        print(f"Test set size: {len(test_texts)}")
        
        # Get training data (all other 9 folds)
        train_texts = []
        train_labels = []
        
        for i in range(10):
            if i != k:  # Skip the test fold
                train_texts.extend(texts_by_fold[i])
                train_labels.extend(labels_by_fold[i])
        
        print(f"Combined train set size: {len(train_texts)}")
        
        # Split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            train_texts, train_labels, 
            test_size=validation_ratio, 
            random_state=random_state,
            stratify=train_labels
        )
        
        print(f"Final train set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        
        # Create TSV files
        def create_tsv(texts, labels, filename):
            df = pd.DataFrame({
                'text': texts,
                'label': labels
            })
            
            # Add numeric labels for consistency
            df['label_num'] = df['label'].map(label_map)
            
            # Check distribution
            label_counts = df['label'].value_counts()
            print(f"{filename} distribution: {dict(label_counts)}")
            
            # Save as TSV
            filepath = fold_dir / filename
            df[['text', 'label']].to_csv(filepath, sep='\t', index=False)
            
            # Also save with numeric labels
            numeric_filepath = fold_dir / f"{filename.replace('.tsv', '_numeric.tsv')}"
            df[['text', 'label_num']].to_csv(numeric_filepath, sep='\t', 
                                           index=False, header=['text', 'label'])
            
            return len(texts)
        
        # Create files
        create_tsv(X_train, y_train, 'train.tsv')
        create_tsv(X_val, y_val, 'val.tsv')
        create_tsv(test_texts, test_labels, 'test.tsv')
        
        print(f"Created files for fold {fold_num}")

def verify_splits(splits_dir):
    """Verify that splits are created correctly"""
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY")
    print("="*50)
    
    total_samples = 0
    for k in range(1, 11):
        fold_dir = splits_dir / f"fold_{k}"
        
        train_df = pd.read_csv(fold_dir / 'train.tsv', sep='\t')
        val_df = pd.read_csv(fold_dir / 'val.tsv', sep='\t')
        test_df = pd.read_csv(fold_dir / 'test.tsv', sep='\t')
        
        fold_total = len(train_df) + len(val_df) + len(test_df)
        total_samples += fold_total
        
        print(f"Fold {k}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}, Total={fold_total}")
        
        # Check label distribution for test set
        test_dist = test_df['label'].value_counts()
        print(f"  Test distribution: {dict(test_dist)}")
    
    print(f"\nTotal samples across all folds: {total_samples}")
    print(f"Average per fold: {total_samples / 10:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Create unified TSV splits for CSC evaluation')
    parser.add_argument('--validation_ratio', type=float, default=0.15, 
                       help='Ratio of training data to use for validation (default: 0.15)')
    parser.add_argument('--random_state', type=int, default=42, 
                       help='Random seed for train/val split (default: 42)')
    
    args = parser.parse_args()
    
    print("Creating unified TSV splits for Citation Sentiment Corpus evaluation")
    print(f"Validation ratio: {args.validation_ratio}")
    print(f"Random seed: {args.random_state}")
    
    # Create directory structure
    splits_dir = create_splits_directory()
    
    # Load dataset
    texts_by_fold, labels_by_fold = load_dataset()
    
    # Create splits
    create_fold_splits(texts_by_fold, labels_by_fold, splits_dir, args.validation_ratio, args.random_state)
    
    # Verify
    verify_splits(splits_dir)
    
    print("\n✅ Unified splits created successfully!")
    print("Files created:")
    print("  splits/fold_k/train.tsv - Training data")
    print("  splits/fold_k/val.tsv   - Validation data") 
    print("  splits/fold_k/test.tsv  - Test data")
    print("  splits/fold_k/*_numeric.tsv - Numeric label versions")
    
    print("\nLabel mapping: o=0 (neutral), p=1 (positive), n=2 (negative)")

if __name__ == "__main__":
    main()