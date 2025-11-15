#!/usr/bin/env python3
"""
Aggregate results from 10-fold cross-validation experiments.
Computes macro-F1 and per-class F1 scores with mean and standard deviation.

Usage:
    python aggregate_results.py results/
    python aggregate_results.py results/scibert_scivocab_uncased/
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate CV results')
    parser.add_argument('results_dir', type=str, help='Results directory')
    parser.add_argument('--output', type=str, default='aggregated_results.json',
                       help='Output file for aggregated results')
    return parser.parse_args()

def load_fold_results(results_dir):
    """Load results from all folds for all models"""
    results_dir = Path(results_dir)
    all_results = defaultdict(list)  # model_name -> list of fold results
    
    # Look for model directories
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        print(f"Processing model: {model_name}")
        
        # Look for fold directories
        for fold_dir in model_dir.iterdir():
            if not fold_dir.is_dir() or not fold_dir.name.startswith('fold_'):
                continue
                
            fold_num = fold_dir.name.replace('fold_', '')
            results_file = fold_dir / 'results.json'
            
            if results_file.exists():
                print(f"  Loading fold {fold_num}")
                with open(results_file, 'r') as f:
                    fold_results = json.load(f)
                    fold_results['fold'] = int(fold_num)
                    all_results[model_name].append(fold_results)
            else:
                print(f"  Warning: No results.json found for fold {fold_num}")
    
    return all_results

def aggregate_metrics(fold_results):
    """Aggregate metrics across folds"""
    if not fold_results:
        return {}
    
    # Extract metrics from test_metrics
    metrics_lists = defaultdict(list)
    
    for fold_result in fold_results:
        test_metrics = fold_result.get('test_metrics', {})
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)) and metric.startswith('eval_'):
                # Remove 'eval_' prefix
                clean_metric = metric.replace('eval_', '')
                metrics_lists[clean_metric].append(value)
    
    # Compute mean and std for each metric
    aggregated = {}
    for metric, values in metrics_lists.items():
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_values'] = values
    
    # Add fold count
    aggregated['num_folds'] = len(fold_results)
    
    return aggregated

def create_summary_table(all_results):
    """Create a summary table comparing all models"""
    summary_data = []
    
    for model_name, fold_results in all_results.items():
        if not fold_results:
            continue
            
        aggregated = aggregate_metrics(fold_results)
        
        # Extract key metrics
        row = {
            'Model': model_name,
            'Folds': aggregated.get('num_folds', 0),
            'Macro F1': f"{aggregated.get('macro_f1_mean', 0):.4f} ± {aggregated.get('macro_f1_std', 0):.4f}",
            'Accuracy': f"{aggregated.get('accuracy_mean', 0):.4f} ± {aggregated.get('accuracy_std', 0):.4f}",
            'F1 Neutral': f"{aggregated.get('f1_neutral_mean', 0):.4f} ± {aggregated.get('f1_neutral_std', 0):.4f}",
            'F1 Positive': f"{aggregated.get('f1_positive_mean', 0):.4f} ± {aggregated.get('f1_positive_std', 0):.4f}",
            'F1 Negative': f"{aggregated.get('f1_negative_mean', 0):.4f} ± {aggregated.get('f1_negative_std', 0):.4f}",
        }
        
        # Add raw means for sorting
        row['_macro_f1_mean'] = aggregated.get('macro_f1_mean', 0)
        row['_accuracy_mean'] = aggregated.get('accuracy_mean', 0)
        
        summary_data.append(row)
    
    # Sort by macro F1 (descending)
    summary_data.sort(key=lambda x: x['_macro_f1_mean'], reverse=True)
    
    # Remove raw columns
    for row in summary_data:
        del row['_macro_f1_mean']
        del row['_accuracy_mean']
    
    return pd.DataFrame(summary_data)

def print_detailed_results(all_results):
    """Print detailed results for each model"""
    print("\n" + "="*80)
    print("DETAILED RESULTS BY MODEL")
    print("="*80)
    
    for model_name, fold_results in all_results.items():
        if not fold_results:
            continue
            
        print(f"\n{model_name.upper()}")
        print("-" * len(model_name))
        
        aggregated = aggregate_metrics(fold_results)
        
        print(f"Number of folds: {aggregated.get('num_folds', 0)}")
        
        # Print key metrics
        key_metrics = [
            ('Macro F1', 'macro_f1'),
            ('Accuracy', 'accuracy'),
            ('F1 Neutral', 'f1_neutral'),
            ('F1 Positive', 'f1_positive'),
            ('F1 Negative', 'f1_negative'),
            ('Precision (macro)', 'precision_macro'),
            ('Recall (macro)', 'recall_macro')
        ]
        
        for metric_name, metric_key in key_metrics:
            mean_key = f'{metric_key}_mean'
            std_key = f'{metric_key}_std'
            if mean_key in aggregated:
                mean_val = aggregated[mean_key]
                std_val = aggregated.get(std_key, 0)
                print(f"  {metric_name:20}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Print per-fold values for macro F1
        if 'macro_f1_values' in aggregated:
            values = aggregated['macro_f1_values']
            print(f"  Macro F1 per fold    : {', '.join([f'{v:.4f}' for v in values])}")

def save_detailed_results(all_results, output_file):
    """Save detailed results to JSON file"""
    detailed_results = {}
    
    for model_name, fold_results in all_results.items():
        if not fold_results:
            continue
            
        aggregated = aggregate_metrics(fold_results)
        detailed_results[model_name] = {
            'aggregated_metrics': aggregated,
            'fold_results': fold_results
        }
    
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")

def main():
    args = parse_args()
    
    print(f"Aggregating results from: {args.results_dir}")
    
    # Load results
    all_results = load_fold_results(args.results_dir)
    
    if not all_results:
        print("No results found!")
        return
    
    print(f"\nFound results for {len(all_results)} models")
    
    # Create summary table
    summary_df = create_summary_table(all_results)
    
    # Print summary
    print("\n" + "="*120)
    print("SUMMARY TABLE")
    print("="*120)
    print(summary_df.to_string(index=False))
    
    # Print detailed results
    print_detailed_results(all_results)
    
    # Save results
    save_detailed_results(all_results, args.output)
    
    # Save summary CSV
    summary_csv = args.output.replace('.json', '_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary table saved to: {summary_csv}")

if __name__ == "__main__":
    main()