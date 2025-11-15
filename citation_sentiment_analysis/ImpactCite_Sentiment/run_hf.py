#!/usr/bin/env python3
"""
Hugging Face runner script for consistent evaluation of SciBERT and XLNet
on Citation Sentiment Corpus with 10-fold cross-validation.

Usage:
    python run_hf.py --model_name_or_path allenai/scibert_scivocab_uncased --fold 1
    python run_hf.py --model_name_or_path xlnet-large-cased --fold 1
"""

import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.metrics import classification_report, f1_score
import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description='HF runner for sentiment analysis')
    
    # Model arguments
    parser.add_argument('--model_name_or_path', type=str, required=True,
                       help='Model name or path (e.g., allenai/scibert_scivocab_uncased, xlnet-large-cased)')
    
    # Data arguments
    parser.add_argument('--fold', type=int, required=True, choices=range(1, 11),
                       help='Fold number (1-10)')
    parser.add_argument('--train_file', type=str, default=None,
                       help='Training file (default: splits/fold_{fold}/train.tsv)')
    parser.add_argument('--validation_file', type=str, default=None,
                       help='Validation file (default: splits/fold_{fold}/val.tsv)')
    parser.add_argument('--test_file', type=str, default=None,
                       help='Test file (default: splits/fold_{fold}/test.tsv)')
    
    # Training arguments
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--per_device_train_batch_size', type=int, default=16,
                       help='Training batch size per device')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32,
                       help='Evaluation batch size per device')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: results/{model_name}/fold_{fold})')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Class weights for imbalanced data
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights for imbalanced data')
    
    return parser.parse_args()

def load_data(file_path):
    """Load data from TSV file"""
    df = pd.read_csv(file_path, sep='\t')
    return df

def create_dataset(df, tokenizer, max_length):
    """Create HuggingFace dataset from DataFrame"""
    
    # Map labels to numbers
    label_map = {'o': 0, 'p': 1, 'n': 2}
    df['labels'] = df['label'].map(label_map)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=False, max_length=max_length)
    
    dataset = Dataset.from_pandas(df[['text', 'labels']])
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Compute macro F1
    macro_f1 = f1_score(labels, predictions, average='macro')
    
    # Compute per-class F1
    f1_scores = f1_score(labels, predictions, average=None)
    
    # Get classification report
    class_names = ['neutral', 'positive', 'negative']
    report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
    
    return {
        'macro_f1': macro_f1,
        'f1_neutral': f1_scores[0],
        'f1_positive': f1_scores[1], 
        'f1_negative': f1_scores[2],
        'accuracy': report['accuracy'],
        'precision_macro': report['macro avg']['precision'],
        'recall_macro': report['macro avg']['recall']
    }

class WeightedTrainer(Trainer):
    """Trainer with class weights support"""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss
            
        return (loss, outputs) if return_outputs else loss

def calculate_class_weights(train_df):
    """Calculate class weights for imbalanced data"""
    from sklearn.utils.class_weight import compute_class_weight
    
    label_map = {'o': 0, 'p': 1, 'n': 2}
    labels = [label_map[label] for label in train_df['label']]
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    return torch.FloatTensor(class_weights)

def main():
    args = parse_args()
    
    # Set default file paths if not provided
    if args.train_file is None:
        args.train_file = f"splits/fold_{args.fold}/train.tsv"
    if args.validation_file is None:
        args.validation_file = f"splits/fold_{args.fold}/val.tsv"
    if args.test_file is None:
        args.test_file = f"splits/fold_{args.fold}/test.tsv"
        
    # Set default output directory
    if args.output_dir is None:
        model_name = args.model_name_or_path.split('/')[-1]
        args.output_dir = f"results/{model_name}/fold_{args.fold}"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running experiment:")
    print(f"  Model: {args.model_name_or_path}")
    print(f"  Fold: {args.fold}")
    print(f"  Output: {args.output_dir}")
    
    # Load data
    print("Loading data...")
    train_df = load_data(args.train_file)
    val_df = load_data(args.validation_file)
    test_df = load_data(args.test_file)
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Print label distributions
    print(f"Train distribution: {dict(train_df['label'].value_counts())}")
    print(f"Test distribution: {dict(test_df['label'].value_counts())}")
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, 
        num_labels=3
    )
    
    # Handle special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = create_dataset(train_df, tokenizer, args.max_length)
    val_dataset = create_dataset(val_df, tokenizer, args.max_length)
    test_dataset = create_dataset(test_df, tokenizer, args.max_length)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Calculate class weights if requested
    class_weights = None
    if args.use_class_weights:
        class_weights = calculate_class_weights(train_df)
        print(f"Using class weights: {class_weights}")
    
    # Initialize trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    # Save test predictions
    test_predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(test_predictions.predictions, axis=1)
    
    # Create results summary
    results = {
        'model': args.model_name_or_path,
        'fold': args.fold,
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'test_metrics': test_results,
        'test_size': len(test_df),
        'train_size': len(train_df),
        'val_size': len(val_df)
    }
    
    # Save results
    with open(f"{args.output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    test_df_with_predictions = test_df.copy()
    test_df_with_predictions['predicted_label'] = predicted_labels
    test_df_with_predictions['predicted_label_name'] = [
        ['neutral', 'positive', 'negative'][pred] for pred in predicted_labels
    ]
    test_df_with_predictions.to_csv(f"{args.output_dir}/predictions.csv", index=False)
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Model: {args.model_name_or_path}")
    print(f"Fold: {args.fold}")
    print(f"Macro F1: {test_results['eval_macro_f1']:.4f}")
    print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"F1 Neutral: {test_results['eval_f1_neutral']:.4f}")
    print(f"F1 Positive: {test_results['eval_f1_positive']:.4f}")
    print(f"F1 Negative: {test_results['eval_f1_negative']:.4f}")
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()