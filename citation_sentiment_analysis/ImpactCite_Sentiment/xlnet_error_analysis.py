# XLNet Error Analysis for Citation Sentiment Classification
# Run this in Google Colab with GPU enabled

import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import random
import pickle

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class ErrorAnalysisTrainer(Trainer):
    """Extended trainer that saves predictions for error analysis"""
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.predictions_log = []
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss
            
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to capture predictions and probabilities"""
        result = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        
        if not prediction_loss_only:
            # Handle both tuple and PredictionOutput formats
            if isinstance(result, tuple):
                logits = result[1]  # (loss, predictions, labels)
            else:
                logits = result.predictions
                
            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
            predictions = np.argmax(logits, axis=-1)
            confidences = np.max(probs.numpy(), axis=-1)
            
            # Store for later analysis - handle tensor conversion properly
            labels = inputs['labels']
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            
            for pred, conf, true_label in zip(predictions, confidences, labels):
                self.predictions_log.append({
                    'prediction': int(pred),
                    'confidence': float(conf),
                    'true_label': int(true_label) if hasattr(true_label, 'item') else int(true_label)
                })
        
        return result

def load_full_dataset(pickle_path="data/dataset/dataset_full.pickle"):
    """Load the full cleaned dataset from pickle file"""
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list) and len(data) == 2:
            texts, labels = data[0], data[1]
            
            # Create DataFrame
            df = pd.DataFrame({
                'text': texts,
                'label': labels
            })
            
            print(f"✅ Loaded full dataset: {len(df):,} samples")
            
            # Show distribution
            label_counts = df['label'].value_counts()
            print("Label distribution:")
            for label, count in label_counts.items():
                label_name = {'o': 'neutral', 'p': 'positive', 'n': 'negative'}[label]
                print(f"   {label_name} ({label}): {count:,} ({count/len(df)*100:.1f}%)")
            
            return df
        else:
            raise ValueError(f"Expected list with 2 elements [texts, labels], got {type(data)}")
            
    except Exception as e:
        print(f"❌ Error loading dataset from {pickle_path}: {e}")
        return None

def create_train_test_split(df, test_size=0.2, random_state=42):
    """Create proper train/test split without data leakage"""
    
    # Stratified split to maintain class distribution
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']
    )
    
    # Further split training into train/val
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,  # 20% of training set for validation
        random_state=random_state,
        stratify=train_df['label']
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"Dataset split:")
    print(f"   Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify no overlap
    train_texts = set(train_df['text'])
    val_texts = set(val_df['text'])
    test_texts = set(test_df['text'])
    
    assert len(train_texts & val_texts) == 0, "Train/Val overlap detected!"
    assert len(train_texts & test_texts) == 0, "Train/Test overlap detected!"
    assert len(val_texts & test_texts) == 0, "Val/Test overlap detected!"
    
    print("✅ No data leakage detected")
    
    return train_df, val_df, test_df

def create_dataset_with_indices(df, tokenizer, max_length):
    """Create dataset with original indices for error analysis"""
    label_map = {'o': 0, 'p': 1, 'n': 2}
    df = df.copy()
    df['labels'] = df['label'].map(label_map)
    df['original_index'] = df.index
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=False, max_length=max_length)
    
    dataset = Dataset.from_pandas(df[['text', 'labels', 'original_index']])
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset, df

def calculate_class_weights(train_df):
    """Calculate class weights for imbalanced data"""
    label_map = {'o': 0, 'p': 1, 'n': 2}
    labels = [label_map[label] for label in train_df['label']]
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    return torch.FloatTensor(class_weights).to('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_text_patterns(df, label_col='label', text_col='text'):
    """Analyze text patterns by sentiment class"""
    analysis = {}
    
    for label in df[label_col].unique():
        subset = df[df[label_col] == label]
        texts = subset[text_col].tolist()
        
        # Text length analysis
        lengths = [len(text.split()) for text in texts]
        
        # Common patterns
        all_text = ' '.join(texts).lower()
        words = re.findall(r'\b\w+\b', all_text)
        
        analysis[label] = {
            'count': len(subset),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'common_words': Counter(words).most_common(10),
            'sample_texts': random.sample(texts, min(5, len(texts)))
        }
    
    return analysis

def create_error_analysis_report(df_test, predictions, probabilities, true_labels):
    """Create comprehensive error analysis report"""
    label_names = ['neutral', 'positive', 'negative']
    label_map = {0: 'o', 1: 'p', 2: 'n'}
    
    # Create results dataframe
    results_df = df_test.copy()
    results_df['predicted_label'] = [label_map[p] for p in predictions]
    results_df['predicted_numeric'] = predictions
    results_df['confidence'] = np.max(probabilities, axis=1)
    results_df['correct'] = (predictions == true_labels)
    
    # Add prediction probabilities
    for i, label in enumerate(label_names):
        results_df[f'prob_{label}'] = probabilities[:, i]
    
    # Error analysis
    errors_df = results_df[~results_df['correct']].copy()
    correct_df = results_df[results_df['correct']].copy()
    
    print("="*80)
    print("📊 XLNET ERROR ANALYSIS REPORT")
    print("="*80)
    
    # Overall metrics
    accuracy = np.mean(results_df['correct'])
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    
    print(f"\n🎯 Overall Performance:")
    print(f"   • Accuracy: {accuracy:.4f} ({np.sum(results_df['correct'])}/{len(results_df)})")
    print(f"   • Macro F1: {macro_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print(f"\n📈 Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(cm_df)
    
    # Class-wise performance
    report = classification_report(true_labels, predictions, target_names=label_names, output_dict=True)
    print(f"\n📋 Per-Class Performance:")
    for i, label in enumerate(label_names):
        f1 = report[label]['f1-score']
        precision = report[label]['precision']
        recall = report[label]['recall']
        support = report[label]['support']
        print(f"   • {label.capitalize()}: F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}, N={support}")
    
    # Error patterns by class
    print(f"\n🔍 Error Patterns:")
    for true_label in [0, 1, 2]:
        true_name = label_names[true_label]
        class_errors = errors_df[errors_df['labels'] == true_label]
        
        if len(class_errors) > 0:
            print(f"\n   {true_name.capitalize()} misclassified as:")
            pred_counts = class_errors['predicted_label'].value_counts()
            for pred_label, count in pred_counts.items():
                pred_name = {'o': 'neutral', 'p': 'positive', 'n': 'negative'}[pred_label]
                pct = (count / len(class_errors)) * 100
                print(f"     - {pred_name}: {count} ({pct:.1f}%)")
    
    # Confidence analysis
    print(f"\n🎲 Confidence Analysis:")
    print(f"   • Correct predictions avg confidence: {correct_df['confidence'].mean():.3f}")
    print(f"   • Incorrect predictions avg confidence: {errors_df['confidence'].mean():.3f}")
    
    # Low confidence predictions (potential uncertain cases)
    low_conf_threshold = 0.6
    low_conf = results_df[results_df['confidence'] < low_conf_threshold]
    print(f"   • Low confidence predictions (<{low_conf_threshold}): {len(low_conf)} ({len(low_conf)/len(results_df)*100:.1f}%)")
    
    return results_df, errors_df, correct_df

def sample_error_cases(errors_df, n_samples=5):
    """Sample and display error cases for manual inspection"""
    print(f"\n🔬 Sample Error Cases for Manual Review:")
    print("="*80)
    
    for label in errors_df['label'].unique():
        label_errors = errors_df[errors_df['label'] == label]
        if len(label_errors) == 0:
            continue
            
        label_name = {'o': 'neutral', 'p': 'positive', 'n': 'negative'}[label]
        print(f"\n{label_name.upper()} Examples (True: {label_name}):")
        print("-" * 60)
        
        # Sample both high and low confidence errors
        sample_size = min(n_samples, len(label_errors))
        sampled = label_errors.sample(n=sample_size, random_state=42)
        
        for idx, row in sampled.iterrows():
            pred_name = {'o': 'neutral', 'p': 'positive', 'n': 'negative'}[row['predicted_label']]
            print(f"\nPredicted: {pred_name} (confidence: {row['confidence']:.3f})")
            print(f"Text: {row['text'][:200]}{'...' if len(row['text']) > 200 else ''}")
            print()

def train_xlnet_model(test_size=0.2, save_model=True):
    """
    Train XLNet model using full dataset with proper train/test split
    
    Args:
        test_size: Proportion of data to use for testing (default 0.2 = 20%)
        save_model: Whether to save the trained model
    
    Returns:
        output_dir: Directory where model and datasets are saved
    """
    model_name = 'xlnet-base-cased'
    print(f"🚀 Training XLNet Model on Full Dataset")
    print(f"   Test size: {test_size*100}% ({100-test_size*100}% for train/val)")
    
    # Load full dataset
    full_df = load_full_dataset()
    if full_df is None:
        print("❌ Failed to load full dataset")
        return None
    
    # Create proper train/val/test split
    train_df, val_df, test_df = create_train_test_split(full_df, test_size=test_size)
    
    print(f"   • Train: {len(train_df):,} samples")
    print(f"   • Val: {len(val_df):,} samples")
    print(f"   • Test: {len(test_df):,} samples")
    
    # Text pattern analysis before training
    print("\n📝 Pre-training text analysis:")
    text_patterns = analyze_text_patterns(train_df)
    for label, stats in text_patterns.items():
        label_name = {'o': 'neutral', 'p': 'positive', 'n': 'negative'}[label]
        print(f"   {label_name}: {stats['count']} samples, avg {stats['avg_length']:.1f} words")
    
    # Load model and tokenizer
    print(f"\n🤖 Loading XLNet...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Create datasets with indices for error tracking
    print("🔤 Tokenizing data...")
    train_dataset, train_df_indexed = create_dataset_with_indices(train_df, tokenizer, 256)
    val_dataset, val_df_indexed = create_dataset_with_indices(val_df, tokenizer, 256)
    test_dataset, test_df_indexed = create_dataset_with_indices(test_df, tokenizer, 256)
    
    # Training setup
    output_dir = f'xlnet_model_full_dataset'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets for later analysis
    train_df_indexed.to_csv(f'{output_dir}/train_data.csv', index=False)
    val_df_indexed.to_csv(f'{output_dir}/val_data.csv', index=False)  
    test_df_indexed.to_csv(f'{output_dir}/test_data.csv', index=False)
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_df)
    print(f"⚖️  Class weights: {class_weights.tolist()}")
    
    # Save class weights for later use
    torch.save(class_weights, f'{output_dir}/class_weights.pt')
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'{output_dir}/checkpoints',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch", 
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        dataloader_num_workers=2,
        report_to="none",
    )
    
    # Initialize trainer
    trainer_kwargs = {
        'class_weights': class_weights,
        'model': model,
        'args': training_args,
        'train_dataset': train_dataset,
        'eval_dataset': val_dataset,
        'data_collator': DataCollatorWithPadding(tokenizer=tokenizer),
    }
    
    # Use processing_class instead of tokenizer for newer versions
    import transformers
    if hasattr(transformers, '__version__') and tuple(map(int, transformers.__version__.split('.')[:2])) >= (4, 46):
        trainer_kwargs['processing_class'] = tokenizer
    else:
        trainer_kwargs['tokenizer'] = tokenizer
    
    trainer = ErrorAnalysisTrainer(**trainer_kwargs)
    
    # Train
    print(f"\n🔥 Training XLNet...")
    trainer.train()
    
    # Save the trained model
    if save_model:
        print(f"💾 Saving trained model...")
        trainer.save_model(f'{output_dir}/final_model')
        tokenizer.save_pretrained(f'{output_dir}/final_model')
        
        # Save training info
        training_info = {
            'model_name': model_name,
            'data_split': 'full_dataset_proper_split',
            'total_samples': len(full_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'test_size': test_size,
            'class_weights': class_weights.tolist(),
            'training_args': training_args.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f'{output_dir}/training_info.json', 'w') as f:
            json.dump(training_info, f, indent=2)
    
    print(f"✅ Training complete! Model saved to: {output_dir}")
    
    return output_dir

def analyze_xlnet_errors(model_dir, n_error_samples=5):
    """
    Analyze errors from a pre-trained XLNet model
    
    Args:
        model_dir: Directory containing trained model and data
        n_error_samples: Number of error samples to display per class
    
    Returns:
        results_df, errors_df, summary: Analysis results
    """
    print(f"🔍 Starting XLNet Error Analysis")
    print(f"   Using model from: {model_dir}")
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        print("   Please run train_xlnet_model() first")
        return None, None, None
    
    # Load training info
    try:
        with open(f'{model_dir}/training_info.json', 'r') as f:
            training_info = json.load(f)
        print(f"   Model trained on: {training_info['data_split']}")
        print(f"   Training date: {training_info['timestamp'][:19]}")
    except:
        print("   ⚠️  Training info not found, proceeding anyway...")
    
    # Load test data
    test_df = pd.read_csv(f'{model_dir}/test_data.csv')
    print(f"   • Test samples: {len(test_df):,}")
    
    # Load model and tokenizer
    print(f"🤖 Loading trained model...")
    model_path = f'{model_dir}/final_model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Load class weights
    try:
        class_weights = torch.load(f'{model_dir}/class_weights.pt')
        print(f"⚖️  Class weights: {class_weights.tolist()}")
    except:
        print("   ⚠️  Class weights not found, proceeding without...")
        class_weights = None
    
    # Create test dataset
    print("🔤 Preparing test data...")
    test_dataset, test_df_indexed = create_dataset_with_indices(test_df, tokenizer, 256)
    
    # Setup for prediction
    training_args = TrainingArguments(
        output_dir=f'{model_dir}/analysis',
        per_device_eval_batch_size=16,
        dataloader_num_workers=2,
        report_to="none",
    )
    
    # Initialize trainer for prediction
    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'data_collator': DataCollatorWithPadding(tokenizer=tokenizer),
    }
    
    # Handle tokenizer parameter
    import transformers
    if hasattr(transformers, '__version__') and tuple(map(int, transformers.__version__.split('.')[:2])) >= (4, 46):
        trainer_kwargs['processing_class'] = tokenizer
    else:
        trainer_kwargs['tokenizer'] = tokenizer
    
    trainer = Trainer(**trainer_kwargs)
    
    # Get predictions with probabilities
    print(f"🔍 Generating predictions...")
    predictions_output = trainer.predict(test_dataset)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
    probabilities = torch.nn.functional.softmax(torch.tensor(predictions_output.predictions), dim=-1).numpy()
    true_labels = predictions_output.label_ids
    
    # Error Analysis
    results_df, errors_df, correct_df = create_error_analysis_report(
        test_df_indexed, predictions, probabilities, true_labels
    )
    
    # Sample error cases for manual review
    sample_error_cases(errors_df, n_samples=n_error_samples)
    
    # Save detailed results
    analysis_dir = f'{model_dir}/analysis_results'
    os.makedirs(analysis_dir, exist_ok=True)
    
    results_file = f'{analysis_dir}/error_analysis_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Save error analysis summary
    summary = {
        'model_dir': model_dir,
        'total_samples': int(len(results_df)),
        'accuracy': float(np.mean(results_df['correct'])),
        'macro_f1': float(f1_score(true_labels, predictions, average='macro')),
        'error_count': int(len(errors_df)),
        'error_rate': float(len(errors_df) / len(results_df)),
        'class_distribution': {str(k): int(v) for k, v in test_df['label'].value_counts().items()},
        'confusion_matrix': confusion_matrix(true_labels, predictions).tolist(),
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    with open(f'{analysis_dir}/analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Error analysis complete! Results saved to: {analysis_dir}")
    
    return results_df, errors_df, summary

# Usage examples:
if __name__ == "__main__":
    print("XLNet Error Analysis - Full Dataset with Proper Split:")
    print("=" * 60)
    print()
    print("🚀 This script now uses the full cleaned dataset with proper train/test split")
    print("   - No data leakage between train and test sets")
    print("   - Stratified split maintains class distribution")
    print("   - 80% train/val, 20% test (configurable)")
    print()
    print("📚 Step 1 - Train Model (run once):")
    print("   model_dir = train_xlnet_model()")
    print("   # or custom test size: model_dir = train_xlnet_model(test_size=0.15)")
    print()
    print("🔍 Step 2 - Analyze Errors (run multiple times with same model):")
    print("   results_df, errors_df, summary = analyze_xlnet_errors(model_dir)")
    print("   # or: results_df, errors_df, summary = analyze_xlnet_errors('xlnet_model_full_dataset')")
    print()
    print("⚠️  IMPORTANT: Previous results using fold combination had data leakage!")
    print("   This version ensures proper train/test separation.")
    print()
    print("🚀 Quick start - Train and analyze:")
    
    # Train model first
    print("\n" + "="*50)
    print("Training model on full dataset...")
    model_dir = train_xlnet_model()
    
    # Then analyze errors  
    if model_dir:
        print("\n" + "="*50)
        print("Analyzing errors...")
        results_df, errors_df, summary = analyze_xlnet_errors(model_dir, n_error_samples=3)