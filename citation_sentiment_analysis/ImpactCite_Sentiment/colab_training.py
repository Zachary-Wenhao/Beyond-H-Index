# Citation Sentiment Analysis - Google Colab Setup
# Run this in Google Colab with GPU enabled

# Install dependencies
!pip install transformers datasets torch accelerate evaluate scikit-learn pandas numpy tqdm

# Mount Google Drive (optional - to save results)
from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import json
from datetime import datetime
from tqdm import tqdm
import time

# Check if GPU is available
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Upload your splits.zip file to Colab, then unzip
# !unzip splits.zip

def load_data(file_path):
    """Load data from TSV file"""
    df = pd.read_csv(file_path, sep='\t')
    return df

def create_dataset(df, tokenizer, max_length):
    """Create HuggingFace dataset from DataFrame"""
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
    
    macro_f1 = f1_score(labels, predictions, average='macro')
    f1_scores = f1_score(labels, predictions, average=None)
    
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
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss with class weights support.
        Compatible with newer transformers versions that pass additional kwargs.
        """
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
    label_map = {'o': 0, 'p': 1, 'n': 2}
    labels = [label_map[label] for label in train_df['label']]
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    return torch.FloatTensor(class_weights).to('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model_name, fold, num_epochs=3, batch_size=16):
    """Train a model on a specific fold"""
    start_time = time.time()
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
    print(f"\n{'='*60}")
    print(f"🚀 Starting {model_short} on fold {fold}")
    print(f"{'='*60}")
    
    # Load data
    print("📊 Loading data...")
    train_df = load_data(f'splits/fold_{fold}/train.tsv')
    val_df = load_data(f'splits/fold_{fold}/val.tsv')
    test_df = load_data(f'splits/fold_{fold}/test.tsv')
    
    print(f"  • Train: {len(train_df):,} samples")
    print(f"  • Val: {len(val_df):,} samples") 
    print(f"  • Test: {len(test_df):,} samples")
    
    # Show class distribution
    train_dist = train_df['label'].value_counts().sort_index()
    print(f"  • Class distribution: {dict(train_dist)}")
    
    # Load model and tokenizer
    print(f"🤖 Loading {model_short}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Create datasets
    print("🔤 Tokenizing data...")
    train_dataset = create_dataset(train_df, tokenizer, 256)
    val_dataset = create_dataset(val_df, tokenizer, 256)
    test_dataset = create_dataset(test_df, tokenizer, 256)
    
    # Training arguments (optimized for Colab GPU)
    output_dir = f'results/{model_name.replace("/", "_")}/fold_{fold}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate training steps for progress tracking
    steps_per_epoch = len(train_dataset) // batch_size
    total_steps = steps_per_epoch * num_epochs
    
    print(f"📈 Training configuration:")
    print(f"  • Epochs: {num_epochs}")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Steps per epoch: {steps_per_epoch:,}")
    print(f"  • Total training steps: {total_steps:,}")
    
    # More realistic time estimates for GPU training
    if torch.cuda.is_available():
        estimated_seconds_per_step = 0.5  # Much faster with GPU
        estimated_time = total_steps * estimated_seconds_per_step
        print(f"  • Estimated time (GPU): {estimated_time:.0f} seconds ({estimated_time / 60:.1f} minutes)")
    else:
        estimated_seconds_per_step = 3  # Slower with CPU
        estimated_time = total_steps * estimated_seconds_per_step
        print(f"  • Estimated time (CPU): {estimated_time:.0f} seconds ({estimated_time / 60:.1f} minutes)")
        print(f"  ⚠️  CPU training will be very slow - consider using GPU!")
    
    # Check transformers version for compatibility
    import transformers
    transformers_version = transformers.__version__
    print(f"  • Transformers version: {transformers_version}")
    
    # Handle parameter naming changes in newer versions
    # Try to create a dummy TrainingArguments to check supported parameters
    try:
        # Test if eval_strategy parameter exists
        TrainingArguments(output_dir="./temp", eval_strategy="no")
        eval_strategy_param = "eval_strategy"
    except TypeError:
        # Fall back to evaluation_strategy for older versions
        eval_strategy_param = "evaluation_strategy"
    
    training_args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size * 2,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": max(10, steps_per_epoch // 10),
        eval_strategy_param: "epoch",  # Compatible with both old and new versions
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "fp16": True,
        "dataloader_num_workers": 2,
        "report_to": "none",
        "disable_tqdm": False,
    }
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_df)
    print(f"⚖️  Class weights: {class_weights.tolist()}")
    
    # Initialize trainer
    print("🎯 Initializing trainer...")
    
    # Handle tokenizer parameter deprecation in newer transformers versions
    trainer_kwargs = {
        'class_weights': class_weights,
        'model': model,
        'args': training_args,
        'train_dataset': train_dataset,
        'eval_dataset': val_dataset,
        'data_collator': DataCollatorWithPadding(tokenizer=tokenizer),
        'compute_metrics': compute_metrics,
    }
    
    # Use processing_class instead of tokenizer for newer versions
    import transformers
    if hasattr(transformers, '__version__') and tuple(map(int, transformers.__version__.split('.')[:2])) >= (4, 46):
        trainer_kwargs['processing_class'] = tokenizer
    else:
        trainer_kwargs['tokenizer'] = tokenizer
    
    trainer = WeightedTrainer(**trainer_kwargs)
    
    # Train
    if torch.cuda.is_available():
        eta_minutes = total_steps * 0.5 / 60
        print(f"🔥 Starting training... (ETA: ~{eta_minutes:.1f} minutes with GPU)")
    else:
        eta_minutes = total_steps * 3 / 60
        print(f"🔥 Starting training... (ETA: ~{eta_minutes:.1f} minutes with CPU)")
    
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start
    
    print(f"✅ Training completed in {train_time / 60:.1f} minutes")
    
    # Evaluate
    print("🔍 Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    # Save results
    total_time = time.time() - start_time
    results = {
        'model': model_name,
        'fold': fold,
        'timestamp': datetime.now().isoformat(),
        'test_metrics': test_results,
        'test_size': len(test_df),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'training_time_minutes': train_time / 60,
        'total_time_minutes': total_time / 60,
        'hyperparameters': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1
        }
    }
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"{'='*60}")
    print(f"🎉 Completed {model_short} fold {fold}")
    print(f"📊 Results:")
    print(f"  • Macro F1: {test_results['eval_macro_f1']:.4f}")
    print(f"  • F1 Neutral: {test_results['eval_f1_neutral']:.4f}")
    print(f"  • F1 Positive: {test_results['eval_f1_positive']:.4f}")
    print(f"  • F1 Negative: {test_results['eval_f1_negative']:.4f}")
    print(f"  • Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"⏱️  Training time: {train_time / 60:.1f} minutes")
    print(f"⏱️  Total time: {total_time / 60:.1f} minutes")
    print(f"{'='*60}\n")
    
    return results

# Example usage - Test single run first:
print("🧪 Testing single run to validate setup...")
try:
    test_result = train_model('allenai/scibert_scivocab_uncased', 1, num_epochs=1, batch_size=8)
    print("✅ Single run test passed! Ready for full evaluation.")
    print(f"Test result macro-F1: {test_result['test_metrics']['eval_macro_f1']:.4f}")
except Exception as e:
    print(f"❌ Single run test failed: {e}")
    print("Fix the issue before running full evaluation.")

# Uncomment below to run full evaluation after testing:
# """

# Run all folds for both models (or add third model)
models = [
    'allenai/scibert_scivocab_uncased',  # Domain-specific SciBERT
    'xlnet-base-cased',                   # XLNet architecture
    # 'roberta-base',                     # General domain RoBERTa (uncomment to add)
    # 'microsoft/deberta-v3-base',        # DeBERTa (newer architecture)
    # 'facebook/bart-base'                # BART (encoder-decoder)
]
all_results = []
failed_runs = []

print(f"🎯 Starting comprehensive evaluation:")
print(f"  • Models: {len(models)} ({', '.join([m.split('/')[-1] for m in models])})")
print(f"  • Folds: 10")
print(f"  • Total runs: {len(models) * 10}")
print(f"  • Estimated total time: {len(models) * 10 * 15:.0f} minutes ({len(models) * 10 * 15 / 60:.1f} hours)")
print()

total_start = time.time()
with tqdm(total=len(models) * 10, desc="Overall Progress", unit="run") as pbar:
    for model_idx, model in enumerate(models):
        model_short = model.split('/')[-1] if '/' in model else model
        pbar.set_description(f"Training {model_short}")
        
        for fold in range(1, 11):  # All 10 folds
            run_id = f"{model_short}_fold_{fold}"
            try:
                pbar.set_postfix(fold=fold)
                result = train_model(model, fold, num_epochs=3, batch_size=16)
                all_results.append(result)
                pbar.update(1)
                print(f"✅ {run_id} completed successfully")
            except KeyboardInterrupt:
                print(f"\n⚠️  Training interrupted by user at {run_id}")
                pbar.update(1)
                break
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Failed {run_id}: {error_msg}")
                failed_runs.append({'run_id': run_id, 'error': error_msg})
                pbar.update(1)
                
                # Continue with next run instead of stopping
                continue

total_time = time.time() - total_start

print(f"\n{'='*60}")
print(f"🎉 Comprehensive evaluation completed!")
print(f"✅ Successfully completed: {len(all_results)}/{len(models) * 10} runs")
if failed_runs:
    print(f"❌ Failed runs: {len(failed_runs)}")
    for failure in failed_runs:
        print(f"  • {failure['run_id']}: {failure['error']}")
print(f"⏱️  Total time: {total_time / 3600:.1f} hours")

# Save summary
summary = {
    'total_runs': len(models) * 10,
    'successful_runs': len(all_results),
    'failed_runs': failed_runs,
    'total_time_hours': total_time / 3600,
    'results': all_results
}

with open('training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
    
print(f"📁 Results saved to training_summary.json")
# """