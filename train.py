import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from evaluate import load
import numpy as np
from tqdm import tqdm

# Constants
MODEL_NAME = "VietAI/vit5-base"  # Vietnamese T5 model
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 2  # Giảm batch size để tránh lỗi hết RAM
EPOCHS = 3
LEARNING_RATE = 5e-5

def prepare_dataset(df, tokenizer):
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples['text'],
            max_length=MAX_SOURCE_LENGTH,
            padding='max_length',
            truncation=True
        )
        
        labels = tokenizer(
            examples['summary'],
            max_length=MAX_TARGET_LENGTH,
            padding='max_length',
            truncation=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Convert pandas DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # Preprocess the dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.column_names,
        batch_size=BATCH_SIZE
    )
    
    return tokenized_dataset

def compute_metrics(pred, tokenizer):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Load and compute ROUGE scores
    rouge = load('rouge')
    results = rouge.compute(predictions=pred_str, references=label_str)
    
    return {
        'rouge1': results['rouge1'],
        'rouge2': results['rouge2'],
        'rougeL': results['rougeL'],
    }

def main():
    # Load data
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # --- GPU debug / force device ---
    # Print torch/cuda info and move model to GPU if available so we can verify Trainer uses it
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    try:
        print("Model device after .to():", next(model.parameters()).device)
    except StopIteration:
        print("Model has no parameters to check device.")

    # Create datasets
    train_dataset = prepare_dataset(train_df, tokenizer)
    val_dataset = prepare_dataset(val_df, tokenizer)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,  # Effective batch size = BATCH_SIZE * gradient_accumulation_steps
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        save_total_limit=3,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        save_strategy="steps",
        logging_steps=50,
        eval_steps=200,
        save_steps=200
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # Create compute_metrics function that uses our tokenizer
    compute_metrics_with_tokenizer = lambda pred: compute_metrics(pred, tokenizer)

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_with_tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("./models/summarizer")
    tokenizer.save_pretrained("./models/summarizer")

if __name__ == "__main__":
    main()