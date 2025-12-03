import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import os

def generate_predictions():
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    model_path = "./models/summarizer"
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load test data
    test_file = "data/processed/test.csv"
    print(f"Loading test data from {test_file}...")
    try:
        df_test = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"Error: Test file not found at {test_file}")
        return

    # Prepare for predictions
    predictions = []
    
    print("Generating summaries...")
    # Batch processing could be faster, but let's stick to simple loop for clarity/safety first
    for index, row in tqdm(df_test.iterrows(), total=len(df_test)):
        text = row['text']
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        
        # Generate
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=128,
                min_length=30,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
        
        # Decode
        pred_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        predictions.append(pred_summary)

    # Add predictions to dataframe
    df_test['predicted_summary'] = predictions

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Save results
    output_file = "results/predictions_test.csv"
    df_test.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    generate_predictions()
