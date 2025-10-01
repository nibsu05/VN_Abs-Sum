import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from evaluate import load
from tqdm import tqdm

MODEL_DIR = "./models/summarizer"
TEST_CSV = "data/processed/test.csv"
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 8  # tăng nếu đủ GPU/VRAM

def load_test_dataset(csv_path, tokenizer):
    df = pd.read_csv(csv_path)
    ds = Dataset.from_pandas(df)
    def preprocess(ex):
        return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=MAX_SOURCE_LENGTH)
    ds = ds.map(preprocess, batched=True)
    ds.set_format(type="torch", columns=["input_ids","attention_mask"])
    return ds, df

def generate_and_score():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    ds, df = load_test_dataset(TEST_CSV, tokenizer)
    rouge = load("rouge")

    preds = []
    refs = []

    for i in range(0, len(ds), BATCH_SIZE):
        batch = ds[i:i+BATCH_SIZE]
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_TARGET_LENGTH,
                num_beams=5,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        batch_preds = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds.extend(batch_preds)
        # references from original dataframe (string)
        refs.extend(df["summary"].iloc[i:i+BATCH_SIZE].tolist())

    # compute rouge
    results = rouge.compute(predictions=preds, references=refs)
    print("ROUGE:", results)

    # save outputs
    out_df = df.copy()
    out_df["predicted_summary"] = preds
    out_df.to_csv("results/predictions_test.csv", index=False)
    print("Saved predictions to results/predictions_test.csv")

if __name__ == "__main__":
    generate_and_score()