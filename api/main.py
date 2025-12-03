from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sys
import os

# Add parent directory to path to import preprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import clean_text, tokenize_text

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
# Load model and tokenizer
# Use the fine-tuned model path
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(MODEL_DIR, "models", "summarizer")

# Fallback to base model if fine-tuned model doesn't exist
if not os.path.exists(MODEL_PATH):
    print(f"Warning: Fine-tuned model not found at {MODEL_PATH}. Using base model.")
    MODEL_PATH = "VietAI/vit5-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model from {MODEL_PATH} to {device}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

class SummarizeRequest(BaseModel):
    text: str

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    try:
        # Preprocess
        raw_text = request.text
        cleaned_text = clean_text(raw_text)
        tokenized_text = tokenize_text(cleaned_text)
        
        # Tokenize for model
        inputs = tokenizer(
            tokenized_text,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        # Decode
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
