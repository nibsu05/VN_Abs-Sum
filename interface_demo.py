import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Config
MODEL_DIR = "./models/summarizer"
MAX_SOURCE_LENGTH = 384
DEFAULT_MAX_TARGET = 128


def load_model_and_tokenizer(model_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer, device


def summarize(text: str, model, tokenizer, device, max_length=DEFAULT_MAX_TARGET, num_beams=5):
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_SOURCE_LENGTH,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Inference demo for the trained summarizer")
    parser.add_argument("--text", "-t", type=str, default=None, help="Input text to summarize. If omitted, read from stdin.")
    parser.add_argument("--max_length", "-m", type=int, default=DEFAULT_MAX_TARGET, help="Max length of generated summary")
    parser.add_argument("--beams", "-b", type=int, default=5, help="Number of beams for beam search")
    args = parser.parse_args()

    if args.text is None:
        print("Enter/paste the text to summarize. Finish input with Ctrl-D (Unix) or Ctrl-Z then Enter (Windows):\n")
        # read from stdin until EOF
        import sys

        text = sys.stdin.read().strip()
        if not text:
            print("No input received. Exiting.")
            return
    else:
        text = args.text

    model, tokenizer, device = load_model_and_tokenizer(MODEL_DIR)
    print(f"Using device: {device}")
    summary = summarize(text, model, tokenizer, device, max_length=args.max_length, num_beams=args.beams)
    print("\n=== Generated summary ===\n")
    print(summary)


if __name__ == "__main__":
    main()
