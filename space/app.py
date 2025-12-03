# app.py
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "nibsu5/summarizer-model"  # thay bằng model của bạn
MAX_SOURCE_LENGTH = 384
DEFAULT_MAX_TARGET = 128

@gr.cache_resource  # Gradio sẽ cache resource giữa request
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return model, tokenizer, device

def summarize(text, max_length=DEFAULT_MAX_TARGET, beams=4):
    model, tokenizer, device = load_model()
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
            max_length=int(max_length),
            num_beams=int(beams),
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

with gr.Blocks() as demo:
    gr.Markdown("## Summarizer Demo")
    inp = gr.Textbox(lines=10, placeholder="Paste text here...")
    max_len = gr.Slider(16, 512, value=DEFAULT_MAX_TARGET, step=1, label="Max summary length")
    beams = gr.Slider(1, 6, value=4, step=1, label="Beams")
    out = gr.Textbox(label="Summary")
    btn = gr.Button("Generate")
    btn.click(summarize, inputs=[inp, max_len, beams], outputs=[out])

if __name__ == "__main__":
    demo.launch()