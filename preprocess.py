import json
import re
from tqdm import tqdm
import pandas as pd
from underthesea import word_tokenize
import numpy as np

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove special characters and extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.,!?\-đĐ]', '', text)
    text = text.strip()
    return text

def tokenize_text(text):
    # Tokenize Vietnamese text using underthesea
    return " ".join(word_tokenize(text))

def process_article(article):
    if not article.get('content') or not article.get('title'):
        # print("Skipping article - missing content or title")
        return None
    
    # Clean and tokenize the text
    text = clean_text(article['content'])
    title = clean_text(article['title'])
    
    # Get the meta description as summary if available, otherwise use title
    summary = clean_text(article.get('summary_meta', title))
    if summary == title and len(text.split('\n')) > 1:
        # Use first paragraph as summary if no meta description
        summary = clean_text(text.split('\n')[0])
    
    # Remove " - VnExpress" suffix
    summary = re.sub(r'\s*-\s*VnExpress$', '', summary)
    
    # Skip if text or summary is too short
    if len(text.split()) < 50 or len(summary.split()) < 5:
        # print("Skipping article - text too short")
        return None
        
    return {
        'text': tokenize_text(text),
        'summary': tokenize_text(summary),
        'url': article.get('url', '')
    }

def main():
    # Read JSONL file
    articles = []
    # Using relative path as the file is in the current directory
    with open('vnexpress_articles.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                article = json.loads(line)
                processed = process_article(article)
                if processed:
                    articles.append(processed)
            except json.JSONDecodeError:
                continue
    
    # Convert to DataFrame
    df = pd.DataFrame(articles)
    
    # Split into train/val/test
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    # Save processed data
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print(f"Processed {len(df)} articles")
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    main()