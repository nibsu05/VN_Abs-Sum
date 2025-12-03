import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create assets directory if not exists
if not os.path.exists('assets'):
    os.makedirs('assets')

# Load data
try:
    df = pd.read_csv('data/processed/train.csv')
    
    # Calculate lengths
    df['text_len'] = df['text'].str.len()
    df['summary_len'] = df['summary'].str.len()

    # Set style
    sns.set_theme(style="whitegrid")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Text Length Distribution
    sns.histplot(data=df, x='text_len', bins=50, kde=True, color='skyblue', ax=axes[0])
    axes[0].set_title('Distribution of Original Text Lengths')
    axes[0].set_xlabel('Length (characters)')
    axes[0].set_ylabel('Count')
    
    # Plot Summary Length Distribution
    sns.histplot(data=df, x='summary_len', bins=50, kde=True, color='orange', ax=axes[1])
    axes[1].set_title('Distribution of Summary Lengths')
    axes[1].set_xlabel('Length (characters)')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('assets/data_distribution.png')
    print("Successfully generated assets/data_distribution.png")
    
except Exception as e:
    print(f"Error generating visualization: {e}")
    # Create a placeholder image if data is missing
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, 'Data Visualization Placeholder\n(Run preprocessing to generate data)', 
             ha='center', va='center', fontsize=20)
    plt.savefig('assets/data_distribution.png')
