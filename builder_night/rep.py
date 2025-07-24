
import os
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import glob

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    answers_dir = os.path.join(script_dir, 'llm_answers_2')

    # 1. Read all text files from the directory
    all_lines = []
    labels = []
    file_paths = glob.glob(os.path.join(answers_dir, '*.txt'))
    
    for file_path in file_paths:
        lines = read_file(file_path)
        all_lines.extend(lines)
        labels.extend([os.path.basename(file_path)] * len(lines))

    if not all_lines:
        print(f"No text files found in {answers_dir}")
        return

    # 2. Load a pre-trained model
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Generate embeddings
    print("Generating embeddings for the text...")
    embeddings = model.encode(all_lines, show_progress_bar=True)

    # 4. Reduce dimensionality with t-SNE
    print("Reducing dimensionality with t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_lines) - 1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # 5. Plot the results
    print("Plotting the results...")
    
    unique_labels = sorted(list(set(labels)))
    colors = ['black', 'blue', 'red']
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

    # --- Plot and save individual representations (no hover) ---
    print("Generating individual plots...")
    for label in unique_labels:
        plt.figure(figsize=(10, 8))
        indices = [j for j, l in enumerate(labels) if l == label]
        
        if not indices:
            continue
            
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], c=color_map[label], label=label, alpha=0.7)
        
        plt.title(f'2D Semantic Representation of {label}')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.grid(True)
        
        base_filename = os.path.splitext(label)[0]
        output_filename = f'semantic_representation_{base_filename}.png'
        output_path = os.path.join(script_dir, output_filename)
        
        plt.savefig(output_path)
        plt.close() # Close the figure to free memory
        print(f"Individual plot saved to {output_path}")

    # --- Plot and save combined representation ---
    print("Generating combined plot...")
    plt.figure(figsize=(12, 10))
    
    for label in unique_labels:
        indices = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], c=color_map[label], label=label, alpha=0.7)

    plt.title('Combined 2D Semantic Representation of LLM Answers')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(script_dir, 'semantic_representation_combined.png')
    plt.savefig(output_path)
    print(f"Combined plot saved to {output_path}")
    plt.close()

if __name__ == '__main__':
    main()

