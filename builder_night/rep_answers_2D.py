
import os
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import glob
import mplcursors # Added for hover effect

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Set the target answers directory here (change to 'llm_answers_2' if needed)
    answers_dir = os.path.join(script_dir, 'llm_answers_2') 

    # 1. Read all text files from the specified directory
    all_lines = []
    labels = []
    file_paths = glob.glob(os.path.join(answers_dir, '*.txt'))
    file_paths.sort() # Ensure consistent ordering for colors
    
    for file_path in file_paths:
        lines = read_file(file_path)
        all_lines.extend(lines)
        # Label by filename (e.g., 'answer_r6.txt')
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

    # 4. Reduce dimensionality with t-SNE (to 2D)
    print("Reducing dimensionality with t-SNE to 2D...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_lines) - 1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Define colors for individual answers
    unique_labels = sorted(list(set(labels)))
    colors = ['black', 'blue', 'red'] # Cycle through these colors
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

    # --- Plot and save individual representations (no hover) ---
    print("Generating individual plots for each answer file...")
    current_start_index = 0
    for i, file_path in enumerate(file_paths):
        file_lines_count = len(read_file(file_path)) # Re-read to get count as all_lines has all of them
        file_label = os.path.basename(file_path)
        file_end_index = current_start_index + file_lines_count

        plt.figure(figsize=(10, 8))
        
        # Get embeddings and labels only for this file
        indices_for_file = list(range(current_start_index, file_end_index))
        plt.scatter(embeddings_2d[indices_for_file, 0], embeddings_2d[indices_for_file, 1],
                    c=color_map[file_label], label=file_label, alpha=0.7)
        
        plt.title(f'2D Semantic Representation of {file_label}')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.grid(True)
        
        base_filename = os.path.splitext(file_label)[0]
        output_filename = f'semantic_representation_{base_filename}.png'
        output_path = os.path.join(script_dir, output_filename)
        
        plt.savefig(output_path)
        plt.close() # Close the figure to free memory
        print(f"Individual plot saved to {output_path}")

        current_start_index = file_end_index

    # --- Plot and save combined representation (with hover) ---
    print("Generating combined 2D plot...")
    fig, ax = plt.subplots(figsize=(12, 10)) # Use fig, ax for mplcursors
    
    scatter_artists = [] # To store scatter plot objects for mplcursors

    for label in unique_labels:
        # Get indices for all lines belonging to this label (individual answer file)
        indices = [j for j, l in enumerate(labels) if l == label]
        scatter = ax.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], c=color_map[label], label=label, alpha=0.7)
        scatter_artists.append(scatter)

    # Apply mplcursors for hover effect
    cursor = mplcursors.cursor(scatter_artists, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(all_lines[sel.index]) # sel.index refers to the index in the original all_lines list

    ax.set_title(f'Combined 2D Semantic Representation for {os.path.basename(answers_dir)}')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend()
    ax.grid(True)
    
    output_path = os.path.join(script_dir, f'semantic_representation_combined_{os.path.basename(answers_dir)}.png')
    plt.savefig(output_path)
    print(f"Combined plot saved to {output_path}")
    
    print("Showing interactive 2D plot. Close the window to exit.")
    plt.show() # Show interactive plot
    plt.close()

if __name__ == '__main__':
    main() 