
import os
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import glob
import mplcursors # Re-adding mplcursors

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    answers_dir = os.path.join(script_dir, 'llm_answers_4') # Updated to llm_answers_4

    # 1. Read all text files from the directory
    all_lines = []
    labels = []
    file_paths = glob.glob(os.path.join(answers_dir, '*.txt'))
    
    for file_path in file_paths:
        lines = read_file(file_path)
        all_lines.extend(lines)
        # Extract model name from filename (e.g., 'ChatGPT-4o' from 'ChatGPT-4o_r10.txt')
        file_basename = os.path.basename(file_path)
        model_name = file_basename.split('_')[0] # Assumes model name is before the first underscore
        labels.extend([model_name] * len(lines))

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
        
        # Use original file basename for individual plot filename
        original_file_names = [os.path.basename(p) for p in file_paths if model_name in p] # This logic is tricky, better to pass file_path directly
        # Let's revise the label for individual plots to be the full filename
        current_file_path = [p for p in file_paths if os.path.basename(p).startswith(label)][0] # Assumes unique start for label
        base_filename = os.path.splitext(os.path.basename(current_file_path))[0]
        output_filename = f'semantic_representation_{base_filename}.png'
        output_path = os.path.join(script_dir, output_filename)
        
        plt.savefig(output_path)
        plt.close() # Close the figure to free memory
        print(f"Individual plot saved to {output_path}")

    # --- Plot and save combined representation (with hover) ---
    print("Generating combined plot...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter_artists = [] # To store all scatter plot objects for mplcursors

    for label in unique_labels:
        indices = [j for j, l in enumerate(labels) if l == label]
        scatter = ax.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], c=color_map[label], label=label, alpha=0.7)
        scatter_artists.append(scatter)

    # Apply mplcursors to all scatter plot artists for the combined plot
    cursor = mplcursors.cursor(scatter_artists, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        # sel.index gives the index of the hovered point in the *original* data array
        # which corresponds to the index in `all_lines`
        sel.annotation.set_text(all_lines[sel.index])
        # Removed sel.annotation.arrow_patch.set() to avoid errors

    ax.set_title('Combined 2D Semantic Representation of LLM Answers')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend()
    ax.grid(True)
    
    output_path = os.path.join(script_dir, 'semantic_representation_combined.png')
    plt.savefig(output_path)
    print(f"Combined plot saved to {output_path}")

    print("Showing interactive plot. Close the window to exit.")
    plt.show() # Show interactive plot
    plt.close()

if __name__ == '__main__':
    main()

