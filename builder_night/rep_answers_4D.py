
import os
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import glob
import mplcursors

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Set the target answers directory here (change to 'llm_answers_2' if needed)
    answers_dir = os.path.join(script_dir, 'llm_answers_1')

    # 1. Read all text files from the specified directory
    all_lines = []
    labels = [] # Labels are kept for consistency but not used for coloring
    file_paths = glob.glob(os.path.join(answers_dir, '*.txt'))
    file_paths.sort() # Ensure consistent ordering for processing
    
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

    # 4. Reduce dimensionality with t-SNE (to 3D)
    print("Reducing dimensionality with t-SNE to 3D...")
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(all_lines) - 1))
    embeddings_3d = tsne.fit_transform(embeddings)

    # Calculate the 4th dimension (sentence length) and normalize for color mapping
    sentence_lengths = np.array([len(line) for line in all_lines])
    # Normalize lengths to a 0-1 range for colormap
    norm_lengths = (sentence_lengths - sentence_lengths.min()) / (sentence_lengths.max() - sentence_lengths.min())

    # Choose a colormap for sentence length (viridis)
    cmap = plt.cm.viridis # Default viridis: lighter for shorter, darker for longer
    colors_for_plot = cmap(norm_lengths)

    # --- Plot and save 3D combined representation (with hover and rotation) ---
    print("Generating 3D combined plot with sentence length as color...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        c=colors_for_plot, # Color mapped to sentence length
        alpha=0.7
    )

    # Create a colorbar for the 4th dimension
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Sentence Length (Normalized)')

    # Apply mplcursors for hover effect
    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(
            f"Length: {sentence_lengths[sel.index]}\n"
            f"Text: {all_lines[sel.index]}"
        )

    ax.set_title(f'Combined 3D Semantic Representation with Sentence Length as Color for {os.path.basename(answers_dir)}')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    # No legend needed for labels as color represents length
    ax.grid(True)
    
    output_path = os.path.join(script_dir, f'semantic_representation_combined_4D_{os.path.basename(answers_dir)}.png')
    plt.savefig(output_path)
    print(f"Combined 3D plot saved to {output_path}")

    print("Showing interactive 3D plot. Rotate with mouse. Hover for details. Close the window to exit.")
    plt.show()
    plt.close()

if __name__ == '__main__':
    main() 