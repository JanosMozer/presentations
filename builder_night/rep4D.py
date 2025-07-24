
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
    answers_dir = os.path.join(script_dir, 'llm_answers_4') # Corrected to llm_answers_4

    # 1. Read all text files from the directory
    all_lines = []
    labels = [] # We still keep labels for potential future use or debugging, but colors will be from length
    file_paths = glob.glob(os.path.join(answers_dir, '*.txt'))
    
    for file_path in file_paths:
        lines = read_file(file_path)
        all_lines.extend(lines)
        file_basename = os.path.basename(file_path)
        model_name = file_basename.split('_')[0]
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

    # 4. Reduce dimensionality with t-SNE (to 3D)
    print("Reducing dimensionality with t-SNE to 3D...")
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(all_lines) - 1))
    embeddings_3d = tsne.fit_transform(embeddings)

    # Calculate the 4th dimension (sentence length) and normalize for color mapping
    sentence_lengths = np.array([len(line) for line in all_lines])
    # Normalize lengths to a 0-1 range for colormap
    norm_lengths = (sentence_lengths - sentence_lengths.min()) / (sentence_lengths.max() - sentence_lengths.min())

    # Choose a colormap for sentence length (viridis)
    cmap = plt.cm.viridis # Reverted to viridis
    colors_for_plot = cmap(norm_lengths)

    # --- Plot and save 3D combined representation (with hover and rotation) ---
    print("Generating 3D combined plot with 4th dimension as color...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        c=colors_for_plot, # Color mapped to 4th dimension
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

    ax.set_title('Combined 3D Semantic Representation with Sentence Length as Color')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    # We no longer need legend for labels, as color represents the 4th dimension
    # ax.legend()
    ax.grid(True)
    
    output_path = os.path.join(script_dir, 'semantic_representation_combined_4D.png')
    plt.savefig(output_path)
    print(f"Combined 3D plot saved to {output_path}")

    print("Showing interactive 3D plot. Rotate with mouse. Hover for details. Close the window to exit.")
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()

