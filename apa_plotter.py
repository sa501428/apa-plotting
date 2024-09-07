"""
APA Heatmap Plotter for Hi-C Data

This script loads an APA matrix from a NumPy (.npy) file, calculates the P2LL score, and generates a heatmap visualization
of the matrix. The heatmap is saved as a PNG file.

Usage:
    python apa_plotter.py [input_file.npy]

Arguments:
    input_file.npy: The path to the .npy file containing the APA matrix data.

Output:
    Saves the heatmap plot as a PNG file replacing the '.npy' extension with '.png' in the input file's name.

Requirements:
    numpy, matplotlib
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('agg')  # Set the backend before importing pyplot for non-GUI environments
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set default font for plots
plt.rcParams['font.family'] = 'Arial'

# Define a red colormap for heatmaps
REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1, 1, 1), (1, 0, 0)])

def get_score(matrix):
    """Calculate the score based on the center value and the lower left corner mean."""
    rows = matrix.shape[0]
    buffer = rows // 4
    color_limit = np.max(matrix)
    center = matrix[rows // 2, rows // 2]
    lower_left_mean = np.mean(matrix[-buffer:, :buffer])
    score = center / lower_left_mean
    return score, color_limit

def plot_hic_map(dense_matrix, ax):
    """Plot heatmap of the matrix with an annotated score."""
    score, color_limit = get_score(dense_matrix)
    ax.matshow(dense_matrix, cmap=REDMAP, vmin=0, vmax=color_limit)
    ax.xaxis.set_tick_params(labelbottom=False)  # Turn off x-axis labels
    ax.yaxis.set_tick_params(labelleft=False)  # Turn off y-axis labels
    ax.set_title(f'APA score = {score:.2f}')

def main():
    if len(sys.argv) != 2:
        print("Usage: python apa_plotter.py [input_file.npy]")
        sys.exit(1)

    # Load data and set up figure for plotting
    data = np.load(sys.argv[1])
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_hic_map(data, ax)

    # Save the figure and close
    output_location = sys.argv[1].replace(".npy", '.png')
    plt.savefig(output_location, format='png')
    plt.close(fig)

if __name__ == '__main__':
    main()
