#!/usr/bin/env python3

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import helper_vis as hf

# Author: Lakshwin
# When: 21st Oct 2023

# Animate the process of stress diffusion in a single individual

# steps:

# 1. A single frame of the src matrix + a single frame of the target matrix.

def show_mat(mat, kind, cmap):

    # getting past np's pass by ref
    to_show_mat = mat.copy()
    flip_mat = np.rot90(to_show_mat, k=2, axes = (0,1))

    fig, ax = plt.subplots(1, 1, figsize=(5,5), dpi=200)
    ax.axis(False)
    fig.set_facecolor("black")
    ax.set_facecolor("black")
    cmesh = ax.pcolormesh(flip_mat, edgecolors='k', vmin=0, vmax=1, linewidth=2, cmap =cmap)

    # plt.savefig(f"/Users/niwhskal/competency2d/visualisation/single_frames/{kind}.png")
    plt.show()

def show_single_frame(src_mat, tar_mat):
    show_mat(src_mat, "src", plt.cm.coolwarm)
    show_mat(tar_mat, "tar", plt.cm.coolwarm)


# 2. Highlight cells in their correct positions with green + leave incorrect cells with the same color.

def highlight_src(src_mat, tar_mat):
    src = src_mat.copy()
    tar = tar_mat.copy()
    # flip 180 because of pcolormesh
    src = np.rot90(src, k=2, axes = (0,1))
    tar = np.rot90(tar, k=2, axes = (0,1))

    # green: cells in their correct pos, white: 1.0, black: 0.0
    colors = 'black white #44AF69'.split()
    cmap = matplotlib.colors.ListedColormap(colors, name='highl', N = None)

    # set correct pos cellst to idx of 2.0
    src[np.where(src == tar)] = 2.0

    # show mat with custom cmap
    fig, ax = plt.subplots(1, 1, figsize=(5,5), dpi=200)
    ax.axis(False)
    cmesh = ax.pcolormesh(src, edgecolors = 'k', vmin = 0, vmax=2, linewidth=2, cmap=cmap)

    # saving frame
    plt.savefig(os.path.join(path, "movement/src_highlight.png"))
    plt.show()


# 3. Pick a single fixed cell, let it send signals (for a single position) out to other cells based on how far they are.

# implemented in after effects




def main():
    src_mat = np.load(os.path.join(path, "single_frames/src.npy"))
    tar_mat = np.load(os.path.join(path, "single_frames/tar.npy"))

    # show_single_frame(src_mat, tar_mat)
    highlight_src(src_mat, tar_mat)

if __name__ == "__main__":
    path = "/Users/niwhskal/competency2d/visualisation/"
    main()



# with stress-sharing disabled, show that target cells are blocked by fixed cells.
# 4. A chosen target cell responds to the signal by moving to that position

# with stress-sharing enabled, show that cells can move through fixed cells.
# 5. Create a new animation for how a single competency step occurs.


# 6. Create a new video visualizing a single iteration of evolution with stress sharing enabled and disabled
# 7. as evolution unfolds, create high grade fitness graphs based on recorded data



