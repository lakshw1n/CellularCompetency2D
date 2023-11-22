#!/usr/bin/env python3

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import helper_vis as hf
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Author: Lakshwin
# When: 21st Oct 2023

# Animate the process of stress diffusion in a single individual

# steps:

# 1. A single frame of the src matrix + a single frame of the target matrix.

def show_mat(mat, kind, cmap):

    # getting past np's pass by ref
    to_show_mat = mat.copy()
    # flip_mat = np.rot90(to_show_mat, k=2, axes = (0,1))
    flip_mat = np.flip(to_show_mat, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(5,5), dpi=200)
    ax.axis(False)
    # fig.set_facecolor("black")
    # ax.set_facecolor("black")
    colors = 'black white'.split()
    cmap = matplotlib.colors.ListedColormap(colors, name='b_w', N = None)

    cmesh = ax.pcolormesh(flip_mat, edgecolors='#333333', vmin=0, vmax=1, linewidth=2, cmap =cmap)

    fig.tight_layout()
    plt.savefig(f"/Users/niwhskal/competency2d/visualisation/single_frames/{kind}.png")
    plt.show()

def show_single_frame(src_mat, tar_mat):
    show_mat(src_mat, "src", plt.cm.coolwarm)
    show_mat(tar_mat, "tar", plt.cm.coolwarm)


# 2. Highlight cells in their correct positions with green + leave incorrect cells with the same color.

def highlight_src(src_mat, tar_mat, kind = ""):
    src = src_mat.copy()
    tar = tar_mat.copy()
    # flip 180 because of pcolormesh
    src = np.rot90(src, k=2, axes = (0,1))
    tar = np.rot90(tar, k=2, axes = (0,1))

    # green: cells in their correct pos, white: 1.0, black: 0.0

    colors = 'black white #548C2F'.split()
    cmap = matplotlib.colors.ListedColormap(colors, name='highl', N = None)

    # set correct pos cellst to idx of 2.0
    src[np.where(src == tar)] = 2.0

    # show mat with custom cmap
    fig, ax = plt.subplots(1, 1, figsize=(5,5), dpi=200)
    ax.axis(False)
    cmesh = ax.pcolormesh(src, edgecolors = '#333333', vmin = 0, vmax=2, linewidth=2, cmap=cmap)

    fig.tight_layout()

    # saving frame
    plt.savefig(os.path.join(path, f"movement/src_highlight_{kind}.png"))
    plt.show()

def apply_graded_signal(src_mat, tar_mat):
    src = src_mat.copy()
    src = src.astype(np.float32)

    tar = tar_mat.copy()
    tar = tar.astype(np.float32)

    # flip 180 because of pcolormesh
    # src = np.rot90(src, k=2, axes = (0,1))
    # tar = np.rot90(tar, k=2, axes = (0,1))

    # color fixed cells
    src[np.where(src == tar)] = 3.0

    # get coordinates of white unfixed cells
    x_s, y_s = np.where(src == 1.0)

    # grade the opacity of each coordinate by its distance from the chosen cell
    chosen_cell_coords = (7, 12)

    # store all distances and scale it by max
    all_dist = []

    distance = lambda p1, p2: np.sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 )

    for x,y in zip(x_s, y_s):
        all_dist.append(distance((x,y), chosen_cell_coords))

    graded_dist = all_dist/max(all_dist)
    src[np.where(src == 1.0)] = 1.0 + graded_dist

    src = np.rot90(src, k=2, axes = (0,1))

    cvals  = [0, 1, 2, 3]
    colors = 'black #FC5130 white #548C2F'.split()

    norm = plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    # colors = 'black orange white #44AF69'.split()
    # cmap = matplotlib.colors.ListedColormap(colors, name='highl', N = None)

    fig, ax = plt.subplots(1, 1, figsize=(5,5), dpi=200)
    ax.axis(False)
    cmesh = ax.pcolormesh(src, edgecolors = 'k', linewidth=2, cmap=cmap)

    fig.tight_layout()

    # saving frame
    plt.savefig(os.path.join(path, "movement/graded_signal_src.png"))
    plt.show()


def movement_sharing_disabled(src_mat, tar_mat):
    src = src_mat.copy()
    src = src.astype(np.float32)

    tar = tar_mat.copy()
    tar = tar.astype(np.float32)

    # flip 180 because of pcolormesh
    # src = np.rot90(src, k=2, axes = (0,1))
    # tar = np.rot90(tar, k=2, axes = (0,1))

    # color fixed cells
    src[np.where(src == tar)] = 3.0

    # get coordinates of white unfixed cells
    x_s, y_s = np.where(src == 1.0)

    # grade the opacity of each coordinate by its distance from the chosen cell
    chosen_cell_coords = (7, 12)

    # store all distances and scale it by max
    all_dist = []

    distance = lambda p1, p2: np.sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 )

    for x,y in zip(x_s, y_s):
        all_dist.append(distance((x,y), chosen_cell_coords))

    graded_dist = all_dist/max(all_dist)
    src[np.where(src == 1.0)] = 1.0 + graded_dist

    src_cell_coords = (7, 12)
    tar_cell_coords = [(6, 11), (6, 10)]

    count = 1
    for x_t, y_t in tar_cell_coords:
        temp = src[src_cell_coords[0], src_cell_coords[1]]
        src[src_cell_coords[0], src_cell_coords[1]] = src[x_t, y_t]
        src[x_t, y_t] = temp
        src_cell_coords = (x_t, y_t)

        src = np.rot90(src, k=2, axes = (0,1))

        cvals  = [0, 1, 2, 3]
        colors = 'black #FC5130 white #548C2F'.split()

        norm = plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(norm, cvals), colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

        # colors = 'black orange white #44AF69'.split()
        # cmap = matplotlib.colors.ListedColormap(colors, name='highl', N = None)

        fig, ax = plt.subplots(1, 1, figsize=(5,5), dpi=200)
        ax.axis(False)
        cmesh = ax.pcolormesh(src, edgecolors = '#000', linewidth=2, cmap=cmap)

        fig.tight_layout()
        # saving frame
        plt.savefig(os.path.join(path, f"movement/movement_stressed_off_{count}.png"))

        src = np.rot90(src, k=2, axes = (0,1))
        count += 1
        # plt.show()



# 3. Pick a single fixed cell, let it send signals (for a single position) out to other cells based on how far they are.

# implemented in after effects


# 4. Movement with stress sharing enabled vs disabled

def example_movement():

    colors = 'black #FFF #548C2F #FC5130'.split()

    mat = np.array([[2, 2, 2, 2],
                    [2, 3, 2, 2],
                    [2, 2, 2, 1],
                    [0, 2, 1, 2]])

    # mat = np.rot(mat, )
    mat = np.flip(mat, axis = 0)

    enabled_matrix = mat.copy()
    disabled_matrix = mat.copy()

    cmap = matplotlib.colors.ListedColormap(colors, name='highl', N = None)

    # show mat with custom cmap
    fig, ax = plt.subplots(1, 1, figsize=(3,3), dpi=200)
    ax.axis(False)
    # fig.set_facecolor("black")
    # ax.set_facecolor("black")

    cmesh = ax.pcolormesh(enabled_matrix, edgecolors =  "#333333", vmin = 0, vmax=3, linewidth=3, cmap=cmap)

    # saving frame
    plt.savefig(os.path.join(path, "movement/example_enabled_mat.png"))
    plt.show()

def show_some_srcMatrices(tar, rng, n=6):
    for i in range(n):
        src_temp = hf.scramble(tar, rng)
        highlight_src(src_temp, tar, str(i))


def create_stress_movie(frames):

    cvals  = [0, 1]
    colors = 'black orange'.split()

    norm = plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    fig, ax = plt.subplots(1, 1, figsize=(3,3), dpi=200)
    ax.axis(False)

    cmesh = ax.pcolormesh(np.flip(frames[0], axis = 0), edgecolors =  "#333333", vmin = 0, vmax=2, linewidth=2, cmap=cmap)

    def update(frame):
        cmesh = ax.pcolormesh(np.flip(frame, axis = 0), edgecolors =  "#333333", vmin = 0, vmax=1, linewidth=2, cmap=cmap)

        return cmesh

    ani = FuncAnimation(fig, update, frames=frames)


    # show mat with custom cmap
       # fig.set_facecolor("black")
    # ax.set_facecolor("black")


    # saving frame
    # plt.savefig(os.path.join(path, "movement/example_enabled_mat.png"))

    writervideo = FFMpegWriter(fps=2)
    ani.save('stress_frames_pflagTrue.mp4', writer=writervideo)
    # plt.show()



def competency_process_vis(tar, rng, pflag):
    src = hf.get_init_pop(tar, 1, rng)
    frames = []
    while (hf.fitness(src, tar) != 1.0):
        frames.append(hf.highlight_stressLoc(src[0], tar, pflag))
        src, used_moves, _ = hf.apply_competency(src, tar, 100, rng, 1.0, pflag)
        print(used_moves)

    create_stress_movie(frames)
    # print(frames[10])


def main(rng):
    # src_mat = np.load(os.path.join(path, "single_frames/src.npy"))
    tar_mat = np.load(os.path.join(path, "single_frames/tar.npy"))

    # src_frames = np.load("/Users/niwhskal/competency2d/output/gen_statesFalse.npy", allow_pickle = True)

    # show_single_frame(src_mat, tar_mat)
    # highlight_src(src_mat, tar_mat)
    # apply_graded_signal(src_mat, tar_mat)
    # movement_sharing_disabled(src_mat, tar_mat)
    # example_movement()
    # show_some_srcMatrices(tar_mat, rng)

    competency_process_vis(tar_mat, rng, True)




if __name__ == "__main__":
    rng = np.random.default_rng(234452)
    path = "/Users/niwhskal/competency2d/visualisation/"
    main(rng)



# with stress-sharing disabled, show that target cells are blocked by fixed cells.
# 4. A chosen target cell responds to the signal by moving to that position

# with stress-sharing enabled, show that cells can move through fixed cells.
# 5. Create a new animation for how a single competency step occurs.


# 6. Create a new video visualizing a single iteration of evolution with stress sharing enabled and disabled
# 7. as evolution unfolds, create high grade fitness graphs based on recorded data
