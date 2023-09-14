#!/usr/bin/env python3

#author: Lakshwin Shreesha
#date: 9th Sept 2023
#Evolution of 2D stress based competency

import numpy as np
import cv2
import matplotlib.pyplot as plt
import helperfunction as hf

# Methodology:
# Given a scrambled matrix, get the stress of each of its cells.
# The stress = the difference of cell value between the src and the target at any given position
# Until fitness = max (i.e, src == target), cells which have a stress of 0, send out global signals requesting appropriate neighbors.
# stressed cells closest to the signal respond by moving towards the indicated position.
# stressed cells cannot move through fixed cells (i.e, cells with stress = 0)

# Setup 2:
# Each stressed cell tells its neighbor that it is stressed and requests passage, making the overall structure more plastic.


# Functions:
# Initialize a target from a file/set it manually,
# initialize_src(tar, n_indv) -> inp: target, n_individuals | out: n_indv scrambled copies of the target
# evolve() pseudocode:
#   until max_generations:
#       genotypic_fitness_calc()
#       apply_competency()
#       selection(comp_population, genotypic_fitness)
#       mutation(selected_pop)

# apply_competency() pseudocode
#for each_matrix:
# while !max_n_swaps or !no_stressed_cells:
#       get_stress()
#       required_neighbors() : Each fixed cell (0 stress) identifies its stressed neighbor and calculates the "what cell do I need there?" i.e, [(indx, req_cell)] array.

#       send_gradient_signal(): Each fixed cell sends out a signal of what it needs through a graded signal (a gaussian spread)
#       incentive_to_move(): each stressed cell receives a degraded signal-value from multiple un-stressed cells.

#       summate_receivers(): For cells which match the requirement, signals to move to
#       the same position add up. Eg: If two neighboring fixed cells send out a signal requiring a "1" to move to position (0,1) then each of the stressed 1's in the #       grid receives and adds them up.

#       move(): For each signalled position, the stressed cell with the highest sum will move towards it by swapping with other stressed #       cells in
#       between. If a cell finds itself surrounded by fixed cells, then consider it a special category of fixed-but-no-signal cell. Put it in an exclusion list and ignore
#       all computations on it.

#       move fn variation: if plastic_flag = True, then stressed cells which are surrounded by non-stressed cells can swap with stressed_cells until they reach their goal pos.

#       update swap_counter

def main():
    # target = get_target_from_file()
    stringency = 0.1
    N_indv = 100 #n of indviduals
    n_gen = 50
    comp_value = 100
    pf_Flag = True
    mut_rate = 0.1
    N_mutations = 2 #this is sketchy, change this based on 2d grid shape

    gen_fname = "/Users/niwhskal/competency2d/output/gen_matrix.npy"
    phen_fname = "/Users/niwhskal/competency2d/output/phen_matrix.npy"
    comp_fname = "/Users/niwhskal/competency2d/output/comp_vals.npy"

    rng = np.random.default_rng(12345)

    # # target = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #                   [0, 0, 0, 1, 0, 0,0, 0, 0, 0],
    #                    [0, 0, 0, 0, 1, 0,0, 0, 0, 0],
    #                    [0, 0, 0, 0, 0, 1,0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0,1, 0, 0, 0],[0, 0, 0, 0, 0, 0,0, 1, 0, 0],[0, 0, 0, 0, 0, 0,0, 0, 1, 0], [0, 0, 0, 0, 0, 0,0, 0, 0, 1]])
    target = hf.load_from_txt("/Users/niwhskal/Downloads/MNIST_6_0.png")
    print(target.shape)

    src_pop = hf.get_init_pop(target, N_indv, rng)
    hf.evolve(src_pop, target, n_gen, comp_value, rng, pf_Flag, mut_rate, N_mutations, N_indv)
    hf.plot(gen_fname, phen_fname, comp_fname, pf_Flag)

if (__name__ == "__main__"):
    main()

