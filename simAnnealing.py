#!/usr/bin/env python3

#author: Lakshwin Shreesha
#date: 9th Sept 2023
#Evolution of 2D stress based competency

import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import helperfunction as hf
import multiprocessing

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
    stringency = 0.1
    N_runs = 5
    N_indv = 100 #n of indviduals
    tar_shape = 100 #25
    n_gen = 2000
    comp_value = int((tar_shape**2)* 0.75)
    pf_Flag = True
    mut_rate = 0.3
    N_mutations = int(np.ceil(tar_shape*0.3))

    plot_dist = True
    src_dir = os.path.join(cwd, f"output/plasticity_{pf_Flag}/")
    plots_path = os.path.join(src_dir, "plots")

    cwd = os.getcwd()
    if not os.path.exists(plots_path):
        os.makedirs(directory)

    p_recalc = 1.0 #increasing this value delays the time it takes to reach max fitness. Eg: a p_recalc probability of 1.0 takes ~300 generations more compared to a p_recalc prob of 0.3 in order to reach max fitness.

    switch_at = 0#round(n_gen/2)

    print(f"Settings:\n pf_flag: {pf_Flag} \nruns: {N_runs} \n n_gen: {n_gen}\ncomp_value: {comp_value}\n Shape: {tar_shape}\nn_indv: {N_indv}\nplot_dist: {plot_dist}\n")

    if (switch_at ==0):
        gen_fname = f"{src_dir}/gen_matrix{pf_Flag}_{tar_shape}.npy"
        phen_fname = f"{src_dir}/phen_matrix{pf_Flag}_{tar_shape}.npy"
        comp_fname = f"{src_dir}/comp_vals{pf_Flag}_{tar_shape}.npy"
        dist_fname = f"{src_dir}/tot_dist{pf_Flag}_{tar_shape}.npy"

        gen_state_fname = ""#f"{src_dir}/gen_states{pf_Flag}_{tar_shape}.npy"
        phen_state_fname = ""#f"{src_dir}/phen_states{pf_Flag}_{tar_shape}.npy"

    else:
        gen_fname = f"{src_dir}/gen_matrix_ax_{tar_shape}.npy"
        phen_fname = f"{src_dir}/phen_matrix_ax_{tar_shape}.npy"
        comp_fname = f"{src_dir}/comp_vals_ax_{tar_shape}.npy"
        dist_fname = f"{src_dir}/tot_dist_ax_{tar_shape}.npy"

        gen_state_fname ="" #f"{src_dir}/gen_states_ax_{tar_shape}.npy"
        phen_state_fname ="" #f"{src_dir}/phen_states_ax_{tar_shape}.npy"

    rng = np.random.default_rng(12345)

    # # target = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #                    [0, 0, 0, 1, 0, 0,0, 0, 0, 0],
    #                     [0, 0, 0, 0, 1, 0,0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 1,0, 0, 0, 0],
    #                      [0, 0, 0, 0, 0, 0,1, 0, 0, 0],[0, 0, 0, 0, 0, 0,0, 1, 0, 0],[0, 0, 0, 0, 0, 0,0, 0, 1, 0], [0, 0, 0, 0, 0, 0,0, 0, 0, 1]])

    # ---
    target = hf.load_from_txt(os.path.join(cwd, "/smiley.png"), tar_shape)
    print(target.shape)

    g_fitnesses = np.zeros((N_runs, n_gen, N_indv))
    phen_fitnesses = np.zeros((N_runs, n_gen, N_indv))
    allCompVals = np.zeros((N_runs, n_gen, N_indv))
    allDistVals = np.zeros((N_runs, n_gen, N_indv))

    allGStates = {}
    allPStates = {}


    tarArgs = [target]*N_runs
    idvArgs = [N_indv]*N_runs
    rngArgs = [rng]*N_runs

    srcArgs = [*zip(tarArgs, idvArgs, rngArgs)]


    loop_start = time.time()
    src_popAll = []
    for i in range(N_runs):
     src_popAll.append(hf.get_init_pop(target, N_indv, rng))

    loop_end = time.time()
    print(f"Time (loop): {loop_end - loop_start} s")

    g = []
    p = []
    cv = []
    dis = []
    g_log = []
    p_log = []

    evolveArgs = [*zip(src_popAll, tarArgs, [n_gen]*N_runs, [comp_value]*N_runs, rngArgs, [pf_Flag]*N_runs, [mut_rate]*N_runs, [N_mutations]*N_runs, idvArgs, [*range(N_runs)], [switch_at]*N_runs, [p_recalc]*N_runs)]

    pool_start = time.time()
    pool = multiprocessing.Pool(os.cpu_count() -1)
    g, p, cv, dis, g_log, p_log = [*zip(*pool.starmap(hf.evolve, iterable=evolveArgs))]
    pool_end = time.time()
    print(f"Time (pool): {pool_end - pool_start} s")

    g = np.array(g)
    p = np.array(p)
    cv = np.array(cv)
    dis = np.array(dis)
    # g_log = np.array(g_log)
    # p_log = np.array(p_log)


     # old code: non-parallelized execution

     # loop_start= time.time()
     # for curr_run in range(N_runs):
     #     src_pop = hf.get_init_pop(target, N_indv, rng)
     #     g, p, cv, dis, g_log, p_log = hf.evolve(src_pop, target, n_gen, comp_value, rng, pf_Flag, mut_rate, N_mutations, N_indv, curr_run, switch_at, p_recalc)

     #     g_fitnesses[curr_run] = g
     #     phen_fitnesses[curr_run] = p
     #     allCompVals[curr_run] = cv
     #     allDistVals[curr_run] = dis

     #     allGStates[curr_run] = g_log
     #     allPStates[curr_run] = p_log

     # loop_end = time.time()
     # print(f"loop time: {loop_end-loop_start} s")

    np.save(gen_fname, g)
    np.save(phen_fname, p)
    np.save(comp_fname, cv)
    np.save(dist_fname, dis)

    np.save(gen_state_fname, g_log[0])
    np.save(phen_state_fname, p_log[0])

    hf.plot(gen_fname, phen_fname, comp_fname, dist_fname, gen_state_fname, phen_state_fname, tar_shape, pf_Flag, comp_value, plot_dist, target, plots_path)


    # run_singleidv_test(target, rng)
    # run_dict_purgetest(rng)

def run_singleidv_test(target, rng):
    src = hf.get_init_pop(target, 100, rng)
    mvs = 0
    run_count = 0
    run_count += 1
    src, mvs, dist = hf.apply_competency(src, target, 1000, rng, 1.0, True)
    print(f"run: {run_count} | moves: {mvs} | dist: {dist} | fitness: {hf.fitness(src, target)}")

def run_dict_purgetest(rng):
    d = {}
    for i in range(20):
        temp = {}
        for j in range(10):
            if (rng.random() <0.5):
                val = 0.0

            else:
                val = rng.random()


            first, sec = rng.integers(10, size = 2)
            temp[tuple([first, sec])] = val

        first_o, sec_o = rng.integers(10, size = 2)

        d[tuple([first_o, sec_o])] = temp

    hf.purge_emptykeys(d)

    print(d)


# def save_scrambled_src(tar, rng):
#     src = hf.scramble(tar, rng)
#     src = src.astype(np.int32)
#     with open('/Users/niwhskal/p5_test.js/src.txt', 'w') as F:
#         for i in range(src.shape[0]):
#             for j in range(src.shape[1]):
#                 F.write(str(int(src[i][j])))
#             F.write('\n');

#     with open('/Users/niwhskal/p5_test.js/tar.txt', 'w') as F:
#         for i in range(tar.shape[0]):
#             for j in range(tar.shape[1]):
#                 F.write(str(int(tar[i][j])))
#             F.write('\n');

#     np.save("/Users/niwhskal/competency2d/visualisation/single_frames/src.npy", src)
#     np.save("/Users/niwhskal/competency2d/visualisation/single_frames/tar.npy", tar)



if (__name__ == "__main__"):

    rng = np.random.default_rng(12345)
    # save_scrambled_src(target, rng)

    main()
