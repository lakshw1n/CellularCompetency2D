#!/usr/bin/env python3

#author: Lakshwin Shreesha
#date: 9/09/2023
#library functions for the stress based competency model
import os
import numpy as np
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib import animation
import time
import multiprocessing
import cv2

def load_from_txt(fname, tar_shape):
    im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (tar_shape, tar_shape), interpolation = cv2.INTER_AREA)
    im = np.round(im/255.0)
    # cv2.imshow("test", (im*255.0).astype(np.uint8))
    # cv2.waitKey(0)
    return im


def scramble(tar, rng):
    if tar.shape[0]*tar.shape[1] == 0:
        raise Exception("target empty\n")

    temp = tar.copy()
    rng.shuffle(temp, axis = 0)
    rng.shuffle(temp, axis = 1)

    return temp

def get_init_pop(target, n, rng):
    if target.shape[0]*target.shape[1] == 0:
        raise Exception("target empty\n")

    if n <=0:
        raise Exception("n cannot be empty")

    pop = [scramble(target, rng) for _ in range(n)]
    pop_array = np.array(pop).reshape(n, target.shape[0], target.shape[1])

    return pop_array

def fitness(src, tar):
    if tar.shape[0]*tar.shape[1] == 0 or src.shape[0]*src.shape[1] == 0:
        raise Exception("input matrices cannot be empty\n")

    return 1.0 - np.mean((src-tar)**2)

def get_stress(src, tar):
    return np.abs(src-tar)

def get_cellId_from_direction(stress, tar, curr_loc, dir):

    if (dir>=8):
        raise Exception("Direction cannot be >7\n")

    i, j = curr_loc
    if (dir == 0): #north
        row_pos = i-1
        if (row_pos <0):
            return -1, -1
        col_pos = j
        new_idx = (row_pos, col_pos)
        return new_idx, tar[new_idx]

    elif (dir == 1): #south
        row_pos = i+1
        if (row_pos >= tar.shape[0]):
            return -1, -1
        col_pos = j
        new_idx = (row_pos, col_pos)
        return new_idx, tar[new_idx]

    elif (dir == 2): #west
        row_pos = i
        col_pos = j-1
        if (col_pos <0):
            return -1, -1

        new_idx = (row_pos, col_pos)
        return new_idx, tar[new_idx]

    elif (dir == 3): #east
        row_pos = i
        col_pos = j+1
        if (col_pos >=tar.shape[1]):
            return -1, -1

        new_idx = (row_pos, col_pos)
        return new_idx, tar[new_idx]

    elif (dir == 4): #nw
        row_pos = i-1
        col_pos = j-1
        if (col_pos <0 or row_pos <0):
            return -1, -1

        new_idx = (row_pos, col_pos)
        return new_idx, tar[new_idx]

    elif (dir == 5): #ne
        row_pos = i-1
        col_pos = j+1
        if (col_pos >=tar.shape[1] or row_pos <0):
            return -1, -1

        new_idx = (row_pos, col_pos)
        return new_idx, tar[new_idx]

    elif (dir == 6): #sw
        row_pos = i+1
        col_pos = j-1
        if (col_pos <0 or row_pos >=tar.shape[0]):
            return -1, -1

        new_idx = (row_pos, col_pos)
        return new_idx, tar[new_idx]

    elif (dir == 7): #se
        row_pos = i+1
        col_pos = j+1
        if (col_pos >=tar.shape[1] or row_pos >=tar.shape[0]):
            return -1, -1

        new_idx = (row_pos, col_pos)
        return new_idx, tar[new_idx]


def indv_cell_requirement(idx, src, tar, stress):
    if (idx[0]<0 or idx[1]<0):
        raise Exception("indexes must be positive\n")
    if (idx[0]>src.shape[0] or idx[1]>src.shape[1]):
        raise Exception("indexes cannot be greater than src shape\n")

    if (src.shape != tar.shape):
        raise Exception("src and tar must be of the same shape\n")

    # each cell has to look in eight different directions

    #look up
    neighbor_list = []
    for dir in range(8):
        to_be_idx, to_be_neighbor = get_cellId_from_direction(stress, tar, idx, dir)
        if (to_be_neighbor == -1):
            continue

        if (stress[to_be_idx]<=0):
            continue

        neighbor_list.append((to_be_idx, to_be_neighbor))

    return neighbor_list


def get_required_neighbors(src, tar, stress):
    # get indexes of fixed cells
    loc_x, loc_y = np.where(stress == 0)
    fixed_idxs = list(zip(loc_x, loc_y))

    # for each fixed idx, get neighbor list
    neighbor_needs = {}
    for idx in fixed_idxs:
        neighbor_needs[idx] = indv_cell_requirement(idx, src, tar, stress)

    return neighbor_needs

def send_graded_signal(neighbor_requirement, src, stress):

    # get stressed positions
    x,y = np.where(stress !=0)

    vote_dict = {(i,j): 0.0 for i,j in zip(x,y)}
    stressed_dict = {(i,j): vote_dict for i,j in zip(x,y)}

    A = 1.0
    sigmas = np.array([3*src.shape[0]//6, 3*src.shape[1]//6])
    gaussian_signal = lambda X, mu, sigma : A*np.exp(-((((x[0]-mu[0])**2)/(2*sigma[0]**2)) + (((x[1]-mu[1])**2)/(2*sigma[1]**2))))

    # this is a terrible way to do it.

    for fixed_idx, all_signals in neighbor_requirement.items():
        for to_be_idx, neighbor_kind in all_signals:
            #each fixed cell spreads its signal to its required stressed-cell

            for k, v in stressed_dict.items():
                # in each stressed cell, first check if it is of the appropriate kind
                if (src[k] == neighbor_kind):
                    # then, degrade the signal and spread it to the required to-be-cell position counter. i.e: each stressed cell-kind can be required to move to multiple positions, keep a counter to track signal magnitude in each
                    #careful: dicts are mutable !, always modify a copy and re-initialize
                    temp_value = v.copy()
                    temp_value[to_be_idx] += gaussian_signal(k, fixed_idx, sigmas)
                    stressed_dict[k] = temp_value

                else:
                    continue

    return stressed_dict


#def calc_moves(from_pos, to_pos):
#    #check if to-position is below from-pos or vice-versa
#    f = from_pos.copy()
#    t = to_pos.copy()

#    moves = 0
#    while (f[0]!=t[0] or f[1]!=t[1]):
#        if (t[0]>f[0] or t[1]>f[1]):

#            #move diagnonally
#            if (t[0]>f[0] and t[1]>f[1]):
#                f[0] += 1
#                f[1] += 1

#            #move down
#            elif(t[0]>f[0]):
#                f[0] += 1

#            #move right
#            elif (t[1]>f[1]):
#                f[1] += 1


#        #if to-position is above from-position
#        elif (t[0]<f[0] or t[1]<f[1]):

#            #move diagnonally
#            if (t[0]<f[0] and t[1]<f[1]):
#                f[0] -= 1
#                f[1] -= 1

#            #move up
#            elif(t[0]<f[0]):
#                f[0] -= 1

#            #move left
#            elif (t[1]<f[1]):
#                f[1] -= 1

#        moves += 1

#    return moves

def delete_pos(stressed_dict, from_pos, to_pos, hard_delete = 1):
    if len(stressed_dict)==0:
        return 0

    #if plasticity flag is false then hard_delete = 0, then, go to the from_pos key and delete its to pos value and return

    # a way to delete a single key from stressed_dict
    if (hard_delete == 2):
        try:
            del stressed_dict[tuple(from_pos)]
        except KeyError:
            pass
        return 0

    elif (hard_delete == 0):
        try:
            del stressed_dict[tuple(from_pos)][tuple(to_pos)]
        except KeyError:
            pass
        return 0

    elif (hard_delete == 1):
        try:
            del stressed_dict[tuple(from_pos)]
            del stressed_dict[tuple(to_pos)]
        except KeyError:
            pass

    if len(stressed_dict) ==0:
        return 0

    # delete postition entries of already swapped cells from all dict values.
    for k, v in stressed_dict.items():
        try:
            del v[tuple(from_pos)]
            del v[tuple(to_pos)]
        except KeyError:
            continue

    return 0

def swap(indv, from_pos, to_pos):
    temp = indv[tuple(from_pos)]
    indv[tuple(from_pos)] = indv[tuple(to_pos)]
    indv[tuple(to_pos)] = temp

def purge_emptykeys(stressed_dict):

    outer_key_del = []
    inner_key_del = []

    for from_k, from_v in stressed_dict.items():
        if (len(from_v.keys()) ==0):
            outer_key_del.append(tuple(from_k))
            continue
        for k_inner, v_inner in from_v.items():
            if (v_inner == 0.0):
                inner_key_del.append({from_k: k_inner})


    for d_p in outer_key_del:
        delete_pos(stressed_dict, d_p, -100, hard_delete = 2)

    for i in inner_key_del:
        fr_k = tuple(i.keys())[0]
        to_k = tuple(i.values())[0]
        delete_pos(stressed_dict, fr_k, to_k, hard_delete = 0)

    #make sure those from_keys with no elements are again checked and deleted

    del_again = []
    for from_k, from_v in stressed_dict.items():
        if (len(from_v) == 0):
            del_again.append(tuple(from_k))

    for i in del_again:
        delete_pos(stressed_dict, i, -100, hard_delete = 2)

def move(stressed_dict, src, tar, stress, comp_value, plasticity_flag, rng, current_move_count, current_dist_count):
    #for each stressed cell find the maximum to_be_idx signal value
    #move once for each position

    distance = lambda f,t: np.sqrt((f[0]-t[0])**2 + (f[1] - t[1])**2)
    moves = current_move_count #so that we can break as soon as mvs hits comp_value
    tot_distance = current_dist_count
    break_flag = False

    while(len(stressed_dict) and moves < comp_value):

        # purge those cells with empty values of with a value of 0.0
        purge_emptykeys(stressed_dict)

        if (len(stressed_dict) == 0):
            break_flag = True
            break

        #pick a random stressed position
        kys = list(stressed_dict.keys())
        from_pos = rng.choice(kys)

        v = stressed_dict[tuple(from_pos)]

        max_pos = np.argmax(v.values())
        to_pos = list(v.keys())[max_pos]

        from_pos = np.array(from_pos)
        to_pos = np.array(to_pos)

        # if a from_key contains a to_key of the same indexes, delete it
        if(tuple(from_pos) == tuple(to_pos)):
            delete_pos(stressed_dict, from_pos, to_pos, hard_delete = 0)
            continue

        #check if any other cell in stressed_dict needs to move to from_pos
        #if so, swap, and delete their entries in stressed_dict
        d = distance(from_pos, to_pos)
        if (d<=np.sqrt(2)):

            moves += 1 #direct swap with a neighbor contributing just 1 unit
            tot_distance += d

            #make sure the move counter does not exceed the competency value
            if (moves > comp_value):
                moves -=1
                tot_distance -= d
                break_flag = True
                break

            #then swap can occur
            swap(src, from_pos, to_pos)
            #delete their entries in the dict
            delete_pos(stressed_dict, from_pos, to_pos)

            # print("---"*5)
            # print(stressed_dict)

        else:
            if (plasticity_flag ==True):
                #if not, then move only if plasticity is True
                temp_moves_counter = np.floor(d) + (np.floor(d) - 1) #floor of distance for movement in one direction(approx) + (that distance -1 for movement of the other cell in the opposite direction)
                moves += temp_moves_counter
                tot_distance += d

                #make sure the move counter does not exceed the competency value
                if (moves > comp_value):
                    moves -= temp_moves_counter
                    tot_distance -= d
                    break_flag = True
                    break

                swap(src, from_pos, to_pos)
                #remove their scores from every stressed_cell
                delete_pos(stressed_dict, from_pos, to_pos)
            else:
                # print("stuck")
                delete_pos(stressed_dict, from_pos, to_pos, hard_delete = 0) #no way you can swap, so might as well delete the to_pos from the from_pos

    return moves, tot_distance, break_flag

def remove_faroffcells(stressed_dict):

    distance = lambda f,t: np.sqrt((f[0]-t[0])**2 + (f[1] - t[1])**2)
    to_del = []

    to_del_inner = []

    for from_key, v in stressed_dict.items():
        for to_key, v_inner in v.items():
            if (distance(from_key, to_key) > np.sqrt(2)):
                # delete_pos(stressed_dict, from_key, to_key, hard_delete = 0)
                to_del_inner.append([tuple(from_key), tuple(to_key)])


    for src_del_inner, tar_del_inner in to_del_inner:
        delete_pos(stressed_dict, src_del_inner, tar_del_inner, hard_delete = 0)

    for from_key, v in stressed_dict.items():
        #if its keys are empty delete the entry altogether
        if (len(stressed_dict[tuple(from_key)].values()) == 0):
            to_del.append(tuple(from_key))

    for val_to_del in to_del:
        delete_pos(stressed_dict, val_to_del, -100, hard_delete = 2)

def apply_competency_single_idv(src, tar, comp_value, plasticity_flag, rng):

    overall_mvs = 0
    overall_dist = 0
    stressed_dict_copy = [-9999] #initialize with a single element so that the while loop runs atleast once
    while ((overall_mvs < comp_value) and (len(stressed_dict_copy))):
        stress = get_stress(src, tar)
        neighbor_requirement = get_required_neighbors(src, tar, stress)
        stressed_dict = send_graded_signal(neighbor_requirement, src, stress)

        # delete cells with empty values or those with a value of 0.0
        purge_emptykeys(stressed_dict)

        if (plasticity_flag == False):
            # make sure only nearby cells exist
            remove_faroffcells(stressed_dict)
            # in case there's none, then break
            if (len(stressed_dict) == 0):
                break

        stressed_dict_copy = stressed_dict.copy()

        mvs, tot_dist, break_flag = move(stressed_dict, src, tar, stress, comp_value, plasticity_flag, rng, current_move_count = overall_mvs, current_dist_count = overall_dist)
        overall_mvs = mvs
        overall_dist = tot_dist
        if (break_flag):
            break

    return overall_mvs, overall_dist


def apply_competency(src_pop_main, tar, comp_value, rng, p_recalc, plasticity_flag):
    src_pop = src_pop_main.copy()
    used_moves = []
    distance_log = []
    if src_pop.shape[0] == 0:
        raise Exception("src population cannot be empty\n")

    if comp_value <0:
        raise Exception("competency must be a positive integer\n")

    #parallelize
    src_pop_size = src_pop.shape[0]

    pool_start = time.time()
    evolveArgs = [*zip(src_pop, [tar]*src_pop_size, [comp_value]*src_pop_size, [plasticity_flag]*src_pop_size, [rng]*src_pop_size)]
    pool = multiprocessing.Pool(os.cpu_count() - 1)
    used_moves, distance_log = [*zip(*pool.starmap(apply_competency_single_idv, iterable=evolveArgs))]
    pool_end = time.time()
    print(f"Time (pool): {pool_end - pool_start} s")

    #non-parallel execution
    # for curr_n, src in enumerate(src_pop):
    #     print("Idv: " + str(curr_n) +"/"+str(src_pop.shape[0]))

    #     overall_mvs = 0
    #     overall_dist = 0
    #     stressed_dict_copy = [-9999] #initialize with a single element so that the while loop runs atleast once
    #     while ((overall_mvs < comp_value) and (len(stressed_dict_copy))):
    #         stress = get_stress(src, tar)
    #         neighbor_requirement = get_required_neighbors(src, tar, stress)
    #         stressed_dict = send_graded_signal(neighbor_requirement, src, stress)

    #         # delete cells with empty values or those with a value of 0.0
    #         purge_emptykeys(stressed_dict)

    #         if (plasticity_flag == False):
    #             # make sure only nearby cells exist
    #             remove_faroffcells(stressed_dict)
    #             # in case there's none, then break
    #             if (len(stressed_dict) == 0):
    #                 break

    #         stressed_dict_copy = stressed_dict.copy()

    #         mvs, tot_dist, break_flag = move(stressed_dict, src, tar, stress, comp_value, plasticity_flag, rng, current_move_count = overall_mvs, current_dist_count = overall_dist)
    #         overall_mvs = mvs
    #         overall_dist = tot_dist
    #         if (break_flag):
    #             break

    #     used_moves.append(overall_mvs)
    #     distance_log.append(overall_dist)

    return src_pop, used_moves, distance_log

def selection(src_pop, phen_fitness, N, stringency = 0.1):

    fit_index = {k: i for k, i in enumerate(phen_fitness)}
    fitness_organisms = {k: v for k, v in sorted(fit_index.items(), key=lambda item: item[1], reverse=True)}
    orgs_keys = [k for k,v in fitness_organisms.items()]
    orgs_vals = list(fitness_organisms.values())

    new_orgs_keys = orgs_keys[: round(stringency*N)]
    new_orgs_vals = orgs_vals[: round(stringency*N)]

    new_orgs = [src_pop[j] for j in new_orgs_keys]

    return np.array(new_orgs).reshape(-1, src_pop[0].shape[0], src_pop[1].shape[1])

def mutate_pop(new_pop, mut_rate, N_mutations, N_indv, rng):
    sel_pop = new_pop.copy()
    to_pick_ids = np.arange(sel_pop.shape[0]) #so that you pick only from parents
    while(sel_pop.shape[0] < N_indv):
        #pick a random parent
        r_idx = rng.choice(to_pick_ids)
        new_idv = sel_pop[r_idx].copy()

        # random swap
        from_pos = rng.choice(new_idv.shape[0], 2)
        to_pos = rng.choice(new_idv.shape[0], 2)
        swap(new_idv, from_pos, to_pos)

        sel_pop = np.append(sel_pop, new_idv.reshape(1, new_idv.shape[0], new_idv.shape[1]), axis = 0)
    return sel_pop

def point_mutate(pop, mut_rate, N_mut, N, rng):
    mut_pop = pop.copy()
    for idv in mut_pop:
        if (rng.random() <= mut_rate):
            from_idx1 = rng.choice(idv.shape[0], N_mut)
            from_idx2 = rng.choice(idv.shape[0], N_mut)

            from_idx_list = list(zip(from_idx1, from_idx2))

            to_idx1 = rng.choice(idv.shape[0], N_mut)
            to_idx2 = rng.choice(idv.shape[0], N_mut)

            to_idx_list = list(zip(to_idx1, to_idx2))

            for j in range(N_mut):
                first_pos = from_idx_list[j]
                sec_pos = to_idx_list[j]
                swap(idv, first_pos, sec_pos)

    return mut_pop


def evolve(src_pop, tar, n_gen, comp_value, rng, pflag, mut_rate, N_mut, N, run_num, switch_at, p_recalc):
    gen_matrix = np.zeros((n_gen, src_pop.shape[0]))
    phen_matrix = np.zeros((n_gen, src_pop.shape[0]))

    comp_vals = np.zeros((n_gen, src_pop.shape[0]))
    dist_log = np.zeros((n_gen, src_pop.shape[0]))
    gen_state_log = {}
    phen_state_log = {}

    if (pflag == "hw"):
        comp_value = 0
        print("HW run, setting competency to 0")



    print(f"pflag: {pflag}")
    for i in range(n_gen):
        start_time = time.time()

        genotypic_fitness = [fitness(src, tar) for src in src_pop]
        gen_state_log[i] = src_pop.copy()

        if (i == switch_at and switch_at >0):
            if (pflag == True):
                pflag = False

            elif (pflag == False):
                pflag = True

            else:
                pflag = "hw"

            print(f"Switched pflag to: {pflag}")


        mod_pop, used_moves, tot_dist = apply_competency(src_pop, tar, comp_value, rng, p_recalc, plasticity_flag= pflag)

        #parallelize
        phenotypic_fitness = [fitness(src_m, tar) for src_m in mod_pop]
        phen_state_log[i] = mod_pop.copy()

        #the best indv is the one who boosts his fitness the most
        best_indv_idx = np.argmax(phenotypic_fitness)

        sel_pop = selection(src_pop, phenotypic_fitness, N)
        mut_pop = mutate_pop(sel_pop, mut_rate, N_mut, N, rng)
        src_pop = point_mutate(mut_pop, mut_rate, N_mut, N, rng)

        gen_matrix[i] = genotypic_fitness
        phen_matrix[i] = phenotypic_fitness
        comp_vals[i] = list(used_moves)
        dist_log[i] = list(tot_dist)

        end_time = time.time()
        print(f"Last iteration took: {end_time-start_time}s")

        print(f" Run: {run_num} | Gen: {i} | gen_ftn: {genotypic_fitness[best_indv_idx]}| p_ftn: {phenotypic_fitness[best_indv_idx]} | used comp: {used_moves[best_indv_idx]} | dist: {tot_dist[best_indv_idx]} | pop_dist(avg): {np.mean(tot_dist)}")

    return gen_matrix, phen_matrix, comp_vals, dist_log, gen_state_log, phen_state_log


def plot(genf, phenf, compf, dist_fname, gen_state_fname, phen_state_fname, tar_shape, pflag, max_allowed_phen, plot_dist, target, save_path):

    gen = np.load(genf)
    phen = np.load(phenf)
    comp_list = np.load(compf)
    dist_list = np.load(dist_fname)

    # gen_sts = np.load(gen_state_fname, allow_pickle = True)
    # phen_sts = np.load(phen_state_fname, allow_pickle = True)

    #zoom into higher fitness values
    n = 4 # stay b/w 3 and 4
    fitness_mod = lambda f: (9**f)/9.0 #-np.log(1 + 10**(-n) - f) / (np.log(10)*n)

    #create_movie(gen[0], phen[0], comp_list[0], gen_sts.item(), phen_sts.item(), tar_shape, pflag, save_path)
    # create_highletedFrameMovie(gen[0], phen[0], comp_list[0], gen_sts.item(), phen_sts.item(), tar_shape, pflag, target, save_path)

    n_runs = gen.shape[0]
    n_gen = gen.shape[1]

    gen_plot = np.zeros((n_runs, n_gen))
    phen_plot = np.zeros((n_runs, n_gen))
    cv_plot = np.zeros((n_runs, n_gen))
    dist_plot = np.zeros((n_runs, n_gen))

    min_comp_valPlot = np.zeros((n_runs, n_gen))
    max_comp_valPlot = np.zeros((n_runs, n_gen))

    for run_num in range(n_runs):

        max_idxs = [np.argmax(phen[run_num, i]) for i in range(n_gen)]

        gen_plot[run_num] =  [fitness_mod(gen[run_num, i][max_idxs[i]]) for i in range(n_gen)]
        phen_plot[run_num] = [fitness_mod(phen[run_num, i][max_idxs[i]]) for i in range(n_gen)]
        cv_plot[run_num] = [comp_list[run_num, i][max_idxs[i]] for i in range(n_gen)]
        dist_plot[run_num] = [dist_list[run_num, i][max_idxs[i]] for i in range(n_gen)]

        min_comp_valPlot[run_num] = [np.min(comp_list[run_num, i]) for i in range(n_gen)]
        max_comp_valPlot[run_num] = [np.max(comp_list[run_num, i]) for i in range(n_gen)]

    #mean over runs for fitness and competency

    gen_plot_means = np.mean(gen_plot, axis = 0)
    phen_plot_means = np.mean(phen_plot, axis = 0)

    #variances for fitness
    gen_plot_var = np.std(gen_plot, axis=0)
    phen_plot_var = np.std(phen_plot, axis=0)

    #-----

    # comp val of the best individual averaged over all runs
    comp_plot_means = np.mean(cv_plot, axis = 0)

    #min comp vals averaged over all runs
    min_comp_means = np.mean(min_comp_valPlot, axis = 0)

    #max comp vals averaged over all runs
    max_comp_means = np.mean(max_comp_valPlot, axis = 0)

    #----

    #distance of the best individual
    dist_plot_mean = np.mean(dist_plot, axis = 0)

    if (plot_dist == True):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle(f"Stress Sharing:{pflag}")

        # fig.suptitle(f"Hard wired")

        ax1.plot(gen_plot_means, label = "Genotypic Fitness")
        ax1.fill_between(range(len(gen_plot_means)), gen_plot_means-gen_plot_var, gen_plot_means+gen_plot_var, alpha = 0.3)

        ax1.plot(phen_plot_means, label = "Phenotypic Fitness")
        ax1.fill_between(range(len(phen_plot_means)), phen_plot_means-phen_plot_var, phen_plot_means+phen_plot_var, alpha = 0.3)
        ax1.set_ylim([np.min(gen_plot_means)-0.1, np.max(phen_plot_means)+0.02])
        # custom_ticks = np.append(np.arange(0.6, 0.95, 0.05), np.arange(0.95, 1, 0.01))
        # ax1.set_yticks(custom_ticks)

        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Max Fitness")

        ax2.plot(comp_plot_means, label = "Competency value")
        ax2.fill_between(range(len(comp_plot_means)), min_comp_means, max_comp_means, alpha = 0.3)
        # ax2.set_ylim([0, max_allowed_phen])
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Max swaps used")

        ax3.plot(dist_plot_mean, label = "total cell movement")
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Distance travelled(Best Individual)")
        # ax3.set_title(f"Stress_sharing: {pflag}")

        ax1.legend()
        ax2.legend()
        ax3.legend()

    else:


        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f"Stress Sharing:{pflag}")

        ax1.plot(gen_plot_means, label = "Genotypic Fitness")
        ax1.fill_between(range(len(gen_plot_means)), gen_plot_means-gen_plot_var, gen_plot_means+gen_plot_var, alpha = 0.3)

        ax1.plot(phen_plot_means, label = "Phenotypic Fitness")
        ax1.fill_between(range(len(phen_plot_means)), phen_plot_means-phen_plot_var, phen_plot_means+phen_plot_var, alpha = 0.3)
        ax1.set_ylim([0.65, 1.0])

        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Max Fitness")

        print(min_comp_means)
        print(comp_plot_means)
        print(max_comp_means)

        ax2.plot(comp_plot_means, label = "Competency value")
        ax2.fill_between(range(len(comp_plot_means)), min_comp_means, max_comp_means, alpha = 0.3)
        # ax2.set_ylim([0, max_allowed_phen])
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Max swaps used")

        ax1.legend()
        ax2.legend()

    plt.savefig(os.path.join(save_path, f"main_plot{pflag}.png"))
    plt.show()


def create_movie(gen_fitness, phen_fitness, comp_list, gen_states, phen_states, tar_shape, pflag, save_path):

    max_idxs = [np.argmax(phen_fitness[i]) for i in range(phen_fitness.shape[0])]

    max_gen = [gen_fitness[i][max_idxs[i]]for i in range(gen_fitness.shape[0])]
    max_phen = [phen_fitness[i][max_idxs[i]] for i in range(phen_fitness.shape[0])]
    comp_vals = [comp_list[i][max_idxs[i]] for i in range(comp_list.shape[0])]

    all_gen_states = np.array(list(gen_states.values())).reshape(phen_fitness.shape[0], -1, tar_shape, tar_shape)
    all_phen_states = np.array(list(phen_states.values())).reshape(phen_fitness.shape[0], -1, tar_shape, tar_shape)

    # out_gen = cv2.VideoWriter(os.path.join(save_path, f"genotypes_{pflag}.avi"),cv2.VideoWriter_fourcc(*'DIVX'), 15, (tar_shape, tar_shape), 0)
    # out_phen = cv2.VideoWriter(os.path.join(save_path, f"phenotypes_{pflag}.avi"),cv2.VideoWriter_fourcc(*'DIVX'), 15, (tar_shape, tar_shape), 0)

    gen_plot_frames = []
    phen_plot_frames = []

    for i in range(phen_fitness.shape[0]):
        #for each generation, get the max phenotypic fitness

        gen_frame = all_gen_states[i][max_idxs[i]]#.reshape(tar_shape, tar_shape)
        phen_frame = all_phen_states[i][max_idxs[i]]#.reshape(tar_shape, tar_shape, 1)

        gen_plot_frames.append(gen_frame)# = (gen_frame*255).astype(np.uint8)
        phen_plot_frames.append(phen_frame)
        # phen_frame = (phen_frame*255).astype(np.uint8)

    def ImageAnimation(i):
        ax1.matshow(gen_plot_frames[i], interpolation = 'nearest', cmap=cm.Greys_r)
        ax2.matshow(phen_plot_frames[i], interpolation = 'nearest', cmap=cm.Greys_r)

        ax1.set_title(f"Genotype\nFitness: {max_gen[i]:.2f}")
        ax2.set_title(f"Phenotype\nFitness: {max_phen[i]:.2f}\n competency: {comp_vals[i]}")

        ax1.set_axis_off()
        ax2.set_axis_off()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    anim_created = animation.FuncAnimation(fig, ImageAnimation, frames=len(gen_plot_frames), interval=1)
    writervideo = animation.FFMpegWriter(fps=10)
    anim_created.save(os.path.join(save_path, f"rearrangement_{pflag}.mp4"), writer=writervideo)



def create_highletedFrameMovie(gen_fitness, phen_fitness, comp_list, gen_states, phen_states, tar_shape, pflag, tar, save_path):

    max_idxs = [np.argmax(phen_fitness[i]) for i in range(phen_fitness.shape[0])]

    max_gen = [gen_fitness[i][max_idxs[i]]for i in range(gen_fitness.shape[0])]
    max_phen = [phen_fitness[i][max_idxs[i]] for i in range(phen_fitness.shape[0])]
    comp_vals = [comp_list[i][max_idxs[i]] for i in range(comp_list.shape[0])]

    all_gen_states = np.array(list(gen_states.values())).reshape(phen_fitness.shape[0], -1, tar_shape, tar_shape)
    all_phen_states = np.array(list(phen_states.values())).reshape(phen_fitness.shape[0], -1, tar_shape, tar_shape)

    # out_gen = cv2.VideoWriter(os.path.join(save_path, f"genotypes_{pflag}.avi"),cv2.VideoWriter_fourcc(*'DIVX'), 15, (tar_shape, tar_shape), 0)
    # out_phen = cv2.VideoWriter(os.path.join(save_path, f"phenotypes_{pflag}.avi"),cv2.VideoWriter_fourcc(*'DIVX'), 15, (tar_shape, tar_shape), 0)

    gen_plot_frames = []
    phen_plot_frames = []

    tar = np.rot90(tar, k=2, axes = (0,1))
    colors = 'black white #548C2F'.split()
    cmap = matplotlib.colors.ListedColormap(colors, name='highl', N = None)

    for i in range(phen_fitness.shape[0]):

        #for each generation, get the max phenotypic fitness
        gen_frame = all_gen_states[i][max_idxs[i]]#.reshape(tar_shape, tar_shape)
        phen_frame = all_phen_states[i][max_idxs[i]]#.reshape(tar_shape, tar_shape, 1)

        gen_plot_frames.append(gen_frame)# = (gen_frame*255).astype(np.uint8)
        phen_plot_frames.append(phen_frame)
        # phen_frame = (phen_frame*255).astype(np.uint8)

    gen_plot_frames = gen_plot_frames[:300]
    phen_plot_frames = phen_plot_frames[:300]

    def ImageAnimation(i):

        src = gen_plot_frames[i]
        src = np.rot90(src, k=2, axes = (0,1))

    # green: cells in their correct pos, white: 1.0, black: 0.0

        # set correct pos cellst to idx of 2.0
        src[np.where(src == tar)] = 2.0
        ax.pcolormesh(src, edgecolors = '#000', vmin = 0, vmax=2, linewidth=2, cmap=cmap)

    # show mat with custom cmap
    fig, ax = plt.subplots(1, 1, figsize=(5,5), dpi=200)
    fig.tight_layout()
    ax.axis(False)

    anim_created = animation.FuncAnimation(fig, ImageAnimation, frames=len(gen_plot_frames), interval=1)
    writervideo = animation.FFMpegWriter(fps=10)
    anim_created.save(os.path.join(save_path, f"highlighted_movie_{pflag}.mp4"), writer=writervideo)


if __name__ == "__main__":
    rng = np.random.default_rng(12345)
    tar = load_from_txt("./smiley.png", 35)
    all_frames = [scramble(tar, rng) for i in range(100)]

    plt.matshow(all_frames[10])
    plt.show()
    # fig, ax = plt.subplots(1, 1, figsize=(5,5), dpi=200)
    # fig.tight_layout()
    # ax.axis(False)
    # anim_created = animation.FuncAnimation(fig, ImageAnimation, frames=100, interval=1)
    # writervideo = animation.FFMpegWriter(fps=2)
    # anim_created.save('smiley_scrambled.mp4', writer=writervideo)
    # print("hello")
    #test
