#!/usr/bin/env python3

#author: Lakshwin Shreesha
#date: 9/09/2023
#library functions for the stress based competency model
import numpy as np
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import cv2
from multiprocessing import Pool

def load_from_txt(fname):
    im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (30, 30), interpolation = cv2.INTER_AREA)
    im = np.round(im/255.0)
    return im


def scramble(tar, rng):
    if tar.shape[0]*tar.shape[1] == 0:
        raise Exception("target empty\n")

    temp = tar.copy()
    rng.shuffle(temp)

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
        if (stress[new_idx] >0):
            return new_idx, tar[new_idx]

        else:
            return -1, -1

    elif (dir == 1): #south
        row_pos = i+1
        if (row_pos >= tar.shape[0]):
            return -1, -1
        col_pos = j
        new_idx = (row_pos, col_pos)

        if (stress[new_idx] >0):
            return new_idx, tar[new_idx]

        else:
            return -1, -1

    elif (dir == 2): #west
        row_pos = i
        col_pos = j-1
        if (col_pos <0):
            return -1, -1

        new_idx = (row_pos, col_pos)
        if (stress[new_idx]>0):
            return new_idx, tar[new_idx]

        else:
            return -1, -1

    elif (dir == 3): #east
        row_pos = i
        col_pos = j+1
        if (col_pos >=tar.shape[1]):
            return -1, -1

        new_idx = (row_pos, col_pos)
        if (stress[new_idx]>0):
            return new_idx, tar[new_idx]

        else:
            return -1, -1

    elif (dir == 4): #nw
        row_pos = i-1
        col_pos = j-1
        if (col_pos <0 or row_pos <0):
            return -1, -1

        new_idx = (row_pos, col_pos)
        if (stress[new_idx]>0):
            return new_idx, tar[new_idx]

        else:
            return -1, -1

    elif (dir == 5): #ne
        row_pos = i-1
        col_pos = j+1
        if (col_pos >=tar.shape[1] or row_pos <0):
            return -1, -1

        new_idx = (row_pos, col_pos)
        if (stress[new_idx]>0):
            return new_idx, tar[new_idx]

        else:
            return -1, -1

    elif (dir == 6): #sw
        row_pos = i+1
        col_pos = j-1
        if (col_pos <0 or row_pos >=tar.shape[0]):
            return -1, -1

        new_idx = (row_pos, col_pos)
        if (stress[new_idx]>0):
            return new_idx, tar[new_idx]

        else:
            return -1, -1

    elif (dir == 7): #se
        row_pos = i+1
        col_pos = j+1
        if (col_pos >=tar.shape[1] or row_pos >=tar.shape[0]):
            return -1, -1

        new_idx = (row_pos, col_pos)
        if (stress[new_idx]>0):
            return new_idx, tar[new_idx]

        else:
            return -1, -1


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
        if to_be_neighbor == -1:
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

    if (hard_delete==1):
        try:
            del stressed_dict[tuple(from_pos)]
            del stressed_dict[tuple(to_pos)]
        except KeyError:
            pass

    if len(stressed_dict) ==0:
        return 0

    for k, v in stressed_dict.items():
        try:
            del v[tuple(from_pos)]
            del v[tuple(to_pos)]
        except KeyError:
            continue

def swap(indv, from_pos, to_pos):
    temp = indv[tuple(from_pos)]
    indv[tuple(from_pos)] = indv[tuple(to_pos)]
    indv[tuple(to_pos)] = temp

def move(stressed_dict, src, tar, stress, comp_value, plasticity_flag, rng):
    #for each stressed cell find the maximum to_be_idx signal value
    #move once for each position

    distance = lambda f,t: np.sqrt((f[0]-t[0])**2 + (f[1] - t[1])**2)
    moves = 0

    while(len(stressed_dict)!=0 and moves < comp_value):
        #pick a random stressed position
        kys = list(stressed_dict.keys())
        from_pos = rng.choice(kys)

        v = stressed_dict[tuple(from_pos)]

        if (len(list(v.values()))) == 0:
            break

        max_pos = np.argmax(v.values())
        to_pos = list(v.keys())[max_pos]

        from_pos = np.array(from_pos)
        to_pos = np.array(to_pos)

        # the absolute worst way to do exclude from==to swaps, but fuck it
        if(from_pos[0] == to_pos[0] and from_pos[1] == to_pos[1]):
            continue

        #check if any other cell in stressed_dict needs to move to from_pos
        #if so, swap, and delete their entries in stressed_dict
        d = distance(from_pos, to_pos)
        if (d<=np.sqrt(2)):
            #then swap can occur
            swap(src, from_pos, to_pos)
            moves += 1
            #delete their entries in the dict
            delete_pos(stressed_dict, from_pos, to_pos)

        else:
            if (plasticity_flag ==True):
                #if not, then move only if plasticity is True
                moves += 1
                swap(src, from_pos, to_pos)
                #remove their scores from every stressed_cell
                delete_pos(stressed_dict, from_pos, to_pos)
            else:
                # print("stuck")
                delete_pos(stressed_dict, from_pos, to_pos, hard_delete = 0)

    return moves



def apply_competency(src_pop_main, tar, comp_value, rng, plasticity_flag):
    src_pop = src_pop_main.copy()
    used_moves = []
    if src_pop.shape[0] == 0:
        raise Exception("src population cannot be empty\n")

    if comp_value <0:
        raise Exception("competency must be a positive integer\n")

    for curr_n, src in enumerate(src_pop):
        print("Idv: " + str(curr_n) +"/"+str(src_pop.shape[0]))
        stress = get_stress(src, tar)
        neighbor_requirement = get_required_neighbors(src, tar, stress)
        stressed_dict = send_graded_signal(neighbor_requirement, src, stress)
        used_moves.append(move(stressed_dict, src, tar, stress, comp_value, plasticity_flag, rng))

    return src_pop, used_moves

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

def evolve(src_pop, tar, n_gen, comp_value, rng, pf_Flag, mut_rate, N_mut, N):
    gen_matrix = np.zeros((n_gen, src_pop.shape[0]))
    phen_matrix = np.zeros((n_gen, src_pop.shape[0]))
    comp_vals = np.zeros((n_gen, src_pop.shape[0]))

    for i in range(n_gen):
        genotypic_fitness = [fitness(src, tar) for src in src_pop]
        mod_pop, used_moves = apply_competency(src_pop, tar, comp_value, rng, plasticity_flag= pf_Flag)
        phenotypic_fitness = [fitness(src_m, tar) for src_m in mod_pop]
        sel_pop = selection(src_pop, phenotypic_fitness, N)
        src_pop = mutate_pop(sel_pop, mut_rate, N_mut, N, rng)
        gen_matrix[i] = genotypic_fitness
        phen_matrix[i] = phenotypic_fitness
        comp_vals[i] = list(used_moves)
        print(str(i) + " | " + "geno: " + str(max(genotypic_fitness)) + " | " + "phen: "+str(max(phenotypic_fitness))+" | " + "used comp: " +str(max(list(used_moves))))

    np.save("/Users/niwhskal/competency2d/output/gen_matrix.npy", gen_matrix )
    np.save("/Users/niwhskal/competency2d/output/phen_matrix.npy", phen_matrix)
    np.save("/Users/niwhskal/competency2d/output/comp_vals.npy", comp_vals)

def plot(genf, phenf, compf, pflag):
    gen = np.load(genf)
    phen = np.load(phenf)
    comp_vals = np.load(compf)

    max_gen = [max(gen[i]) for i in range(gen.shape[0])]
    max_phen = [max(phen[i]) for i in range(phen.shape[0])]
    comp_vals = [max(comp_vals[i]) for i in range(comp_vals.shape[0])]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f"Stress Sharing:{pflag}")
    ax1.plot(max_gen, label = "Genotypic Fitness")
    ax1.plot(max_phen, label = "Phenotypic Fitness")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Max Fitness")

    ax2.plot(comp_vals, label = "Competency value")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Max swaps used")

    ax1.legend()
    ax2.legend()

    plt.savefig(f"/Users/niwhskal/competency2d/output/result_ss_{pflag}.png")
    plt.show()


if __name__ == "__main__":
    load_from_txt("/Users/niwhskal/Downloads/MNIST_6_0.png")
    print("hello")
    #test
