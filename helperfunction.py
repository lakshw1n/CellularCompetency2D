#!/usr/bin/env python3

#author: Lakshwin Shreesha
#date: 9/09/2023
#library functions for the stress based competency model

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
    if tar.shape[0]*tar.shape[1] == 0 || src.shape[0]*src.shape[1] == 0:
        raise Exception("input matrices cannot be empty\n")

    return np.mean((src-tar)**2)

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
        if (row_pos >= tar.size[0]):
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
    if (idx[0]<0 || idx[1]<0):
        raise Exception("indexes must be positive\n")
    if (idx[0]>src.shape[0] || idx[1]>src.shape[1]):
        raise Exception("indexes cannot be greater than src shape\n")

    if (src.shape != tar.shape):
        raise Exception("src and tar must be of the same shape\n")

    # each cell has to look in eight different directions

    #look up
    neighbor_list = []
    for i in range(8):
        to_be_idx, to_be_neighbor = get_cellId_from_direction(idx, i)
        if to_be_neighbor = -1:
            continue
        neighbor_list.append((to_be_idx, to_be_neighbor))


def get_required_neighbors(src, tar, stress):
    # get indexes of fixed cells
    loc_x, loc_y = np.where(stress == 0))
    fixed_idxs = list(zip(loc_x, loc_y))

    # for each fixed idx, get neighbor list
    neighbor_needs = {}
    for idx in fixed_idxs:
        neighbor_needs[idx] = indv_cell_requirement(idx, src, tar, stress)





def apply_competency(src_pop, comp_value, rng, plasticity_flag):
    if src.shape[0] == 0:
        raise Exception("src population cannot be empty\n")

    if comp_value <0:
        raise Exception("competency must be a positive integer\n")

    n_swaps = 0
    n_stressed_cells = 1000000 # set to a large number
    for src in src_pop:
        while (n_swaps < comp_value and n_stressed_cells > 0):
            stress = get_stress(src, tar)
            neighbor_requirement = required_neighbors(src, tar, stress)



def evolve(src_pop, tar, n_gen, rng):
    for i in range(n_gen):
        genotypic_fitness = [fitness(src, tar) for src in src_pop]
        apply_competency(src_pop, comp_value, rng, plasticity_flag= False)





