#!/usr/bin/env python

# Implementation of algorithm in Gensane, T., Ryckelynck, P. Improved Dense Packings of
# Congruent Squares in a Square. Discrete Comput Geom 34, 97–109 (2005)

import numpy as np
from matplotlib import pyplot as plt


# Create a random initialization for n squares
def gen_random(n):
    x = np.random.uniform(-1.0, 1.0, n)
    y = np.random.uniform(-1.0, 1.0, n)
    t = np.random.uniform(0, np.pi / 2, n)
    return x, y, t


# Find largest square size for each individual square such that it doesnt intersect with bounding square
def phi_0(a, b, theta):
    val = np.minimum(1 - np.abs(a), 1 - np.abs(b)) / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
    return val


# Find largest possible square size such that no square intersects with bounding square
def phi(a, b, theta):
    num_dims = len(theta.shape)
    retval = np.amin(phi_0(a, b, theta), axis=num_dims-1)
    return retval


def psi_0(a, b, theta):
    shape = theta.shape
    num_dims = len(shape)
    axes = (0, 1) if num_dims == 2 else 0
    expand = (1, 1) if num_dims == 2 else (1,)

    indices = np.tile(np.expand_dims(np.array([1, 2, 3, 4]), axis=axes), (*shape, 1))
    numerator = np.tile(np.expand_dims(np.abs(a) + np.abs(b), axis=num_dims), (*expand, 4))
    angles = np.tile(np.expand_dims(theta, axis=num_dims), (*expand, 4)) + (np.pi / 4) + (np.pi / 2) * indices
    signs = np.tile(np.expand_dims(np.sign(a) * np.sign(b), axis=num_dims), (*expand, 4))
    result = numerator / np.abs(1 - np.sqrt(2) * signs * np.sin(angles))

    return np.min(result, axis=num_dims)


# Find largest square size with no intersection for each pair of squares
def psi(a_i, b_i, t_i, a_j, b_j, t_j):
    diff_a = a_j - a_i
    diff_b = b_j - b_i
    diff_t = t_j - t_i
    
    cos_t_i = np.cos(t_i)
    sin_t_i = np.sin(t_i)
    cos_t_j = np.cos(t_j)
    sin_t_j = np.sin(t_j)

    psi_1 = psi_0(diff_a * cos_t_i + diff_b * sin_t_i, -diff_a * sin_t_i + diff_b * cos_t_i, diff_t)
    psi_2 = psi_0(-diff_a * cos_t_j - diff_b * sin_t_j, diff_a * sin_t_j - diff_b * cos_t_j, -diff_t)
    
    psi_f = np.maximum(psi_1, psi_2)
    mask = psi_f == 0
    
    psi_f += mask * 4
    return psi_f


# Find largest square size such that no two squares intersect with one another
def psi_c(a, b, theta):
    size = theta.shape[1]   
    
    a_i = np.tile(a, (1, size)) 
    b_i = np.tile(b, (1, size)) 
    t_i = np.tile(theta, (1, size)) 
    
    a_j = np.repeat(a, size, axis=1) 
    b_j = np.repeat(b, size, axis=1) 
    t_j = np.repeat(theta, size, axis=1)
    
    psi_f = psi(a_i, b_i, t_i, a_j, b_j, t_j)
    return np.amin(psi_f, axis=1)


# Find largest square size such that no square intersects another square or the bounding box
def find_max_size(gen):
    reshape = False
    if len(gen.shape) == 1:
        reshape = True
        gen = np.reshape(gen, (1, gen.shape[0]))

    x, y, theta, n = split_genome(gen)
    a = psi_c(x, y, theta)
    b = phi(x, y, theta)

    return np.minimum(a, b)[0] if reshape else np.minimum(a, b)


# Calculate what the bounding size of the given configuration would be if square had unit length
def find_bound(gen):
    return np.sqrt(2) / find_max_size(gen)


# Split combined genome into x, y, and theta
def split_genome(gen):
    if len(gen.shape) == 1:
        num = gen.shape[0] // 3
        return gen[:num], gen[num:num*2], gen[num*2:], num
    else:
        num = gen.shape[1] // 3
        return gen[:, :num], gen[:, num:num*2], gen[:, num*2:], num


# Calculate the corner vertices from the center and angle
def get_corners(center, angle, size):
    s = size/2
    if len(center.shape) < 3:
        center = np.expand_dims(center, axis=0)
        angle = np.expand_dims(angle, axis=0)
        
    p, n = center.shape[0:2]
    center = np.reshape(center, (p*n, 2))
    angle = np.reshape(angle, (p*n))
    
    corners = np.tile(np.array([[-s, -s, s, s],[s, -s, -s, s]]), (p*n,1,1))
    rot_mat = np.transpose(np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]]), (2,0,1))
    return np.reshape(rot_mat @ corners + np.tile(np.expand_dims(center, axis=2), (1, 1, 4)), (p, n, 2, 4))


# Creates subplot of given square configuration on ax
def subplot_squares(ax, gen):
    size = 2 * find_max_size(gen) / np.sqrt(2)
    x, y, t, n = split_genome(gen)
    centers = np.stack((x, y), axis=1)
    
    angles = t + np.pi / 4
    corners = get_corners(centers, angles, size)
    corners = corners[0]
    ax.axis('equal')

    for square in corners:
        ax.fill(square[0], square[1], facecolor='none', edgecolor='black', linewidth=0.5)

    xs = np.array([-1, -1, 1, 1])
    ys = np.array([-1, 1, 1, -1])
    ax.fill(xs, ys, facecolor='none', edgecolor='red', linewidth=0.5)


# Create plot of given square configuration
def draw_squares(gen):
    size = 2 * find_max_size(gen) / np.sqrt(2)
    x, y, t, n = split_genome(gen)
    centers = np.stack((x, y), axis=1)
    
    angles = t + np.pi / 4
    corners = get_corners(centers, angles, size)
    corners = corners[0]
    plt.axis('equal')

    for square in corners:
        plt.fill(square[0], square[1], facecolor='none', edgecolor='black', linewidth=1)

    xs = np.array([-1, -1, 1, 1])
    ys = np.array([-1, 1, 1, -1])
    plt.fill(xs, ys, facecolor='none', edgecolor='red', linewidth=1)
    plt.show()


# Randomly shift all squares according to uniform distribution on a ball of radius eps
# TODO: Untested
def uniform_shift(a, b, theta, eps):
    if len(a.shape) == 0:
        a = np.expand_dims(a, axis=0)
        b = np.expand_dims(b, axis=0)
        theta = np.expand_dims(theta, axis=0)
        
    shape = theta.shape
    t_shift = np.random.uniform(-eps, eps, shape)
    new_t = theta + t_shift
    
    new_a = a
    new_b = b
    mask = np.ones_like(a)
    
    while (np.sum(mask) > 0):
        shift_size = np.sqrt(np.random.uniform(0, 1, shape)) * eps
        shift_angle = np.random.uniform(0, 2 * np.pi, shape)

        new_a = (1 - mask) * new_a + mask * (a + shift_size * np.cos(shift_angle))
        mask_a = np.abs(new_a) > 1

        new_b = (1 - mask) * new_b + mask * (b + shift_size * np.sin(shift_angle))
        mask_b = np.abs(new_b) > 1

        mask = mask_a | mask_b
    return new_a, new_b, new_t


# Randomly shift all squares in configuration according to uniform distribution on [0, eps)
def uniform_shift_square(a, b, theta, eps):
    if len(a.shape) == 0:
        a = np.expand_dims(a, axis=0)
        b = np.expand_dims(b, axis=0)
        theta = np.expand_dims(theta, axis=0)
        
    shape = theta.shape 
    upper = np.ones_like(a)
    lower = -1 * upper
    
    new_t = np.random.uniform(theta-eps, theta+eps)
    new_a = np.random.uniform(np.maximum(a-eps, lower), np.minimum(a+eps, upper))
    new_b = np.random.uniform(np.maximum(b-eps, lower), np.minimum(b+eps, upper))

    return new_a, new_b, new_t


# Stochastic hill climbing algorithm that randomly shifts one square at a time
def random_walk(gen, num_steps, eps, size, no_theta=None):
    x, y, t, n = split_genome(gen)

    for k in range(num_steps):
        if len(gen.shape) == 2:
            pop = gen.shape[0]
            i = np.random.randint(n, size=pop)
            indivs = np.arange(pop)
            new_x, new_y, new_t = uniform_shift_square(x[indivs, i], y[indivs, i], t[indivs, i], eps)
            new_t = new_t % (np.pi/2)

            new_x_r = np.repeat(np.expand_dims(new_x, axis=1), n, axis=1)
            new_y_r = np.repeat(np.expand_dims(new_y, axis=1), n, axis=1)
            new_t_r = np.repeat(np.expand_dims(new_t, axis=1), n, axis=1)

            new_x_r[indivs, i] = x[indivs, i]
            new_y_r[indivs, i] = y[indivs, i]
            new_t_r[indivs, i] = t[indivs, i]

            first = np.amin(psi(new_x_r, new_y_r, new_t_r, x, y, t), axis=1) >= size
            second = phi_0(new_x, new_y, new_t) >= size
            mask = first & second

            x[indivs, i] = np.where(mask, new_x, x[indivs, i])
            y[indivs, i] = np.where(mask, new_y, y[indivs, i])
            t[indivs, i] = np.where(mask, new_t, t[indivs, i])
        else:
            i = np.random.randint(n)
            new_x, new_y, new_t = uniform_shift_square(x[i], y[i], t[i], eps)

            new_x_r = np.tile(new_x, n)
            new_y_r = np.tile(new_y, n)
            new_t_r = np.tile(new_t, n)
            
            new_x_r[i] = x[i]
            new_y_r[i] = y[i]
            new_t_r[i] = t[i]
        
            if np.amin(psi(new_x_r, new_y_r, new_t_r, x, y, t)) >= size and phi(new_x, new_y, new_t) >= size:
                x[i] = new_x
                y[i] = new_y
                t[i] = new_t


# Dynamically adjusts the random shift size of the hill-climbing algorithm until
# threshhold size is reached
def billiard_squares(gen, eps_i, eps_f, num_steps, verb=True):
    _, _, _, n = split_genome(gen)

    n_eval = 1
    eps = eps_i
    size = find_max_size(gen)

    while eps > eps_f:
        if verb:
            print("B: ", np.log10(eps))

        random_walk(gen, num_steps, eps, size)
        new_size = find_max_size(gen)
        if (new_size > size):
            size = new_size
            eps = min(eps*2, 2)
        elif new_size == size:
            eps /= 2
        else:
            print("ERROR: Random walk made solution worse")

        n_eval += 1 + num_steps / n

    return n_eval


# Hill climbing algorithm that randomly shifts all squares simultaneously. Dynamically
# adjusted similar to billiard_squares
def with_perturbations(gen, eps_i, eps_f, factor, num_steps):
    n_eval = 1
    eps = eps_i
    size = find_max_size(gen)
    
    while eps > eps_f:
        print("P: ", np.log10(eps))
        x, y, t, n = split_genome(gen)
        new_x, new_y, new_t = uniform_shift_square(x, y, t, eps)
        new_gen = np.concatenate((new_x, new_y, new_t))
        n_eval += billiard_squares(new_gen, eps, eps / factor, num_steps, False)
        
        new_size = find_max_size(new_gen)
        if new_size > size:
            size = new_size
            gen = new_gen
            eps *= 2
        else:
            eps /= 2

        n_eval += 1
            
    return gen, n_eval


