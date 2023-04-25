from evosquares import perturbations
from evosquares.operators.utils import sort_closest
import numpy as np 

def match_swap_crossover(X, n_closest=1, repair_steps=0, repair_size=0):
    # The input has the following shape (n_parents, n_matings, n_var)
    _, n_matings, n_var = X.shape
    n = n_var // 3
    mag = 2 / np.ceil(np.sqrt(n))
    n_closest = np.random.randint(1, n_closest - 1)

    # The output has shape (n_offsprings, n_matings, n_var)
    # Because the number of parents and offsprings are equal it keeps the shape of X
    Y = np.full_like(X, None, dtype='float64')

    x_r_center = np.tile(np.expand_dims(np.random.uniform(-1, 1, size=(n_matings)), axis=1), (1, n))
    y_r_center = np.tile(np.expand_dims(np.random.uniform(-1, 1, size=(n_matings)), axis=1), (1, n))
    R = np.concatenate((x_r_center, y_r_center), axis=1)

    x_a_sorted, y_a_sorted, t_a_sorted = sort_closest(X[0], R)
    x_b_sorted, y_b_sorted, t_b_sorted = sort_closest(X[1], R)

    temp_x_a = np.copy(x_a_sorted[:, :n_closest])
    temp_y_a = np.copy(y_a_sorted[:, :n_closest])
    temp_t_a = np.copy(t_a_sorted[:, :n_closest])

    x_a_sorted[:, :n_closest] = x_b_sorted[:, :n_closest]
    y_a_sorted[:, :n_closest] = y_b_sorted[:, :n_closest]
    t_a_sorted[:, :n_closest] = t_b_sorted[:, :n_closest]

    x_b_sorted[:, :n_closest] = temp_x_a
    y_b_sorted[:, :n_closest] = temp_y_a
    t_b_sorted[:, :n_closest] = temp_t_a

    return x_a_sorted, y_a_sorted, t_a_sorted, x_b_sorted, y_b_sorted, t_b_sorted

    Y[0] = np.concatenate((x_a_sorted, y_a_sorted, t_a_sorted), axis=1)
    Y[1] = np.concatenate((x_b_sorted, y_b_sorted, t_b_sorted), axis=1)

    a_size, b_size = perturbations.find_max_size(Y[0]), perturbations.find_max_size(Y[1])
    perturbations.random_walk(Y[0], repair_steps, repair_size, a_size)
    perturbations.random_walk(Y[1], repair_steps, repair_size, b_size)

    return Y

def random_swap_crossover(X, n_closest=1, repair_steps=0, repair_size=0):
    # The input has the following shape (n_parents, n_matings, n_var)
    _, n_matings, n_var = X.shape
    n = n_var // 3
    mag = 2 / np.ceil(np.sqrt(n))
    n_closest = np.random.randint(1, n_closest - 1)

    # The output has shape (n_offsprings, n_matings, n_var)
    # Because the number of parents and offsprings are equal it keeps the shape of X
    Y = np.full_like(X, None, dtype='float64')

    x_r_center = np.tile(np.expand_dims(np.random.uniform(-1, 1, size=(n_matings)), axis=1), (1, n))
    y_r_center = np.tile(np.expand_dims(np.random.uniform(-1, 1, size=(n_matings)), axis=1), (1, n))
    R = np.concatenate((x_r_center, y_r_center), axis=1)

    x_a_sorted, y_a_sorted, t_a_sorted = sort_closest(X[0], R)

    x_r_center = np.random.uniform(-1, 1, size=(n_matings, n))
    y_r_center = np.random.uniform(-1, 1, size=(n_matings, n))
    R = np.concatenate((x_r_center, y_r_center), axis=1)

    x_b_sorted, y_b_sorted, t_b_sorted = sort_closest(X[1], R)

    x_offset = np.mean(x_b_sorted[:, :n_closest]) - np.mean(x_a_sorted[:, :n_closest])
    y_offset = np.mean(y_b_sorted[:, :n_closest]) - np.mean(y_a_sorted[:, :n_closest])

    temp_x_a = np.copy(x_a_sorted[:, :n_closest])
    temp_y_a = np.copy(y_a_sorted[:, :n_closest])
    temp_t_a = np.copy(t_a_sorted[:, :n_closest])

    x_a_sorted[:, :n_closest] = np.clip(x_b_sorted[:, :n_closest] - x_offset, -1 + mag / 2, 1 - mag / 2)
    y_a_sorted[:, :n_closest] = np.clip(y_b_sorted[:, :n_closest] - y_offset, -1 + mag / 2, 1 - mag / 2)
    t_a_sorted[:, :n_closest] = t_b_sorted[:, :n_closest]

    x_b_sorted[:, :n_closest] = np.clip(temp_x_a + x_offset, -1 + mag / 2, 1 - mag / 2)
    y_b_sorted[:, :n_closest] = np.clip(temp_y_a + y_offset, -1 + mag / 2, 1 - mag / 2)
    t_b_sorted[:, :n_closest] = temp_t_a

    Y[0] = np.concatenate((x_a_sorted, y_a_sorted, t_a_sorted), axis=1)
    Y[1] = np.concatenate((x_b_sorted, y_b_sorted, t_b_sorted), axis=1)

    a_size, b_size = perturbations.find_max_size(Y[0]), perturbations.find_max_size(Y[1])
    perturbations.random_walk(Y[0], repair_steps, repair_size, a_size)
    perturbations.random_walk(Y[1], repair_steps, repair_size, b_size)

    return Y
