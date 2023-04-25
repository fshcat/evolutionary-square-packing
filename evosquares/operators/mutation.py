from evosquares import perturbations
from evosquares.operators.utils import sort_closest
import numpy as np 

def perturb_mutation(X, perturb_prob, perturb_num, perturb_size):
    pop, n3 = X.shape
    n = n3 // 3

    rand = np.argwhere(np.random.random(size=pop) < perturb_prob)[:, 0]
    sub_pop = len(rand)
    perturb_num = np.random.randint(perturb_num)

    #copy = np.copy(X[rand])

    if sub_pop != 0:
        X_R = X[rand]

        x_r_center = np.tile(np.expand_dims(np.random.uniform(-1, 1, size=(sub_pop)), axis=1), (1, n))
        y_r_center = np.tile(np.expand_dims(np.random.uniform(-1, 1, size=(sub_pop)), axis=1), (1, n))
        R = np.concatenate((x_r_center, y_r_center), axis=1)

        sorted_x, sorted_y, sorted_t = sort_closest(X_R, R)
        sorted_x[:, :perturb_num], sorted_y[:, :perturb_num], sorted_t[:, :perturb_num]\
                = perturbations.uniform_shift_square(sorted_x[:, :perturb_num], sorted_y[:, :perturb_num], sorted_t[:, :perturb_num], perturb_size)
        X[rand] = np.concatenate((sorted_x, sorted_y, sorted_t % (np.pi/2)), axis=1)

    #print(np.sum(np.abs(copy - X[rand])))
    return X

def random_walk_mutation(X, walk_prob, walk_steps, walk_size, no_theta_prob):
    pop, n3 = X.shape
    n = n3 // 3
    
    rand = np.argwhere(np.random.random(size=pop) < walk_prob)[:, 0]
    sub_pop = len(rand)

    if sub_pop != 0:
        no_t = np.random.random(size=sub_pop) < no_theta_prob

        for w_size, w_steps in zip(walk_size, walk_steps):
            sizes = perturbations.find_max_size(X[rand])
            R = X[rand]
            perturbations.random_walk(R, w_steps, w_size, sizes, no_t)
            X[rand] = R

    return X

def rotation_mutation(X, rot_prob, rot_max_num, rot_size):
    pop, n3 = X.shape
    n = n3 // 3
    mag = 2 / np.ceil(np.sqrt(n))

    rot_num = np.random.randint(rot_max_num)

    rand = np.argwhere(np.random.random(size=pop) < rot_prob)[:, 0]
    sub_pop = len(rand)

    if sub_pop != 0:
        X_R = X[rand]

        angle = np.random.uniform(0, rot_size, size=(sub_pop))

        x_r_center = np.tile(np.expand_dims(np.random.uniform(-1, 1, size=(sub_pop)), axis=1), (1, n))
        y_r_center = np.tile(np.expand_dims(np.random.uniform(-1, 1, size=(sub_pop)), axis=1), (1, n))
        R = np.concatenate((x_r_center, y_r_center), axis=1)

        angle = np.expand_dims(np.random.uniform(0, rot_size, size=(sub_pop)), axis=1)
        angle = np.tile(angle, (1, rot_num))

        sorted_x, sorted_y, sorted_t = sort_closest(X_R, R)

        flat_x = np.reshape(sorted_x[:, :rot_num], (sub_pop * rot_num))
        flat_y = np.reshape(sorted_y[:, :rot_num], (sub_pop * rot_num))
        centers = np.expand_dims(np.stack((flat_x, flat_y), axis=1), axis=2)

        rot_x = np.tile(np.expand_dims(np.mean(sorted_x[:, :rot_num], axis=1), axis=1), (1, rot_num))
        rot_y = np.tile(np.expand_dims(np.mean(sorted_y[:, :rot_num], axis=1), axis=1), (1, rot_num))
        flat_rot_x = np.reshape(rot_x[:, :rot_num], (sub_pop * rot_num))
        flat_rot_y = np.reshape(rot_y[:, :rot_num], (sub_pop * rot_num))
        rot_point = np.expand_dims(np.stack((flat_rot_x, flat_rot_y), axis=1), axis=2)

        flat_angle = np.reshape(angle, (sub_pop * rot_num))
        rot_mat = np.transpose(np.array([[np.cos(flat_angle), -np.sin(flat_angle)],[np.sin(flat_angle), np.cos(flat_angle)]]), (2,0,1))

        rot_centers = rot_mat @ (centers - rot_point) + rot_point
        rot_centers = np.reshape(rot_centers, (sub_pop, rot_num, 2))

        sorted_x[:, :rot_num] = rot_centers[:, :, 0]
        sorted_y[:, :rot_num] = rot_centers[:, :, 1]
        sorted_t[:, :rot_num] = (sorted_t[:, :rot_num] + angle) % (np.pi/2)

        X[rand] = np.concatenate((np.clip(sorted_x, -1 + mag/2, 1 - mag/2), np.clip(sorted_y, -1 + mag/2, 1 - mag/2), sorted_t), axis=1)

    return X
