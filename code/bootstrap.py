from tda import PD
from scipy import optimize
from matplotlib import pyplot as plt

import tda
import numpy as np
import random


def list_matrix_pwgk(list_pd, f_kernel, f_weight, x, ee, num_mesh=5):
    num_pd = len(list_pd)
    # prepare a grid matrix
    num_half = num_mesh // 2
    mat_grid = np.empty((num_mesh, num_mesh, 2))
    for i in range(num_mesh):
        for j in range(num_mesh):
            mat_grid[i, j, 0] = x[0] + (i - num_half) * ee
            mat_grid[i, j, 1] = x[1] + (j - num_half) * ee

    # prepare weight vectors
    def vector_weight(diagram):
        num_point = int(diagram.size / 2)
        vec = np.empty(num_point)
        for temp_i in range(num_point):
            vec[temp_i] = f_weight(diagram[temp_i, :])
        return vec

    list_weight = []
    for k in range(num_pd):
        list_weight.append(vector_weight(list_pd[k]))

    # compute pwgk values
    def matrix_pwgk(diagram, vec_weight):
        def value_pwgk(z):
            num_points = int(diagram.size / 2)
            vec_k = np.empty(num_points)
            for temp_ii in range(num_points):
                vec_k[temp_ii] = f_kernel(z, diagram[temp_ii, :])
            return np.inner(vec_weight, vec_k)

        mat = np.empty((num_mesh, num_mesh))
        for temp_i in range(num_mesh):
            for temp_j in range(num_mesh):
                mat[temp_i, temp_j] = value_pwgk(mat_grid[temp_i, temp_j])
        return mat

    list_mat = []
    for k in range(num_pd):
        list_mat.append(matrix_pwgk(list_pd[k], list_weight[k]))

    return list_mat


def quantile_bootstrap(list_pd, f_kernel, f_weight, x, ee, alpha=0.05,
                       num_boot=int(1e+3)):
    num_pd = len(list_pd)

    def average_list(list_mat):
        num_mesh = list_mat[0].shape[0]
        mat = np.zeros((num_mesh, num_mesh))
        for k in range(num_pd):
            mat += list_mat[k]
        return mat / num_pd

    def list_boot(list_mat):
        idx_boot = random.choices(range(num_pd), k=num_pd)
        list_temp = []
        for k in range(num_pd):
            list_temp.append(list_mat[idx_boot[k]])
        return list_temp

    list_mat_pwgk = list_matrix_pwgk(list_pd, f_kernel, f_weight, x, ee)
    mat_emp = average_list(list_mat_pwgk)
    vec_boot = np.empty(num_boot)
    for b in range(num_boot):
        mat_boot = average_list(list_boot(list_mat_pwgk))
        vec_boot[b] = np.sqrt(num_pd) * np.max(np.abs(mat_boot - mat_emp))

    def func_emp(c):
        cumulative = len(np.where(vec_boot < c)[0]) / len(vec_boot)
        return cumulative - (1 - alpha)

    c_hat = optimize.brentq(func_emp, min(vec_boot), max(vec_boot))
    num_half = int(mat_emp.shape[0] // 2)

    return mat_emp[num_half, num_half],  np.float64(c_hat) / np.sqrt(num_pd)


CONST_PD = 1
CONST_ALPHA = 0.05
CONST_IID = 100
CONST_USE = 100

main = ["lattice", "matern"][1]
name_dir = "../data/" + main

# type of data set
if main == "lattice":
    CONST_PCD = 2
    CONST_SIDE = 20
    CONST_X = [0.5, 0.6]
    CONST_EE = 0.05
    # list_name_pcd = ["square_014", "square_017", "square_020", "gauss_010"]
    list_name_pcd = ["square_017", "gauss_010"]
    name_parameter = "pcd%s_side%s_iid%s" % (CONST_PCD, CONST_SIDE, CONST_IID)
else:  # matern
    CONST_PCD = 2
    CONST_LAMBDA = 100
    CONST_WIDTH = 1
    CONST_DISTANCE = 0.05
    CONST_X = [0.07, 0.075]
    CONST_EE = 0.01
    # list_name_pcd = ["type_0", "type_1", "type_2"]
    list_name_pcd = ["type_0", "type_2"]
    name_parameter = "pcd%s_lambda%s_width%s_distance%s_iid%s" % (
        CONST_PCD, CONST_LAMBDA, CONST_WIDTH,
        str(CONST_DISTANCE).replace(".", ""), CONST_IID)

# compute/import parameters
temp_v = tda.import_parameters(
    name_dir, name_parameter, list_name_pcd, CONST_IID, CONST_PD)
min_birth, max_death, med_pers, max_pers, med_sigma = temp_v

# define a kernel and a weight function
name_large = ["", "_small", "_large"][0]
for name_weight in ["one", "linear", "arctan"]:
    print(name_weight, name_large)

    if name_large == "_small":
        sigma = med_sigma / 10
    elif name_large == "_large":
        sigma = med_sigma * 10
    else:
        sigma = med_sigma
    func_kernel = tda.function_kernel("Gaussian", sigma=sigma)
    func_weight = tda.function_weight(
        name_weight, arc_c=med_pers, arc_p=5, lin_el=max_death)

    print(main, list_name_pcd, CONST_USE, CONST_X, CONST_EE)
    print("%s percent confidence interval" % (int((1 - CONST_ALPHA) * 100)))

    num_type = len(list_name_pcd)
    vec_ave = np.empty(num_type)
    vec_q = np.empty(num_type)
    for idx_temp, name_pcd in enumerate(list_name_pcd):
        # import diagrams
        name_data = "%s_%s" % (name_parameter, name_pcd)
        list_diagram = PD(name_dir, name_data,
                          CONST_IID, CONST_PD).data[0:CONST_USE]

        # define the empirical bootstrap process
        value_ave, q_side = quantile_bootstrap(
            list_diagram, func_kernel, func_weight, CONST_X, CONST_EE,
            CONST_ALPHA)
        vec_ave[idx_temp] = value_ave
        vec_q[idx_temp] = q_side

    print(vec_ave - vec_q)
    print(vec_ave + vec_q)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_lin = np.linspace(0, num_type - 1, num_type)
    ax.errorbar(x_lin, vec_ave, yerr=[vec_q, vec_q], fmt='o')
    ax.set_xlim(-0.5, num_type - 0.5)
    plt.xticks(x_lin, list_name_pcd)
    plt.savefig("%s/bootstrap_%s%s.png" % (
        name_dir, name_weight, name_large))
    plt.close()
    # plt.show()
