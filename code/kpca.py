from tda import PD, PI, PWGK
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tda
import os
import numpy as np
import math


def plot_vec(vec, title=""):
    plt.figure()
    plt.plot(vec)
    plt.title(title)
    plt.show()


def plot_mat(mat, title=""):
    plt.figure()
    plt.imshow(mat, interpolation="nearest", cmap="YlOrRd")
    plt.title(title)
    plt.show()


def kfdr(mat_gram, gamma=1e-4):
    num_pd = mat_gram.shape[0]
    vec = np.empty(num_pd - 1)

    for k in range(1, num_pd):
        vec_one1 = np.r_[np.ones(k), np.zeros(num_pd - k)]
        vec_one2 = np.r_[np.zeros(k), np.ones(num_pd - k)]
        vec_eta = - vec_one1 / k + vec_one2 / (num_pd - k)
        mat_q = ((np.matrix(vec_one1).T * vec_one1) / k
                 + (np.matrix(vec_one2).T * vec_one2) / (num_pd - k))
        mat_one = np.identity(num_pd)
        mat_inv = np.linalg.inv(mat_gram * (mat_one - mat_q) + gamma * mat_one)
        vec[k - 1] = (k * (num_pd - k) / num_pd) * (
            vec_eta * mat_inv * mat_gram * np.matrix(vec_eta).T)
    return vec


def matrix_kpca(mat_gram):
    num_pd = len(mat_gram)
    mat_center = np.empty((num_pd, num_pd))
    vec = np.sum(mat_gram, 0)
    val_total = np.sum(vec)

    for i in range(num_pd):
        for j in range(i + 1):
            mat_center[i, j] = (mat_gram[i, j] - ((vec[i] + vec[j]) / num_pd)
                                + (val_total / (num_pd ** 2)))
            mat_center[j, i] = mat_center[i, j]

    vec_eigen, mat_eigen = np.linalg.eigh(mat_center)
    idx_eigen = vec_eigen.argsort()[::-1]
    vec_eigen = vec_eigen[idx_eigen]
    mat_eigen = mat_eigen[:, idx_eigen]

    mat_kpca = np.empty((3, num_pd))
    for k in range(num_pd):
        mat_kpca[0, k] = (math.pow(vec_eigen[0], -0.5)) * mat_eigen[k, 0]
        mat_kpca[1, k] = (math.pow(vec_eigen[1], -0.5)) * mat_eigen[k, 1]
        mat_kpca[2, k] = (math.pow(vec_eigen[2], -0.5)) * mat_eigen[k, 2]

    print("Contribution rate 2d: ", vec_eigen[0:2].sum() / vec_eigen.sum())
    print("Contribution rate 3d: ", vec_eigen[0:3].sum() / vec_eigen.sum())
    return mat_kpca, vec_eigen


def plot_kpca(gram_mat, title=""):
    num_pd = len(gram_mat)

    vec_kfdr = kfdr(gram_mat)
    idx_kfdr = vec_kfdr.argmax() + 1
    print("Index of KFDR: ", idx_kfdr)

    mat_kpca, vec_eigen = matrix_kpca(gram_mat)
    mat_before = mat_kpca[:, 0:idx_kfdr]
    mat_after = mat_kpca[:, idx_kfdr:num_pd]
    len_after = num_pd - idx_kfdr - 1

    plt.figure(figsize=(12, 6))
    plt.rcParams["font.size"] = 12
    scale = 0.7
    plt.annotate("PD(1)",
                 xy=(mat_before[0, 0], mat_before[1, 0]),
                 xytext=(scale * mat_before[0, 0], scale * mat_before[1, 0]),
                 arrowprops=dict(facecolor="black", width=0.1, headwidth=0))
    plt.annotate("PD(%s)" % num_pd,
                 xy=(mat_after[0, len_after], mat_after[1, len_after]),
                 xytext=(scale * mat_after[0, len_after],
                         scale * mat_after[1, len_after]),
                 arrowprops=dict(facecolor="black", width=0.1, headwidth=0))

    plt.plot(mat_before[0, :], mat_before[1, :], "bx",
             mat_after[0, :], mat_after[1, :], "ro")
    plt.title(title)
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(mat_before[0, :], mat_before[1, :], mat_before[2, :], "bx")
    ax.plot(mat_after[0, :], mat_after[1, :], mat_after[2, :], "ro")
    ax.text(scale * mat_before[0, 0], scale * mat_before[1, 0],
            scale * mat_before[2, 0], "PD(1)")
    ax.text(scale * mat_after[0, len_after], scale * mat_after[1, len_after],
            scale * mat_after[2, len_after], "PD(%s)" % num_pd)
    plt.show()


def plot_stat(mat_gram, title=""):
    plot_mat(mat_gram, title)
    plot_vec(kfdr(mat_gram), title)
    plot_kpca(mat_gram, title)


CONST_PD = 1
CONST_IID = 100

main = ["lattice", "matern"][1]
name_dir = "../data/" + main

# type of data set
if main == "lattice":
    CONST_PCD = 2
    CONST_SIDE = 20
    list_name_pcd = ["square_017", "gauss_010"]
    # list_name_pcd = ["square_020", "gauss_010"]
    name_parameter = "pcd%s_side%s_iid%s" % (CONST_PCD, CONST_SIDE, CONST_IID)
else:  # matern
    CONST_PCD = 2
    CONST_LAMBDA = 100
    CONST_WIDTH = 1
    CONST_DISTANCE = 0.05
    list_name_pcd = ["type_0", "type_2"]
    # list_name_pcd = ["type_1", "type_2"]
    name_parameter = "pcd%s_lambda%s_width%s_distance%s_iid%s" % (
        CONST_PCD, CONST_LAMBDA, CONST_WIDTH,
        str(CONST_DISTANCE).replace(".", ""), CONST_IID)

# make directory to save the Gram matrix
name_dir_gram = "%s/%s_pd%s_%s_vs_%s" % (
    name_dir, name_parameter, CONST_PD, list_name_pcd[0],
    list_name_pcd[1])
tda.os_mkdir(name_dir_gram)

# compute/import parameters
temp_v = tda.import_parameters(
    name_dir, name_parameter, list_name_pcd, CONST_IID, CONST_PD)
min_birth, max_death, med_pers, max_pers, med_sigma = temp_v

# import diagrams of P and Q
list_diagram_xy = []
for name_pcd in list_name_pcd:
    name_data = "%s_%s" % (name_parameter, name_pcd)
    diagram = PD(name_dir, name_data, CONST_IID, CONST_PD)
    list_diagram_xy.extend(diagram.data)

# define a kernel and a weight function
name_large = ["", "_small", "_large"][0]
name_weight = ["one", "linear", "arctan"][2]
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

# compute/import the (k,w)-linear Gram matrix
name_linear_pwgk = "%s/gram_mat_pwgk_%s%s_Linear.txt" % (
    name_dir_gram, name_weight, name_large)
if not os.path.exists(name_linear_pwgk):
    pwgk = PWGK(list_diagram_xy, func_kernel, func_weight,
                sigma=sigma, name_rkhs="Linear",
                approx=True)
    mat_linear_pwgk = pwgk.gram_matrix()
    np.savetxt(name_linear_pwgk, mat_linear_pwgk)
else:
    mat_linear_pwgk = np.loadtxt(name_linear_pwgk)

# define the Gram matrix on persistence diagrams
name_rkhs = ["Linear", "Gaussian"][1]
mat_gram_pwgk = tda.matrix_gram(
    mat_linear_pwgk, name_rkhs)[0]

plot_stat(mat_gram_pwgk, "pwgk")
