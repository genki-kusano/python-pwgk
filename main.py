from tda import PD, PWGK, PL, PSSK

import tda
import numpy as np
import os
import random


def n_mmd(mat_gram, unbias=True):
    n_total = mat_gram.shape[0]
    n = int(n_total / 2)
    mat_xx = mat_gram[0:n, 0:n]
    mat_yy = mat_gram[n:n_total, n:n_total]
    mat_xy = mat_gram[0:n, n:n_total]
    sum_xx = sum(sum(mat_xx))
    sum_yy = sum(sum(mat_yy))
    sum_xy = sum(sum(mat_xy))
    if unbias:
        sum_xx -= sum(np.diag(mat_xx))
        sum_yy -= sum(np.diag(mat_yy))
        sum_xy -= sum(np.diag(mat_xy))
        return (sum_xx + sum_yy - 2 * sum_xy) / (n - 1)
    else:
        return (sum_xx + sum_yy - 2 * sum_xy) / n


def hist_wchi(mat_gram, num_hist=int(1e+4)):
    n = len(mat_gram)

    # centered Gram matrix
    mat_center = np.empty((n, n))
    vec_gram = sum(mat_gram)
    val_total = sum(vec_gram)
    for i in range(n):
        for j in range(i + 1):
            mat_center[i, j] = (mat_gram[i, j]
                                - ((vec_gram[i] + vec_gram[j]) / n)
                                + (val_total / (n ** 2)))
            mat_center[j, i] = mat_center[i, j]

    # estimated eigenvalues
    vec_nu = np.sort(np.linalg.eigh(mat_center)[0])[::-1][0: - 1]
    vec_lambda = vec_nu / (n - 1)
    sum_lambda = sum(vec_lambda)

    # histogram of the null distribution (weighted chi square)
    vec_hist = np.empty(num_hist)
    for i in range(num_hist):
        vec_z = np.random.normal(0, np.sqrt(2), n - 1) ** 2
        vec_hist[i] = np.inner(vec_lambda, vec_z) - 2 * sum_lambda

    return np.sort(vec_hist)[::-1]


def extract_submat(mat_gram, num_m=None):
    n_total = mat_gram.shape[0]
    n = int(n_total / 2)
    if num_m is None:
        num_m = n - 1
    d = int(2 * num_m)
    mat = np.empty((d, d))
    idx_x = random.sample(range(0, n), num_m)
    idx_y = random.sample(range(n, n_total), num_m)
    idx_xy = idx_x + idx_y
    for i, a in enumerate(idx_xy):
        for j, b in enumerate(idx_xy):
            mat[i, j] = mat_gram[a, b]
    return mat


def two_sample_test(mat_gram, alpha=0.05, num_m=None, num_test=1000):
    vec_wchi = hist_wchi(mat_gram)
    vec_p_value = np.empty(num_test)
    for temp_test in range(num_test):
        mat_reduced = extract_submat(mat_gram, num_m)
        value_mmd = n_mmd(mat_reduced)
        vec_temp = np.where(vec_wchi > value_mmd)[0]
        vec_p_value[temp_test] = len(vec_temp) / len(vec_wchi)
    return vec_p_value, len(np.where(vec_p_value < alpha)[0]) / num_test


pwgk = [True, False][0]
landscape = [True, False][0]
pssk = [True, False][0]

CONST_PD = 1
CONST_ALPHA = 0.05
CONST_IID = 100
CONST_M = 20
type_one = [False, True][0]

main = ["lattice", "matern"][1]
name_dir = "../data/" + main

# type of P and Q
if main == "lattice":
    list_list_pcd = [["square_014", "gauss_010"], ["square_017", "gauss_010"],
                     ["square_020", "gauss_010"]]
    CONST_PCD = 2
    CONST_SIDE = 20
    name_parameter = "pcd%s_side%s_iid%s" % (CONST_PCD, CONST_SIDE, CONST_IID)
else:   # matern
    list_list_pcd = [["type_0", "type_1"], ["type_0", "type_2"],
                     ["type_1", "type_2"]]
    CONST_PCD = 2
    CONST_LAMBDA = 100
    CONST_WIDTH = 1
    CONST_DISTANCE = 0.05
    name_parameter = "pcd%s_lambda%s_width%s_distance%s_iid%s" % (
        CONST_PCD, CONST_LAMBDA, CONST_WIDTH,
        str(CONST_DISTANCE).replace(".", ""), CONST_IID)

for list_name_pcd in list_list_pcd:
    # make directory to save the Gram matrix
    name_dir_gram = "%s/%s_pd%s_%s_vs_%s" % (
        name_dir, name_parameter, CONST_PD, list_name_pcd[0],
        list_name_pcd[1])
    tda.os_mkdir(name_dir_gram)

    # compute/import parameters
    temp_v = tda.import_parameters(name_dir, name_parameter, list_name_pcd,
                                   CONST_IID, CONST_PD)
    min_birth, max_death, med_pers, max_pers, med_sigma = temp_v

    # import diagrams of P and Q
    list_diagram_xy = []
    for name_pcd in list_name_pcd:
        name_data = "%s_%s" % (name_parameter, name_pcd)
        diagram = PD(name_dir, name_data, CONST_IID, CONST_PD)
        list_diagram_xy.extend(diagram.data)

    print(main, list_name_pcd)

    if pwgk:
        for name_large in ["", "_small", "_large"]:
            print("===============")
            for name_weight in ["one", "linear", "arctan"]:
                # define a kernel and a weight function
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

                # check type I error
                if type_one:
                    mat_linear_pwgk = mat_linear_pwgk[CONST_IID:2 * CONST_IID,
                                                      CONST_IID:2 * CONST_IID]
                else:
                    pass

                # define the Gram matrix on persistence diagrams
                name_rkhs = ["Linear", "Gaussian"][1]
                mat_gram_pwgk = tda.matrix_gram(
                    mat_linear_pwgk, name_rkhs)[0]

                # test result
                vec_p, num_reject = two_sample_test(
                    mat_gram_pwgk, CONST_ALPHA, CONST_M)
                print(name_weight, name_large)
                print(np.mean(vec_p), num_reject)

    if landscape:
        # compute/import the linear Gram matrix
        name_linear_landscape = "%s/gram_mat_landscape_Linear.txt" % (
            name_dir_gram)
        if not os.path.exists(name_linear_landscape):
            landscape = PL(list_diagram_xy, name_rkhs="Linear")
            mat_linear_land = landscape.gram_matrix()
            np.savetxt(name_linear_landscape, mat_linear_land)
        else:
            mat_linear_land = np.loadtxt(name_linear_landscape)

        # check type I error
        if type_one:
            mat_linear_land = mat_linear_land[CONST_IID:2 * CONST_IID,
                                              CONST_IID:2 * CONST_IID]
        else:
            pass

        # define the Gram matrix on persistence diagrams
        name_rkhs = ["Linear", "Gaussian"][0]
        mat_gram_landscape = tda.matrix_gram(
            mat_linear_land, name_rkhs)[0]

        # test result
        vec_p, num_reject = two_sample_test(
            mat_gram_landscape, CONST_ALPHA, CONST_M)

        print("===============")
        print("landscape")
        print(np.mean(vec_p), num_reject)

    if pssk:
        for name_large in ["", "_small", "_large"]:
            if name_large == "_small":
                sigma_pssk = med_sigma / 10
            elif name_large == "_large":
                sigma_pssk = med_sigma * 10
            else:
                sigma_pssk = med_sigma

            sigma_pssk /= np.sqrt(2)

            # compute/import the linear Gram matrix
            name_linear_pssk = "%s/gram_mat_pssk%s_Linear.txt" % (
                name_dir_gram, name_large)

            if not os.path.exists(name_linear_pssk):
                pssk = PSSK(list_diagram_xy, sigma_pssk, name_rkhs="Linear")
                mat_linear_pssk = pssk.gram_matrix()
                np.savetxt(name_linear_pssk, mat_linear_pssk)
            else:
                mat_linear_pssk = np.loadtxt(name_linear_pssk)

            # check type I error
            if type_one:
                mat_linear_pssk = mat_linear_pssk[CONST_IID:2 * CONST_IID,
                                                  CONST_IID:2 * CONST_IID]
            else:
                pass

            # define the Gram matrix on persistence diagrams
            name_rkhs = ["Linear", "Gaussian"][0]
            mat_gram_pssk = tda.matrix_gram(
                mat_linear_pssk, name_rkhs)[0]

            # test result
            vec_p, num_reject = two_sample_test(
                mat_gram_pssk, CONST_ALPHA, CONST_M)

            print("pssk", name_large)
            print(np.mean(vec_p), num_reject)
