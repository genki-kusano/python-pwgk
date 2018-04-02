from matplotlib import pyplot as plt

import numpy as np
import os
import tda


def poisson_pp(intensity=100, width=1, dim_pcd=2):
    num_poisson = np.random.poisson(intensity)
    x = np.random.uniform(0, width, num_poisson)
    y = np.random.uniform(0, width, num_poisson)
    mat = np.c_[x, y]
    if dim_pcd == 3:
        z = np.random.uniform(0, width, num_poisson)
        mat = np.c_[mat, z]
    return mat


def matern_one_pp(pcd, distance=0.1):
    num_points = pcd.shape[0]
    list_thin = []
    for i in range(num_points):
        for j in range(num_points):
            r = np.linalg.norm(pcd[i] - pcd[j])
            if 0 < r < distance:
                list_thin.append(i)
                break
    return np.delete(pcd, list_thin, 0)


def matern_two_pp(pcd, distance=0.1):
    num_points = pcd.shape[0]
    vec_weight = np.random.uniform(0, 1, num_points)

    list_thin = []
    for i in range(num_points):
        for j in range(num_points):
            r = np.linalg.norm(pcd[i] - pcd[j])
            t = vec_weight[i] - vec_weight[j]
            if 0 < r < distance and t < 0:
                list_thin.append(i)
                break

    return np.delete(pcd, list_thin, 0)


def save_points(list_lat, width, name_save):
    num_lattice = len(list_lat)
    plt.figure()
    for k in range(num_lattice):
        point = list_lat[k]
        plt.plot(point[0], point[1], "bo")
    plt.xlim(-0.1 * width, 1.1 * width)
    plt.ylim(-0.1 * width, 1.1 * width)
    plt.savefig(name_save)
    plt.close()


NAME_DIR = "../data"
if not os.path.exists(NAME_DIR):
    os.mkdir(NAME_DIR)
NAME_DIR += "/matern"
if not os.path.exists(NAME_DIR):
    os.mkdir(NAME_DIR)

main = [True, False][0]
plot = [True, False][0]
CONST_PCD = 2
CONST_LAMBDA = 100
CONST_WIDTH = 1
CONST_DISTANCE = 0.05

if main:
    CONST_IID = 100

    # make directory to save point sets as txt file
    name_parameter = "pcd%s_lambda%s_width%s_distance%s_iid%s" % (
        CONST_PCD, CONST_LAMBDA, CONST_WIDTH, "%03d" % (CONST_DISTANCE * 100),
        CONST_IID)

    for temp_type in range(3):
        name_dir_data = "%s/%s_type_%s" % (NAME_DIR, name_parameter, temp_type)
        if not os.path.exists(name_dir_data):
            os.mkdir(name_dir_data)
        name_dir_pcd = "%s/pcd_pd" % name_dir_data
        if not os.path.exists(name_dir_pcd):
            os.mkdir(name_dir_pcd)

    # generate point sets
    for temp_iid in range(CONST_IID):
        print("%s" % temp_iid)
        pcd_poisson = poisson_pp(CONST_LAMBDA, CONST_WIDTH)
        pcd_matern_one = matern_one_pp(pcd_poisson, CONST_DISTANCE)
        pcd_matern_two = matern_two_pp(pcd_poisson, CONST_DISTANCE)

        np.savetxt("%s/%s_type_0/pcd_pd/pcd_%s.txt" % (
            NAME_DIR, name_parameter, temp_iid), np.asarray(pcd_poisson),
                   delimiter='\t')
        np.savetxt("%s/%s_type_1/pcd_pd/pcd_%s.txt" % (
            NAME_DIR, name_parameter, temp_iid), np.asarray(pcd_matern_one),
                   delimiter='\t')
        np.savetxt("%s/%s_type_2/pcd_pd/pcd_%s.txt" % (
            NAME_DIR, name_parameter, temp_iid), np.asarray(pcd_matern_two),
                   delimiter='\t')

if plot:
    CONST_IID = 10

    # make directory to plot point sets as png file
    name_parameter = "pcd%s_lambda%s_width%s_distance%s_iid%s" % (
        CONST_PCD, CONST_LAMBDA, CONST_WIDTH, "%03d" % (CONST_DISTANCE * 100),
        CONST_IID)
    name_dir_png = "%s/plot_point_%s" % (NAME_DIR, name_parameter)
    tda.os_mkdir(name_dir_png)

    # plot point sets
    for temp_iid in range(CONST_IID):
        print("%s" % temp_iid)
        pcd_poisson = poisson_pp(CONST_LAMBDA, CONST_WIDTH)
        pcd_matern_one = matern_one_pp(pcd_poisson, CONST_DISTANCE)
        pcd_matern_two = matern_two_pp(pcd_poisson, CONST_DISTANCE)

        save_points(pcd_poisson, CONST_WIDTH, "%s/%s_%s.png" % (
            name_dir_png, "type_0", temp_iid))
        save_points(pcd_matern_one, CONST_WIDTH, "%s/%s_%s.png" % (
            name_dir_png, "type_1", temp_iid))
        save_points(pcd_matern_two, CONST_WIDTH, "%s/%s_%s.png" % (
            name_dir_png, "type_2", temp_iid))
