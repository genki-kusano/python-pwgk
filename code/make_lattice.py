from matplotlib import pyplot as plt

import numpy as np
import os
import tda


def measure_square(radius, dim_pcd=2):
    vec = np.empty(dim_pcd)
    for i in range(dim_pcd):
        vec[i] = np.random.uniform(-1 * radius, radius)
    return vec


def measure_circle(radius, dim_pcd=2):
    vec = np.random.randn(dim_pcd)
    r = np.linalg.norm(vec)
    if r != 0:
        vec /= r
    return radius * vec * np.power(np.random.random(), 1 / dim_pcd)


def measure_gauss(sigma, dim_pcd=2):
    mean = np.zeros(dim_pcd)
    cov = np.diag(np.ones(dim_pcd) * (sigma ** 2))
    return np.random.multivariate_normal(mean, cov)


def measure(radius, name_measure, dim_pcd=2):
    if name_measure == "gauss":
        return measure_gauss(radius, dim_pcd)
    elif name_measure == "circle":
        return measure_circle(radius, dim_pcd)
    else:  # uniform on square
        return measure_square(radius, dim_pcd)


def lattice(num_side, dim_pcd=2):
    list_lat = []
    if dim_pcd == 3:
        for i in range(num_side):
            for j in range(num_side):
                for k in range(num_side):
                    list_lat.append([i, j, k])
    else:
        for i in range(num_side):
            for j in range(num_side):
                list_lat.append([i, j])
    return list_lat


def save_points(list_lat, width, name_save):
    num_lattice = len(list_lat)
    plt.figure()
    for k in range(num_lattice):
        point = list_lat[k]
        plt.plot(point[0], point[1], "bo")
    plt.xlim(-1, width)
    plt.ylim(-1, width)
    plt.savefig(name_save)
    plt.close()


NAME_DIR = "../data"
if not os.path.exists(NAME_DIR):
    os.mkdir(NAME_DIR)
NAME_DIR += "/lattice"
if not os.path.exists(NAME_DIR):
    os.mkdir(NAME_DIR)

main = [True, False][0]
plot = [True, False][0]
LIST_MEASURE = ["square", "square", "square", "square", "gauss"]
LIST_RADIUS = [np.sqrt(2) * 0.10, np.sqrt(3) * 0.10, 0.20, 0.10]
CONST_PCD = 2

if main:
    CONST_IID = 100
    CONST_SIDE = 20

    num_type = len(LIST_RADIUS)
    for idx_data in range(num_type):

        # make directory to save point sets as txt file
        name_dir_lattice = "%s/pcd%s_side%s_iid%s_%s_%s" % (
            NAME_DIR, CONST_PCD, CONST_SIDE, CONST_IID,
            LIST_MEASURE[idx_data], "%03d" % (LIST_RADIUS[idx_data] * 100))
        if not os.path.exists(name_dir_lattice):
            os.mkdir(name_dir_lattice)

        # generate point sets
        for temp_iid in range(CONST_IID):
            print(temp_iid)
            name_dir_pcd = "%s/pcd_pd" % name_dir_lattice
            if not os.path.exists(name_dir_pcd):
                os.mkdir(name_dir_pcd)

            pcd = lattice(CONST_SIDE)
            for temp_point in range(len(pcd)):
                pcd[temp_point] += measure(LIST_RADIUS[idx_data],
                                           LIST_MEASURE[idx_data])

            np.savetxt("%s/pcd_%s.txt" % (name_dir_pcd, temp_iid),
                       np.asarray(pcd), delimiter='\t')

if plot:
    CONST_IID = 10
    CONST_SIDE = 5

    # make directory to plot point sets as png file
    name_parameter = "pcd%s_side%s_iid%s" % (CONST_PCD, CONST_SIDE, CONST_IID)
    name_dir_png = "%s/plot_point_%s" % (NAME_DIR, name_parameter)
    tda.os_mkdir(name_dir_png)

    # plot point sets
    num_type = len(LIST_RADIUS)
    for idx_data in range(num_type):
        for temp_iid in range(CONST_IID):
            print(temp_iid)

            pcd = lattice(CONST_SIDE)
            for temp_point in range(len(pcd)):
                pcd[temp_point] += measure(LIST_RADIUS[idx_data],
                                           LIST_MEASURE[idx_data])
            save_points(pcd, CONST_SIDE, "%s/%s_%s_%s.png" % (
                name_dir_png, LIST_MEASURE[idx_data],
                "%03d" % (LIST_RADIUS[idx_data] * 100), temp_iid))
