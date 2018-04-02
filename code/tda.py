from tqdm import tqdm
from scipy import ndimage
from scipy import integrate
from matplotlib import pyplot as plt

import numpy as np
import math
import os
import matplotlib.colors


def os_mkdir(name_dir_save):
    if not os.path.exists(name_dir_save):
        os.mkdir(name_dir_save)


def function_kernel(name_kernel, sigma=1.0):
    if name_kernel == "Gaussian":
        def func_kernel(bd_1, bd_2):
            dif_vec = bd_1 - bd_2
            squared_distance = dif_vec[0] ** 2 + dif_vec[1] ** 2
            return np.exp(-1.0 * squared_distance / (2.0 * math.pow(sigma, 2)))
    else:  # linear kernel
        def func_kernel(bd_1, bd_2):
            return np.dot(bd_1, bd_2)
    return func_kernel


def function_weight(name_weight, arc_c=1.0, arc_p=5.0, lin_el=1.0):
    if name_weight == "arctan":
        def func_weight(bd):
            return np.maximum(
                np.arctan(math.pow((bd[1] - bd[0]) / arc_c, arc_p)), 0.0)
    elif name_weight == "linear":
        def func_weight(bd):
            return np.maximum(np.minimum((bd[1] - bd[0]) / lin_el, 1.0), 0.0)
    else:  # unweighted
        def func_weight(bd):
            return 1.0
    return func_weight


def parameters(list_pd):
    num_pd = len(list_pd)
    min_birth = np.empty(num_pd)
    max_death = np.empty(num_pd)
    median_pers = np.empty(num_pd)
    max_pers = np.empty(num_pd)
    for i in range(num_pd):
        vec_birth = list_pd[i][:, 0]
        vec_death = list_pd[i][:, 1]
        vec_pers = vec_death - vec_birth
        min_birth[i] = min(vec_birth)
        max_death[i] = max(vec_death)
        median_pers[i] = np.median(vec_pers)
        max_pers[i] = max(vec_pers)
    return min(min_birth), max(max_death), np.median(median_pers), max(max_pers)


def parameter_sigma(list_pd):
    def sigma_one(diagram):
        num_points = int(diagram.size / 2)
        if num_points > 1:
            vec = np.empty(int(num_points * (num_points - 1) / 2))
            int_temp = 0
            for i in range(1, num_points):
                for j in range(i):
                    vec[int_temp] = np.linalg.norm(
                        diagram[i, :] - diagram[j, :])
                    int_temp += 1
            sigma = np.median(vec)
        else:
            sigma = 0
        return sigma

    num_pd = len(list_pd)
    vec_sigma = np.empty(num_pd)
    print("Computing sigma...")
    for k in range(num_pd):
        print(k)
        vec_sigma[k] = sigma_one(list_pd[k])
    return vec_sigma


def import_parameters(name_dir, name_parameter, list_name_pcd, num_pd, dim_pd):
    num_type = len(list_name_pcd)
    vec_birth = np.empty(num_type)
    vec_death = np.empty(num_type)
    vec_med_pers = np.empty(num_type)
    vec_max_pers = np.empty(num_type)
    vec_sigma = np.empty(num_type)
    for temp_test in range(num_type):
        name_data = "%s_%s" % (name_parameter, list_name_pcd[temp_test])
        name_sigma = "%s/%s/parameters_pd%s.txt" % (
            name_dir, name_data, dim_pd)
        if not os.path.exists(name_sigma):
            diagram = PD(name_dir, name_data, num_pd, dim_pd)
            list_diagram = diagram.data
            min_birth, max_death, med_pers, max_pers = parameters(list_diagram)
            sigma = np.median(parameter_sigma(list_diagram))
            np.savetxt(name_sigma,
                       [min_birth, max_death, med_pers, max_pers, sigma],
                       delimiter='\t')
        else:
            min_birth, max_death, med_pers, max_pers, sigma = np.loadtxt(
                name_sigma, delimiter='\t')
        vec_birth[temp_test] = min_birth
        vec_death[temp_test] = max_death
        vec_med_pers[temp_test] = med_pers
        vec_max_pers[temp_test] = max_pers
        vec_sigma[temp_test] = sigma

    return [np.min(vec_birth), np.max(vec_death), np.median(vec_med_pers),
            np.median(vec_max_pers), np.mean(vec_sigma)]


def matrix_squared_distance(mat_linear):
    num_pd = len(mat_linear)
    mat = np.zeros((num_pd, num_pd))
    for i in range(num_pd):
        for j in range(i):
            mat[i, j] = (
                mat_linear[i, i] + mat_linear[j, j] - 2 * mat_linear[i, j])
            mat[j, i] = mat[i, j]

    vec = np.empty(int(num_pd * (num_pd - 1) / 2))
    a = 0
    for i in range(num_pd):
        for j in range(i):
            vec[a] = mat[i, j]
            a += 1
    return mat, np.median(vec)


def matrix_gram(mat_linear, name_rkhs="Gaussian", normalize=True):
    if normalize:
        max_mat = mat_linear.max()
        if max_mat > 1e+4:
            mat_linear /= max_mat
        elif max_mat < 1e-4:
            mat_linear /= max_mat
        else:
            pass
    else:
        pass

    mat_distance, tau = matrix_squared_distance(mat_linear)
    if name_rkhs == "Gaussian":
        num_pd = len(mat_distance)
        mat_gram = np.empty((num_pd, num_pd))
        for i in range(num_pd):
            for j in range(i + 1):
                mat_gram[i, j] = np.exp(-1.0 * mat_distance[i, j] / (2.0 * tau))
                mat_gram[j, i] = mat_gram[i, j]
    else:
        mat_gram = mat_linear
    return mat_gram, mat_distance, tau


"""
================================================================================
Persistence diagram
================================================================================

data    = { D_1, ..., D_n } as list of array
norm(p) = ( Pers_p(D_1), ..., Pers_p(D_n) ) as vector
"""


def save_plot_list_pd(list_pd, name_dir_save, range_bd):
    os_mkdir(name_dir_save)
    min_birth, max_death = range_bd

    for k in range(len(list_pd)):
        diagram = list_pd[k]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        hist = ax.hist2d(diagram[:, 0], diagram[:, 1],
                         bins=[np.linspace(min_birth, max_death, 100),
                               np.linspace(min_birth, max_death, 100)],
                         norm=matplotlib.colors.LogNorm())
        m = int(hist[1].max() + 1)
        if m > 3:
            hist[3].set_clim(1, m)
        else:
            hist[3].set_clim(1, int(10**m))
        fig.colorbar(hist[3], ax=ax)
        x = np.linspace(min_birth, max_death, 2)
        plt.plot(x, x, "k-", linewidth=0.3)
        plt.savefig("%s/pd_%s.png" % (name_dir_save, k))
        plt.close()


class PD:
    def __init__(self, name_dir, name_data, num_pd, dim_pd, scale=True):
        self.name_dir = name_dir
        self.name_data = name_data
        self.num_pd = num_pd
        self.dim_pd = dim_pd
        self.scale = scale
        self.data = self.__list_pd()

    def __list_pd(self):
        list_pd = []
        process_bar = tqdm(total=self.num_pd)
        for k in range(self.num_pd):
            process_bar.set_description("Importing PD: %s" % k)
            diagram = np.loadtxt("%s/%s/pcd_pd/dim%s_%s.txt" % (
                self.name_dir, self.name_data, self.dim_pd, k))
            if self.scale:  # (b,d)
                list_pd.append(np.sqrt(diagram.reshape(-1, 2)))
            else:  # CGAL default is (b^2,d^2), not (b,d)
                list_pd.append(diagram.reshape(-1, 2))
            process_bar.update(1)
        process_bar.close()
        return list_pd

    def norm(self, p=1):
        vec = np.empty(self.num_pd)
        for k in range(self.num_pd):
            if p == 0:
                vec[k] = int(self.data[k].size / 2)
            else:
                vec_pers = self.data[k][:, 1] - self.data[k][:, 0]
                vec[k] = np.linalg.norm(vec_pers, p)
        return vec


"""
================================================================================
Persistence image
================================================================================

hist                = { hist(D_1), ..., hist(D_n) } as list of matrix
data                = { PI(D_1), ..., PI(D_n) } as list of matrix
data_vector()       = { vPI(D_1), ..., vPI(D_n)} as list of vector
make_image(D)       = PI(D) as matrix
"""


def average_pi(list_pi):
    num_pi = len(list_pi)
    mat = np.zeros(list_pi[0].shape)
    for k in range(num_pi):
        mat += list_pi[k]
    return mat / num_pi


def plot_image(mat, range_bd, max_image=None):
    num_mesh = mat.shape[0]
    diagonal = np.linspace(0, num_mesh - 1, 2)

    num_grid = 8
    x = np.linspace(0, num_mesh - 1, num_grid)
    birth, death = range_bd
    if death - birth > 1:
        y = np.round(np.linspace(birth, death, num_grid), 1)
    else:
        y = np.round(np.linspace(birth, death, num_grid), 2)

    plt.figure()
    plt.imshow(np.transpose(mat), interpolation="nearest",
               origin='lower', cmap="YlOrRd")
    if max_image is None:
        max_image = mat.max()
        plt.colorbar()
    else:
        plt.clim(0, max_image)
    plt.plot(diagonal, diagonal, "k-", linewidth=0.3)
    plt.xticks(x, y)
    plt.yticks(x[1:], y[1:])


def save_plot_list_pi(list_pi, name_dir_save, range_bd, max_image=None):
    os_mkdir(name_dir_save)
    num_pd = len(list_pi)
    if max_image is None:
        temp = 0
        for k in range(num_pd):
            if list_pi[k].max() > temp:
                temp = list_pi[k].max()
        max_image = temp
    elif max_image == "each":
        pass
    else:
        pass

    for k in range(num_pd):
        plt.figure()
        plot_image(list_pi[k], range_bd, max_image=max_image)
        plt.savefig("%s/image_%s.png" % (name_dir_save, k))
        plt.close()

    return max_image


def save_txt_list_pi(list_pi, name_dir_save):
    os_mkdir(name_dir_save)
    for k in range(len(list_pi)):
        np.savetxt("%s/image_%s.txt" % (name_dir_save, k), list_pi[k])


class PI:
    def __init__(self, list_pd, func_weight, num_mesh=80, sigma=0.1,
                 range_bd=None, name_rkhs="Linear"):
        self.__list_pd = list_pd
        self.num_pd = len(list_pd)
        self.num_mesh = num_mesh
        self.sigma = sigma
        self.range_bd = range_bd
        self.name_rkhs = name_rkhs

        self.tau = None
        self.mat_distance = None

        if self.range_bd is None:
            min_birth, max_death = parameters(list_pd)[0:2]
            width = (max_death - min_birth) * 0.1
            self.range_bd = (min_birth - width, max_death + width)
        else:
            pass

        self.min, self.max = self.range_bd
        self.len_mesh = (self.max - self.min) / self.num_mesh
        self.scaled_sigma = self.sigma / self.len_mesh
        self.bins = [np.linspace(self.min, self.max, self.num_mesh + 1),
                     np.linspace(self.min, self.max, self.num_mesh + 1)]

        self.mat_weight = np.zeros((self.num_mesh, self.num_mesh))
        for j in range(self.num_mesh):
            for i in range(j + 1):
                bd = ((i + 0.5) * self.len_mesh + self.min,
                      (j + 0.5) * self.len_mesh + self.min)
                self.mat_weight[i, j] = func_weight(bd)

        self.hist = self.__list_hist()
        self.data = self.__list_pi()

        self.max = 0
        for k in range(self.num_pd):
            temp = self.data[k].max()
            if temp > self.max:
                self.max = temp

    def __list_hist(self):
        list_hist = []
        for k in range(self.num_pd):
            birth = self.__list_pd[k][:, 0]
            death = self.__list_pd[k][:, 1]
            list_hist.append(np.histogram2d(birth, death, bins=self.bins)[0])
        return list_hist

    def __list_pi(self):
        list_pi = []
        list_hist = self.hist
        process_bar = tqdm(total=self.num_pd)
        for k in range(self.num_pd):
            process_bar.set_description("Computing PI: %s" % k)
            list_pi.append(ndimage.filters.gaussian_filter(
                np.multiply(self.mat_weight, list_hist[k]),
                sigma=self.scaled_sigma))
            process_bar.update(1)
        process_bar.close()
        return list_pi

    def data_vector(self):
        array_vec = np.empty((self.num_pd, self.num_mesh ** 2))
        for k in range(self.num_pd):
            array_vec[k] = self.data[k].reshape(-1)
        return array_vec

    def __matrix_linear(self):
        list_vec = self.data_vector()
        mat = np.empty((self.num_pd, self.num_pd))
        process_bar = tqdm(total=int(self.num_pd * (self.num_pd + 1) / 2))
        for i in range(self.num_pd):
            for j in range(i + 1):
                process_bar.set_description("Computing PI: (%s, %s)" % (i, j))
                mat[i, j] = np.inner(list_vec[i], list_vec[j])
                mat[j, i] = mat[i, j]
                process_bar.update(1)
        process_bar.close()
        return mat

    def gram_matrix(self):
        mat_linear = self.__matrix_linear()
        mat_gram, self.mat_distance, self.tau = matrix_gram(
            mat_linear, self.name_rkhs)
        return mat_gram

    def make_image(self, diagram):
        birth = diagram[:, 0]
        death = diagram[:, 1]
        return ndimage.filters.gaussian_filter(
            np.multiply(self.mat_weight,
                        np.histogram2d(birth, death, bins=self.bins)[0]),
            sigma=self.scaled_sigma)

    def kernel_rkhs(self, diagram_1, diagram_2, name_rkhs=None):
        vec_pi_1 = self.make_image(diagram_1).reshape(-1)
        vec_pi_2 = self.make_image(diagram_2).reshape(-1)
        value_linear = np.inner(vec_pi_1, vec_pi_2)

        if name_rkhs is None:
            name_rkhs = self.name_rkhs
        else:
            pass

        if name_rkhs == "Gaussian":
            a = np.inner(vec_pi_1, vec_pi_1)
            b = np.inner(vec_pi_2, vec_pi_2)
            return np.exp(-(a + b - 2.0 * value_linear) / (2.0 * self.tau))
        else:
            return value_linear


"""
================================================================================
Persistence weighted Gaussian kernel
================================================================================

vector_weight(D) = { w(x_1), ..., w(x_n)} as vector
gram_matrix() = (K(D_i, D_j)) as matrix
self.mat_distance = (d(D_i, D_j)^2) as matrix

kernel_rkhs(D, E, name_rkhs) = K(D, E) 
"""


def matrix_rff(list_pd, list_weight, sigma, num_rff):
    num_pd = len(list_pd)
    mat = np.empty((num_pd, num_rff))
    z = np.random.multivariate_normal(
        [0.0, 0.0], [[sigma ** (-2.0), 0.0], [0.0, sigma ** (-2.0)]], num_rff)
    b = np.random.uniform(0.0, 2.0 * math.pi, num_rff)
    process_bar = tqdm(total=num_pd)
    for k in range(num_pd):
        process_bar.set_description("Computing RFF: %s" % k)
        mat[k, :] = np.dot(list_weight[k], np.cos(np.inner(list_pd[k], z) + b))
        process_bar.update(1)
    process_bar.close()
    return mat


class PWGK:
    def __init__(self, list_pd, func_kernel, func_weight, sigma=0.1,
                 name_rkhs="Gaussian", approx=True, num_rff=int(1e+4)):
        self.__list_pd = list_pd
        self.num_pd = len(list_pd)
        self.func_kernel = func_kernel
        self.sigma = sigma
        self.weight = func_weight
        self.name_rkhs = name_rkhs
        self.approx = approx
        self.num_rff = num_rff

        self.tau = None
        self.mat_distance = None

    def vector_weight(self, diagram):
        num_point = int(diagram.size / 2)
        vec = np.empty(num_point)
        for k in range(num_point):
            vec[k] = self.weight(diagram[k, :])
        return vec

    def __list_vector_weight(self):
        list_weight = []
        for i in range(self.num_pd):
            list_weight.append(self.vector_weight(self.__list_pd[i]))
        return list_weight

    def __kernel_linear(self, diagram_1, diagram_2, vec_weight_1, vec_weight_2):
        s = 0.0
        num_point_1 = int(diagram_1.size / 2)
        num_point_2 = int(diagram_2.size / 2)
        for i in range(num_point_1):
            for j in range(num_point_2):
                s += (vec_weight_1[i] * vec_weight_2[j]
                      * self.func_kernel(diagram_1[i, :], diagram_2[j, :]))
        return s

    def __matrix_linear(self):
        mat = np.empty((self.num_pd, self.num_pd))
        list_weight = self.__list_vector_weight()
        process_bar = tqdm(total=int(self.num_pd * (self.num_pd + 1) / 2))
        for i in range(self.num_pd):
            for j in range(i + 1):
                process_bar.set_description("Computing PWGK: (%s, %s)" % (i, j))
                mat[i, j] = self.__kernel_linear(
                    self.__list_pd[i], self.__list_pd[j],
                    list_weight[i], list_weight[j])
                mat[j, i] = mat[i, j]
                process_bar.update(1)
        process_bar.close()
        return mat

    def gram_matrix(self):
        if self.approx:
            mat_rff = matrix_rff(self.__list_pd, self.__list_vector_weight(),
                                 self.sigma, self.num_rff)
            mat_linear = np.inner(mat_rff, mat_rff)
        else:
            mat_linear = self.__matrix_linear()

        mat_gram, self.mat_distance, self.tau = matrix_gram(
            mat_linear, self.name_rkhs)
        return mat_gram

    def kernel_rkhs(self, diagram_1, diagram_2, name_rkhs=None):
        vec_weight_1 = self.vector_weight(diagram_1)
        vec_weight_2 = self.vector_weight(diagram_2)
        value_linear = self.__kernel_linear(
            diagram_1, diagram_2, vec_weight_1, vec_weight_2)

        if name_rkhs is None:
            name_rkhs = self.name_rkhs
        else:
            pass

        if name_rkhs == "Gaussian":
            a = self.__kernel_linear(
                diagram_1, diagram_1, vec_weight_1, vec_weight_1)
            b = self.__kernel_linear(
                diagram_2, diagram_2, vec_weight_2, vec_weight_2)
            return np.exp(-(a + b - 2.0 * value_linear) / (2.0 * self.tau))
        else:
            return value_linear


"""
================================================================================
Persistence landscape
================================================================================

mat_gram = gram_matrix()
mat_gram                     = (K(D_i, D_j)) as matrix
self.mat_distance            = (d(D_i, D_j)^2) as matrix

kernel_rkhs(D, E, name_rkhs) = K(D, E)
"""


def vector_landscape(diagram, t):
    vec = np.sort(
        np.maximum(np.minimum(t - diagram[:, 0], diagram[:, 1] - t), 0.0))[::-1]
    idx_zero = len(np.where(vec != 0))
    return vec, idx_zero


def function_landscape(diagram):
    return lambda k, t: vector_landscape(diagram, t)[0][k]


def plot_landscape(diagram, range_bd, max_k=1, num_slice=100, ymax=0.1):
    bins = np.linspace(range_bd[0], range_bd[1], num_slice)
    plt.figure()
    for k in range(max_k):
        vec = np.zeros(num_slice)
        for i, t in enumerate(bins):
            vec[i] = function_landscape(diagram)(k, t)
        plt.plot(bins, vec)
    plt.ylim(0, ymax)


def plot_landscape_average(list_pd, range_bd, max_k=1, num_slice=100, ymax=0.1):
    bins = np.linspace(range_bd[0], range_bd[1], num_slice)
    plt.figure()
    for k in range(max_k):
        vec = np.zeros(num_slice)
        for i, t in enumerate(bins):
            for j in range(len(list_pd)):
                vec[i] += function_landscape(list_pd[j])(k, t)
            vec[i] /= len(list_pd)
        plt.plot(bins, vec)
    plt.ylim(0, ymax)


def save_plot_list_landscape(list_pd, range_bd, name_dir_save, max_k=1,
                             num_slice=100, ymax=0.1):
    os_mkdir(name_dir_save)
    for k in range(len(list_pd)):
        plot_landscape(list_pd[k], range_bd, max_k, num_slice, ymax)
        plt.savefig("%s/landscape_%s.png" % (name_dir_save, k))
        plt.close()


class PL:
    def __init__(self, list_pd, name_rkhs="Linear"):
        self.__list_pd = list_pd
        self.num_pd = len(list_pd)
        self.min, self.max = parameters(self.__list_pd)[0:2]
        self.range_bd = (self.min, self.max)
        self.name_rkhs = name_rkhs

        self.tau = None
        self.mat_distance = None

    def __kernel_linear(self, diagram_1, diagram_2):
        def inner_landscape(t):
            vec_1, idx_1 = vector_landscape(diagram_1, t)
            vec_2, idx_2 = vector_landscape(diagram_2, t)
            idx_zero = np.maximum(idx_1, idx_2)
            return np.dot(vec_1[0:idx_zero], vec_2[0:idx_zero])

        return integrate.quad(inner_landscape, self.min, self.max)[0]

    def __matrix_linear(self):
        mat = np.empty((self.num_pd, self.num_pd))
        process_bar = tqdm(total=int(self.num_pd * (self.num_pd + 1) / 2))
        for i in range(self.num_pd):
            for j in range(i + 1):
                process_bar.set_description("Computing PL: (%s, %s)" % (i, j))
                mat[i, j] = self.__kernel_linear(
                    self.__list_pd[i], self.__list_pd[j])
                mat[j, i] = mat[i, j]
                process_bar.update(1)
        process_bar.close()
        return mat

    def gram_matrix(self):
        mat_linear = self.__matrix_linear()
        mat_gram, self.mat_distance, self.tau = matrix_gram(
            mat_linear, self.name_rkhs)
        return mat_gram

    def kernel_rkhs(self, diagram_1, diagram_2, name_rkhs=None):
        value_linear = self.__kernel_linear(diagram_1, diagram_2)

        if name_rkhs is None:
            name_rkhs = self.name_rkhs
        else:
            pass

        if name_rkhs == "Gaussian":
            a = self.__kernel_linear(diagram_1, diagram_1)
            b = self.__kernel_linear(diagram_2, diagram_2)
            return np.exp(-(a + b - 2.0 * value_linear) / (2.0 * self.tau))
        else:
            return value_linear


"""
================================================================================
Persistence scale-space kernel
================================================================================

mat_gram = gram_matrix()
mat_gram                     = (K(D_i, D_j)) as matrix
self.mat_distance            = (d(D_i, D_j)^2) as matrix

kernel_rkhs(D, E, name_rkhs) = K(D, E) 
"""


def diagram_transpose(diagram):
    num_points = int(diagram.size / 2)
    diag_transpose = np.empty((num_points, 2))
    for i in range(num_points):
        diag_transpose[i, 0] = diagram[i, 1]
        diag_transpose[i, 1] = diagram[i, 0]
    return diag_transpose


class PSSK:
    def __init__(self, list_pd, sigma, name_rkhs="Linear", approx=True,
                 num_rff=int(1e+4)):
        self.__list_pd = list_pd
        self.num_pd = len(list_pd)
        self.sigma = 2.0 * sigma
        self.gauss = function_kernel("Gaussian", self.sigma)
        self.name_rkhs = name_rkhs
        self.approx = approx
        self.num_rff = num_rff

        self.tau = None
        self.mat_distance = None

    def __list_tilde(self):
        list_tilde = []
        for i in range(self.num_pd):
            list_tilde.append(
                np.r_[self.__list_pd[i], diagram_transpose(self.__list_pd[i])])
        return list_tilde

    def __list_pss_weight(self):
        list_weight = []
        for i in range(self.num_pd):
            num_points = int(self.__list_pd[i].size / 2)
            list_weight.append(
                np.r_[np.ones(num_points), -1 * np.ones(num_points)])
        return list_weight

    def __kernel_linear(self, diagram_1, diagram_2):
        s = 0.0
        num_point_1 = int(diagram_1.size / 2)
        num_point_2 = int(diagram_2.size / 2)
        transpose_diagram_2 = diagram_transpose(diagram_2)
        for i in range(num_point_1):
            for j in range(num_point_2):
                s += (self.gauss(diagram_1[i, :], diagram_2[j, :])
                      - self.gauss(diagram_1[i, :], transpose_diagram_2[j, :]))
        return s

    def __matrix_linear(self):
        mat = np.empty((self.num_pd, self.num_pd))
        process_bar = tqdm(total=int(self.num_pd * self.num_pd - 1 / 2))
        for i in range(self.num_pd):
            for j in range(i + 1):
                process_bar.set_description("Computing PSSK: (%s, %s)" % (i, j))
                mat[i, j] = self.__kernel_linear(
                    self.__list_pd[i], self.__list_pd[j])
                mat[j, i] = mat[i, j]
                process_bar.update(1)
        process_bar.close()
        return mat

    def gram_matrix(self):
        if self.approx:
            mat_rff = matrix_rff(self.__list_tilde(), self.__list_pss_weight(),
                                 self.sigma, self.num_rff)
            mat_linear = np.inner(mat_rff, mat_rff)
        else:
            mat_linear = self.__matrix_linear()

        mat_gram, self.mat_distance, self.tau = matrix_gram(
            mat_linear, self.name_rkhs)
        return mat_gram

    def kernel_rkhs(self, diagram_1, diagram_2, name_rkhs=None):
        value_linear = self.__kernel_linear(diagram_1, diagram_2)
        if name_rkhs is None:
            name_rkhs = self.name_rkhs
        else:
            pass

        if name_rkhs == "Gaussian":
            a = self.__kernel_linear(diagram_1, diagram_1)
            b = self.__kernel_linear(diagram_2, diagram_2)
            return np.exp(-(a + b - 2.0 * value_linear) / (2.0 * self.tau))
        else:
            return value_linear
