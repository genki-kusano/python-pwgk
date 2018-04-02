from tda import PD, PI
from matplotlib import pyplot as plt

import tda
import os

pwgk = [True, False][0]
landscape = [True, False][0]
CONST_PD = 1
CONST_IID = 100
CONST_USE = 20

main = ["lattice", "matern"][1]
name_dir = "../data/" + main

# type of data set
if main == "lattice":
    CONST_PCD = 2
    CONST_SIDE = 20
    CONST_YSCALE = 0.5
    list_name_pcd = ["square_014", "square_017", "square_020", "gauss_010"]
    name_parameter = "pcd%s_side%s_iid%s" % (CONST_PCD, CONST_SIDE, CONST_IID)
else:  # matern
    CONST_PCD = 2
    CONST_LAMBDA = 100
    CONST_WIDTH = 1
    CONST_DISTANCE = 0.05
    CONST_YSCALE = 0.3
    list_name_pcd = ["type_0", "type_1", "type_2"]
    name_parameter = "pcd%s_lambda%s_width%s_distance%s_iid%s" % (
        CONST_PCD, CONST_LAMBDA, CONST_WIDTH,
        str(CONST_DISTANCE).replace(".", ""), CONST_IID)

# make directory to save persistence images
name_dir_png = "%s/plot_pd%s_%s" % (name_dir, CONST_PD, name_parameter)
tda.os_mkdir(name_dir_png)

# compute/import parameters
temp_v = tda.import_parameters(name_dir, name_parameter, list_name_pcd,
                               CONST_IID, CONST_PD)
min_birth, max_death, med_pers, max_pers, med_sigma = temp_v

# fix a plot range of birth-death pairs
range_large = [True, False][0]
if range_large:
    pers = max_death - min_birth
    range_bd = [min_birth - 0.1 * pers, max_death + 0.1 * pers]
else:
    range_bd = [min_birth, max_death]

# import diagrams
list_diagram_all = []
for name_pcd in list_name_pcd:
    name_data = "%s_%s" % (name_parameter, name_pcd)
    diagram = PD(name_dir, name_data, CONST_IID, CONST_PD)
    list_diagram_all.extend(diagram.data[0:CONST_USE])

# save diagrams
name_hist = "%s/hist" % name_dir_png
if not os.path.exists(name_hist):
    tda.save_plot_list_pd(list_diagram_all, name_hist, range_bd)

# persistence image
if pwgk:
    for name_large in ["", "_small"]:
        for name_weight in ["one", "linear", "arctan"]:

            # define a kernel and a weight function
            if name_large == "_small":
                sigma = med_sigma / 10
            else:
                sigma = med_sigma

            func_weight = tda.function_weight(
                name_weight, arc_c=med_pers, arc_p=5, lin_el=max_death)

            name_image = "%s/image_%s%s" % (
                name_dir_png, name_weight, name_large)
            if not os.path.exists(name_image):
                image = PI(list_diagram_all, func_weight, 100, sigma, range_bd)
                list_image_xy = image.data
                max_image = image.max
                tda.save_plot_list_pi(image.data, name_image, range_bd,
                                      max_image)
                for i, name_pcd in enumerate(list_name_pcd):
                    list_image = list_image_xy[int(i * CONST_USE):
                                               int((i + 1) * CONST_USE)]
                    tda.plot_image(tda.average_pi(list_image), range_bd)
                    plt.savefig("%s/image_ave_%s(%s).png" % (
                        name_image, i, name_pcd))
                    plt.close()

# persistence landscape
if landscape:
    ymax = CONST_YSCALE * max_pers
    name_landscape = "%s/landscape" % name_dir_png
    if not os.path.exists(name_landscape):
        tda.save_plot_list_landscape(
            list_diagram_all, range_bd, name_landscape, ymax=ymax)
        for i, name_pcd in enumerate(list_name_pcd):
            list_diagram = list_diagram_all[int(i * CONST_USE):
                                            int((i + 1) * CONST_USE)]
            tda.plot_landscape_average(list_diagram, range_bd, ymax=ymax)
            plt.savefig("%s/landscape_ave_%s(%s).png" % (
                name_landscape, i, name_pcd))
            plt.close()
