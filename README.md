Edited by Genki Kusano <https://sites.google.com/site/genkikusano/home>

This page is for python codes used in "Persistence Weighted Gaussian Kernel for Probability Distributions on the Space of Persistence Diagrams " <https://arxiv.org/abs/1803.08269>  
For PWGK, peristence landscape, persistence image, kernel PCA, kernel Fisher discriminat ratio, please see "Kernel Method for Persistence Diagrams via Kernel Embedding and Weight Factor" <https://arxiv.org/abs/1706.03472> , which will be appear in JMLR.  
I have confirmed the codes work on MacOS High Sierra and python 3.6.1

If you have your persistence diagrams, skip to 3.  
Then, please rename your persistence diagrams as   
data/{type of data}/{name of data}/pcd/dim{dimension of PD}_{data index}.txt   

You can also use data.zip as an example data.  
Due to the file size restriction, only matern data is available.  

# 1.Prepare point data
You can try make_lattice.py or make_matern.py.  
The point data will be saved as data/matern/pcd2_lambda100_width1_distance005_iid100_type_0/pcd/pcd_0.txt


# 2.Compute the persistence diagrams
You can try run_lattice.sh or run_matern.sh.  
In order to use the shell script, you need to install HomCloud (http://www.wpi-aimr.tohoku.ac.jp/hiraoka_labo/homcloud/index.en.html)  
You can check the persistence diagram of data/matern/pcd2_lambda100_width1_distance005_iid100_type_0/pcd/pcd_0.txt in data/matern/pcd2_lambda100_width1_distance005_iid100_type_0/pcd/dim1_0.txt (dim1 means 1 dimensional persistence diagram)


# (Optional) Let's see the persistence diagrams as histogram, visualized PWGK and persistence landscape.
You can try plot_image.py.  
The point data will be saved as data/lattice/plot_pd1_pcd2_side20_iid100  
hist: histogram  
image: visualized PWGK (= persistence image)  
landscape: persistence landscape with k=1, i.e., one-variable function $\lambda(1, t)$ w.r.t $t$  
The last image_linear/image_ave_hoge.png or landscape/landscape_ave_hoge.png mean the averaged version.


# 3.1 Compute kernel principal component analysis (KPCA) by PWGK
Try kpca.py  
In the case of matern, type_1 and type_2 are different data set.  
1st figure: gram matrix  
2nd figure: result of kernel discriminant Fisher ratio  
3rd figure: result of KPCA on 2 dim plot  
4th figure: result of KPCA on 3 dim plot  
You can check the contribution rates of KPCA in the python console.  

If you change 1.136 to 1.135 by changing the comment out, you can see the difficulty to distinguish type_0 (Poisson point process) and type_2.

# 3.2 Compute kernel two sample test
Try main.py  
The results in python console  
"""  
matern ['type_0', 'type_2']  
one  
0.0011803 1.0  
linear   
0.1276194 0.456  
arctan   
0.0663033 0.655  
"""  
means a null hypothesis is "type_0" = type_2"  
one, linear, arctan mean the weight function w(b,d) = 1, d-b, arctan(C(d-b)^p), resp.  
The left value is the average of p-value and the right is the statistical power (probability to reject the null distribution correctly).  

If you are concern the Type I error, please change False to True in 1.90, and then the right value means the probability that Type I error happens because the null hypothesis is  "type_2" = "type_2" and the rejection is the Type I error.


# 3.3 Compute the confidence intervals
Try bootstrap.py  
The results in python console  
"""  
matern ['type_0', 'type_2'] 100 [0.07, 0.075] 0.01  
95 percent confidence interval  
[ a b ]  
[ c d ]  
"""  
means the confidence interval of 'type_0' is [a,c] and 'type_2' is [b,d].  
You can also check data/lattice/bootstrap_one.png

# 3.* Try what you want to do by kernel method
In kpca.py or main.py, we compute the Gram matrix of the PWGK as "mat_gram_pwgk".  
If you want to apply another kernel methods, you can use this Gram matrix.

