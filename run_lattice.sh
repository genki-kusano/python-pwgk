#!/bin/sh
CONST_IID=100
CONST_SIDE=20
CONST_PCD=2
arr_measure=("square" "square" "square" "gauss")
arr_radius=("014" "017" "020" "010")
temp_pcd=`expr $CONST_PCD - 1`
temp_iid=`expr $CONST_IID - 1`
for j in `seq 0 4`
do
temp_measure=${arr_measure[$j\]}
temp_radius=${arr_radius[$j\]}
date
cd /Users/genki/Desktop/github/data/lattice/pcd${CONST_PCD}_side${CONST_SIDE}_iid${CONST_IID}_${temp_measure}_${temp_radius}/pcd_pd
pwd
for i in `seq 0 $temp_iid`
do
echo "IID: $i"
python3 -m homcloud.pc2diphacomplex -d ${CONST_PCD} -D pcd_$i\.txt pd_$i\.diagram
temp_pcd=`expr $CONST_PCD - 1`
for d in `seq 0 $temp_pcd`
do
python3 -m homcloud.dump_diagram -d $d pd_$i\.diagram -o dim$d\_$i\.txt
done
done
done
