#!/bin/sh
CONST_PCD=2
CONST_LAMBDA=100
CONST_WIDTH=1
CONST_DISTANCE=005
CONST_IID=100
temp_pcd=`expr $CONST_PCD - 1`
temp_iid=`expr $CONST_IID - 1`
for k in `seq 0 2`
do
date
cd /Users/genki/Desktop/github/data/matern/pcd${CONST_PCD}_lambda${CONST_LAMBDA}_width${CONST_WIDTH}_distance${CONST_DISTANCE}_iid${CONST_IID}_type_$k\/pcd_pd
pwd
for i in `seq 0 $temp_iid`
do
echo "IID: $i"
python3 -m homcloud.pc2diphacomplex -d ${CONST_PCD} -D pcd_$i\.txt pd_$i\.diagram
for d in `seq 0 $temp_pcd`
do
python3 -m homcloud.dump_diagram -d $d pd_$i\.diagram -o dim$d\_$i\.txt
done
done
done
