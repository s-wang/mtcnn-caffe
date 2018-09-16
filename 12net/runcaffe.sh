#!/usr/bin/env sh
export PYTHONPATH=$PYTHONPATH:/home/swang/work_space/mtcnn-caffe-python-CongWeilin/12net


set -e
/home/swang/work_space/caffe-c11/build/tools/caffe train \
	 --solver=./solver.prototxt  2>&1 | tee log/12net.log
#	 --weights=./models_cls/solver_iter_250000.caffemodel \
#	 2>&1 | tee log/12net.log

