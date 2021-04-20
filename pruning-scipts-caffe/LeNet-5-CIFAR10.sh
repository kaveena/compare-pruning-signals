#! /bin/bash
arch=LeNet-5
dataset=CIFAR10
train_size=256
eval_size=2
test_size=80
test_interval=1
tolerance=10.0
iterations=8
stop_acc=60.0
. ./pruning-scipts-caffe/base_info.sh
. ./pruning-scipts-caffe/base_info_python.sh
