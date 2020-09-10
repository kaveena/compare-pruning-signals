#! /bin/bash
arch=ResNet-20
dataset=CIFAR10
train_size=50
eval_size=2
test_size=80
test_interval=1
tolerance=10.0
iterations=8
stop_acc=80.0
. ./pruning-scipts-caffe/base_info.sh
