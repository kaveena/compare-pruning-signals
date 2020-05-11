#! /bin/bash
arch=ResNet-20
dataset=CIFAR100
train_size=256
eval_size=2
test_size=80
test_interval=1
tolerance=10.0
iterations=8
. ./pruning-scipts-caffe/base_info.sh
