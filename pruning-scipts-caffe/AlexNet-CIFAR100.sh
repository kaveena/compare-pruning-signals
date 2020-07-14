#! /bin/bash
arch=AlexNet
dataset=CIFAR100
train_size=10
eval_size=2
test_size=80
test_interval=50
tolerance=10.0
iterations=2
stop_acc=40.0
. ./pruning-scipts-caffe/base_info.sh
