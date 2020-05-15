#! /bin/bash
arch=NIN
dataset=CIFAR100
train_size=25
eval_size=2
test_size=80
test_interval=10
tolerance=10.0
iterations=8
stop_acc=40.0
. ./pruning-scipts-caffe/base_info.sh
