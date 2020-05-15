#! /bin/bash
arch=AlexNet
dataset=IMAGENET32x32
train_size=10
eval_size=2
test_size=400
test_interval=50
tolerance=10.0
iterations=8
stop_acc=30.0
. ./pruning-scipts-caffe/base_info.sh
