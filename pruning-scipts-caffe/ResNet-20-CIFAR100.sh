#! /bin/bash
arch=ResNet-20
dataset=CIFAR100
train_size=50
eval_size=2
test_size=80
test_interval=1
tolerance=10.0
iterations=2
stop_acc=50.0
. ./pruning-scipts-caffe/base_info.sh
source pruning-scipts-caffe/best_signals.sh

#overall best weights
prune WEIGHT AVERAGE_INPUT L1 l1_normalisation
prune WEIGHT AVERAGE_INPUT L2 weights_removed
prune WEIGHT AVERAGE_INPUT L1 weights_removed
prune WEIGHT AVERAGE_INPUT L1 l2_normalisation
prune WEIGHT AVERAGE_INPUT L2 l1_normalisation
#overall best activations
prune ACTIVATION AVERAGE_INPUT L2 weights_removed
prune ACTIVATION L1 l1_normalisation
#overall best gradients
prune ACTIVATION TAYLOR_2ND_APPROX2 ABS_SUM weights_removed
prune ACTIVATION TAYLOR_2ND_APPROX2 ABS_SUM no_normalisation
prune ACTIVATION TAYLOR ABS_SUM weights_removed
prune ACTIVATION TAYLOR_2ND_APPROX1 ABS_SUM weights_removed
prune ACTIVATION TAYLOR_2ND_APPROX1 ABS_SUM no_normalisation
