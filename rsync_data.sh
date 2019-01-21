all_src="sweetness toothbrush starbuck"
all_networks="LeNet-5-CIFAR10 AlexNet-CIFAR10 NIN-CIFAR10 CIFAR10-CIFAR10"
for src in $all_src
do
    for network in $all_networks
    do
        rsync \-uvrtP kaveena@$src:~/compare-pruning-signals/$network/results/prune/*.npy ./$network/results/prune/
    done
done
