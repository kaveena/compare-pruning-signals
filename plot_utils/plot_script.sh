#!/bin/bash

python plot_sensitivities_caffe.py \--arch-caffe LeNet-5-CIFAR10 ;
python plot_sensitivities_caffe.py \--arch-caffe CIFAR10-CIFAR10 ;
python plot_sensitivities_caffe.py \--arch-caffe NIN-CIFAR10 \--test-interval 50 ;
python plot_sensitivities_caffe.py \--arch-caffe AlexNet-CIFAR10 \--test-interval 50;
python plot_sensitivities_caffe.py \--arch-caffe SqueezeNet-CIFAR10 \--test-interval 100;

python plot_sensitivities_caffe.py \--arch-caffe LeNet-5-CIFAR10 \--characterise ;
python plot_sensitivities_caffe.py \--arch-caffe CIFAR10-CIFAR10 \--characterise ;
python plot_sensitivities_caffe.py \--arch-caffe NIN-CIFAR10 \--characterise \--test-interval 50 ;
python plot_sensitivities_caffe.py \--arch-caffe AlexNet-CIFAR10 \--characterise \--test-interval 50;
python plot_sensitivities_caffe.py \--arch-caffe SqueezeNet-CIFAR10 \--characterise \--test-interval 100;

python plot_sensitivities_caffe.py \--arch-caffe LeNet-5-CIFAR10 \--characterise \--metric2 training_iterations \--cut;
python plot_sensitivities_caffe.py \--arch-caffe CIFAR10-CIFAR10 \--characterise \--metric2 training_iterations \--cut;
python plot_sensitivities_caffe.py \--arch-caffe NIN-CIFAR10 \--characterise \--test-interval 50 \--metric2 training_iterations \--cut;
python plot_sensitivities_caffe.py \--arch-caffe AlexNet-CIFAR10 \--characterise \--test-interval 50 \--metric2 training_iterations \--cut;
python plot_sensitivities_caffe.py \--arch-caffe SqueezeNet-CIFAR10 \--characterise \--test-interval 100 \--metric2 training_iterations \--cut;
