import caffe
import sys
import random
import numpy as np
import argparse
from utils.utils import *
from utils.pruning_graph import *
import tempfile
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True

def parser():
    parser = argparse.ArgumentParser(description='Caffe Channel Pruning Example')
    parser.add_argument('--arch', action='store', default='AlexNet',
            help='the network architecture to use')
    parser.add_argument('--dataset', action='store', default='CIFAR10',
            help='the dataset to use')
    parser.add_argument('--eval-size', type=int, default=2, 
            help='Number of batches to use for evaluating the saliency')
    return parser

args = parser().parse_args()

if args.arch is None:
  print("Caffe network needed")
  exit(1)

if args.dataset is None:
  print("Dataset required")
  exit(1)

forward_passes = args.eval_size
backward_passes = args.eval_size

eval_index_filename = './caffe-training-data/' + args.dataset + '/eval-index.txt'

saliency_prototxt_file, saliency_prototxt_filename = tempfile.mkstemp()
masked_prototxt_file, masked_prototxt_filename = tempfile.mkstemp()
solver_prototxt_file, solver_prototxt_filename = tempfile.mkstemp()

arch_solver = 'trained-models/' + args.arch + '-' + args.dataset + '/solver-gpu.prototxt'
pretrained = 'trained-models/' + args.arch + '-' + args.dataset + '/original.caffemodel'
solver_prototxt = get_solver_proto_from_file(arch_solver)

original_prototxt_filename = solver_prototxt.net
solver_prototxt.net = masked_prototxt_filename
masked_prototxt = get_prototxt_from_file(original_prototxt_filename)

caffe.set_mode_gpu()

# net with saliency and mask blobs
net = caffe.Net(original_prototxt_filename, caffe.TEST)
net.copy_from(pretrained)

# global pruning helpers
pruning_net = PruningGraph(net, masked_prototxt)

# global pruning helpers
initial_density = 0
total_param = 0 
total_output_channels = 0

net.reshape()

print('Total number of channels to be considered for pruning: ', pruning_net.total_output_channels)

summary = dict()
summary['initial_conv_param'], summary['initial_fc_param'] = pruning_net.GetNumParam()
summary['initial_test_acc'], summary['initial_test_loss'] = test_network(net, args.eval_size)

# Train accuracy and loss
summary['initial_train_acc'], summary['initial_train_loss'] = test_network(net, 100)

activations = dict()
activations_sqr = dict()
gradients = dict()
gradients_sqr = dict()
var_activations = dict()
mean_activations = dict()
var_gradients = dict()
mean_gradients = dict()
var_activations_channel = dict()
mean_activations_channel = dict()
var_gradients_channel = dict()
mean_gradients_channel = dict()

taylor1_activations = dict()
taylor1_activations_sqr = dict()
mean_taylor1_activations = dict()
var_taylor1_activations = dict()
mean_taylor1_activations_channel = dict()
var_taylor1_activations_channel = dict()

taylor2_activations = dict()
taylor2_activations_sqr = dict()
mean_taylor2_activations = dict()
var_taylor2_activations = dict()
mean_taylor2_activations_channel = dict()
var_taylor2_activations_channel = dict()

for l in pruning_net.convolution_list:
  batch = pruning_net.caffe_net.blobs[l].shape[0]
  channels = pruning_net.caffe_net.blobs[l].shape[1]
  height = pruning_net.caffe_net.blobs[l].shape[2]
  width = pruning_net.caffe_net.blobs[l].shape[3]
  
  activations[l] = np.zeros(pruning_net.caffe_net.blobs[l].shape)
  activations_sqr[l] = np.zeros(pruning_net.caffe_net.blobs[l].shape)
  
  gradients[l] = np.zeros(pruning_net.caffe_net.blobs[l].shape)
  gradients_sqr[l] = np.zeros(pruning_net.caffe_net.blobs[l].shape)
  
  var_activations[l] = np.zeros((channels, height, width))
  mean_activations[l] = np.zeros((channels, height, width))
  
  var_gradients[l] = np.zeros((channels, height, width))
  mean_gradients[l] = np.zeros((channels, height, width))
    
  taylor1_activations[l] = np.zeros((batch, channels, height, width))
  taylor1_activations_sqr[l] = np.zeros((batch, channels, height, width))

  taylor2_activations[l] = np.zeros((channels))
  taylor2_activations_sqr[l] = np.zeros((channels))

for i in range(args.eval_size):
  net.clear_param_diffs()
  net.forward()
  net.backward()
  for l in pruning_net.convolution_list:
    activations[l] += net.blobs[l].data
    activations_sqr[l] += net.blobs[l].data ** 2
    
    gradients[l] += net.blobs[l].diff
    gradients_sqr[l] += net.blobs[l].diff**2

    taylor1_activations[l] += np.abs(net.blobs[l].data * net.blobs[l].diff)
    taylor1_activations_sqr[l] += np.abs(net.blobs[l].data * net.blobs[l].diff)**2

    taylor2_activations[l] += np.abs((net.blobs[l].data * net.blobs[l].diff).sum(axis=(2, 3))).sum(axis=0)
    taylor2_activations_sqr[l] += (np.abs((net.blobs[l].data * net.blobs[l].diff).sum(axis=(2, 3)))**2).sum(axis=0)

for l in pruning_net.convolution_list:
  batch = pruning_net.caffe_net.blobs[l].shape[0]
  channels = pruning_net.caffe_net.blobs[l].shape[1]
  height = pruning_net.caffe_net.blobs[l].shape[2]
  width = pruning_net.caffe_net.blobs[l].shape[3]
  
  mean_activations[l] = activations[l].sum(axis=(0)) / (320 * batch)
  var_activations[l] = (activations_sqr[l].sum(axis=0) / (320 * batch)) - (mean_activations[l]**2)
  
  mean_gradients[l] = gradients[l].sum(axis=(0)) / (320 * batch)
  var_gradients[l] = (gradients_sqr[l].sum(axis=0) / (320 * batch)) - (mean_gradients[l]**2)
  
  mean_taylor1_activations[l] = taylor1_activations[l].sum(axis=(0)) / (320 * batch)
  var_taylor1_activations[l] = (taylor1_activations_sqr[l].sum(axis=0) / (320 * batch)) - (mean_taylor1_activations[l]**2)
  
  mean_taylor2_activations[l] = taylor2_activations[l].sum(axis=(0)) / (320 * batch)
  var_taylor2_activations[l] = (taylor2_activations_sqr[l].sum(axis=0) / (320 * batch)) - (mean_taylor2_activations[l]**2)
  
  mean_activations_channel[l] = mean_activations[l].sum(axis=(1,2)) / (height * width)
  var_activations_channel[l] = var_activations[l].sum(axis=(1,2)) / (height * width)
  
  mean_gradients_channel[l] = mean_gradients[l].sum(axis=(1,2)) / (height * width)
  var_gradients_channel[l] = var_gradients[l].sum(axis=(1,2)) / (height * width)
  
  mean_taylor1_activations_channel[l] = mean_taylor1_activations[l].sum(axis=(1,2)) / (height * width)
  var_taylor1_activations_channel[l] = var_taylor1_activations[l].sum(axis=(1,2)) / (height * width)
  
  mean_taylor2_activations_channel[l] = taylor2_activations[l] / (320 *batch *height * width)
  var_taylor2_activations_channel[l] = ((taylor2_activations_sqr[l]/ (320*batch)) - (taylor2_activations[l] / (320 *batch))**2) / (height * width)

plt.figure()
for l in pruning_net.convolution_list:
  plt.scatter(mean_activations_channel[l], var_activations_channel[l], label=l)
plt.legend()
plt.xlabel("Average mean")
plt.ylabel("Average variance")
plt.title("Average statistics for activations")
plt.savefig("{}-{}-activations-mean-var.pdf".format(args.arch, args.dataset))

plt.figure()
for l in pruning_net.convolution_list:
  plt.scatter(mean_gradients_channel[l], var_gradients_channel[l], label=l)
plt.legend()
plt.xlabel("Average mean")
plt.ylabel("Average variance")
plt.title("Average statistics for gradients")
plt.savefig("{}-{}-gradients-mean-var.pdf".format(args.arch, args.dataset))

plt.figure()
for l in pruning_net.convolution_list:
  plt.scatter(mean_activations_channel[l], mean_gradients_channel[l], label=l)
plt.legend()
plt.xlabel("Average mean activation")
plt.ylabel("Average mean gradient")
plt.title("Average statistics for activations and gradients")
plt.savefig("{}-{}-activations-mean-gradients-mean.pdf".format(args.arch, args.dataset))

plt.figure()
for l in pruning_net.convolution_list:
  plt.scatter(mean_gradients_channel[l], var_activations_channel[l], label=l)
plt.legend()
plt.xlabel("Average mean gradient")
plt.ylabel("Average variance activation")
plt.title("Average statistics for activations and gradients")
plt.savefig("{}-{}-activations-var-gradients-mean.pdf".format(args.arch, args.dataset))

plt.figure()
for l in pruning_net.convolution_list:
  plt.scatter(mean_taylor1_activations_channel[l], var_activations_channel[l], label=l)
plt.legend()
plt.xlabel("Average mean Taylor (L1 reduction)")
plt.ylabel("Average variance activation")
plt.title("Average statistics for activations and saliency")
plt.savefig("{}-{}-activations-var-taylor_l1-mean.pdf".format(args.arch, args.dataset))

plt.figure()
for l in pruning_net.convolution_list:
    plt.scatter(mean_taylor2_activations_channel[l], var_activations_channel[l], label=l)
plt.legend()
plt.xlabel("Average mean Taylor (ABS-SUM reduction)")
plt.ylabel("Average variance activation")
plt.title("Average statistics for activations and saliency")
plt.savefig("{}-{}-activations-var-taylor_abs_sum-mean.pdf".format(args.arch, args.dataset))

caffe.set_mode_cpu()
