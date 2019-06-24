import caffe
import os
import struct
import sys
import random
import numpy as np
import caffe
import argparse
import random
import time
from utils import *


sys.dont_write_bytecode = True

_caffe_saliencies_ = caffe._caffe.SALIENCY.names
_caffe_saliency_input_ = caffe._caffe.SALIENCY_INPUT.names 
_caffe_saliency_norm_ = caffe._caffe.SALIENCY_NORM.names

def parser():
    parser = argparse.ArgumentParser(description='Caffe Channel Pruning Example')
    parser.add_argument('--arch', action='store', default='caffe-pruned-models/LeNet-5-CIFAR10/solver-gpu.prototxt',
            help='the caffe solver to use')
    parser.add_argument('--arch-saliency', action='store', default='caffe-pruned-models/LeNet-5-CIFAR10/128-images.prototxt',
            help='saliency prototxt to use')
    parser.add_argument('--pretrained', action='store', default='caffe-pruned-models/LeNet-5-CIFAR10/original.caffemodel',
            help='pretrained caffemodel')
    parser.add_argument('--retrain', type=str2bool, nargs='?', default=False,
            help='retrain the pruned network')
    parser.add_argument('--characterise', type=str2bool, nargs='?', default=False,
            help='characterise the pruned network')
    parser.add_argument('--method', action='store', default='WEIGHT_AVG',
            help='saliency method')
    parser.add_argument('--saliency-norm', action='store', default='NONE',
            help='Caffe saliency_norm')
    parser.add_argument('--saliency-input', action='store', default='WEIGHT',
            help='Caffe saliency_input')
    parser.add_argument('--normalisation', action='store', default='l1_normalisation',
            help='Layer-wise normalisation to use for saliency')
    parser.add_argument('--test-size', type=int, default=1, 
            help='Number of batches to use for testing')
    parser.add_argument('--eval-size', type=int, default=1, 
            help='Number of batches to use for evaluating the saliency')
    parser.add_argument('--input-channels', type=str2bool, nargs='?', default=True,
            help='consider saliency as input channels')
    parser.add_argument('--output-channels', type=str2bool, nargs='?', default=True,
            help='consider saliency as output channels')
    return parser

start = time.time()
args = parser().parse_args()
if args.arch is None:
  print("Caffe solver needed")
  exit(1)
#  Allocate two models : one for masking weights and retraining (weights + mask)
#                          one for computing saliency (weights + masks + saliency)
caffe.set_mode_gpu()
# net with saliency and mask blobs
net = caffe.Net(args.arch_saliency, caffe.TEST)
# net with only mask blobs
saliency_solver = caffe.SGDSolver(args.arch)
saliency_solver.net.copy_from(args.pretrained)
# share the masks blobs between all networks
saliency_solver.test_nets[0].share_with(saliency_solver.net)
net.share_with(saliency_solver.net) 

net.convolution_list = list(filter(lambda x: 'Convolution' in net.layer_dict[x].type, net.layer_dict.keys()))
named_modules = net.layer_dict

in_offset = 0
out_offset = 0
for layer in net.convolution_list:
  conv_module = named_modules[layer]
  # reset mask
  conv_module.blobs[conv_module.mask_pos_].data.fill(1)
  if conv_module.bias_term_:
    conv_module.blobs[conv_module.mask_pos_+1].data.fill(1)
  # create helper attributes
  conv_module.batch = net.blobs[layer].num
  conv_module.height = net.blobs[layer].height
  conv_module.width = net.blobs[layer].width
  conv_module.output_channels = conv_module.blobs[0].data.shape[0]
  conv_module.input_channels = conv_module.blobs[0].data.shape[1]
  conv_module.kernel_size = conv_module.blobs[0].data.shape[2] * conv_module.blobs[0].data.shape[3]
  conv_module.sources = []
  conv_module.sinks = []
  conv_module.source_junction_type = []
  conv_module.sink_junction_type = []
  conv_module.source_conv_index = []
  conv_module.sink_conv_index = []
  # populate sources, sinks, ...
  get_sources(net, layer, conv_module.sources, conv_module.source_junction_type, conv_module.source_conv_index)
  get_sinks(net, layer, conv_module.sinks, conv_module.sink_junction_type, conv_module.sink_conv_index)
  input_layer = net.bottom_names[layer][0] 
  # create more helper attributes
  conv_module.group = int(net.blobs[input_layer].channels / conv_module.blobs[0].channels) 
  conv_module.active_input_channels = np.ones(conv_module.input_channels)
  conv_module.active_ifm = np.ones(conv_module.input_channels * conv_module.group)
  conv_module.active_output_channels = np.ones(conv_module.output_channels)
  conv_module.active_ofm = np.ones(conv_module.output_channels)
  conv_module.input_channel_idx = list(range(in_offset, in_offset + conv_module.input_channels))
  conv_module.output_channel_idx = list(range(out_offset, out_offset + conv_module.output_channels))
  for l in conv_module.sinks:
    if 'InnerProduct' in net.layer_dict[l].type:
      net.layer_dict[l].input_channels = conv_module.output_channels
      net.layer_dict[l].input_size = net.layer_dict[l].blobs[0].data.shape[1] / conv_module.output_channels
      net.layer_dict[l].output_size = net.layer_dict[l].blobs[0].data.shape[0]
      net.layer_dict[l].active_input_channels = np.ones(conv_module.output_channels)
  in_offset += conv_module.input_channels
  out_offset += conv_module.output_channels

# choose saliency measure
if args.method in _caffe_saliencies_.keys():
  for layer in net.convolution_list:
    conv_module = named_modules[layer]
    conv_module.saliency_ = _caffe_saliencies_[args.method] 
    conv_module.saliency_norm_ = _caffe_saliency_norm_[args.saliency_norm]
    conv_module.saliency_input_ = _caffe_saliency_input_[args.saliency_input]

# global pruning helpers
method = args.method
initial_density = 0
total_param = 0 
total_output_channels = 0
total_input_channels = 0
output_channels = []
input_channels = []
in_offset = 0
out_offset = 0
net.first_layer = []
net.last_layer = []
for layer in net.convolution_list:
  conv_module = named_modules[layer]
  total_output_channels += conv_module.output_channels
  total_input_channels += conv_module.input_channels
  output_channels.append(conv_module.output_channels)
  input_channels.append(conv_module.input_channels)
  initial_density += named_modules[layer].blobs[0].num * named_modules[layer].blobs[0].height * named_modules[layer].blobs[0].width * named_modules[layer].blobs[0].channels
  if 'data' in conv_module.sources:
    net.first_layer.append(layer)
  for l in conv_module.sinks:
    if 'InnerProduct' in net.layer_dict[l].type:
      net.last_layer.append(layer)
      break
  if len(conv_module.sinks) == 0:
    net.last_layer.append(layer)
  conv_module.output_channel_input_idx = np.zeros(conv_module.output_channels).reshape(-1, 1).tolist()
  for i in range(conv_module.output_channels):
    conv_module.output_channel_input_idx[i] = channel_output_dependency(i, layer, net)
for layer in net.convolution_list:
  conv_module = named_modules[layer]
  conv_module.input_channel_output_idx = np.zeros(conv_module.input_channels).reshape(-1, 1).tolist()
  for i in range(conv_module.input_channels):
    conv_module.input_channel_output_idx[i] = channel_input_dependency(i, layer, net)
output_channels = np.array(output_channels)
net.output_channels_boundary = np.cumsum(output_channels)
input_channels = np.array(input_channels)
net.input_channels_boundary = np.cumsum(input_channels)

total_channels = named_modules[net.last_layer[-1]].output_channel_idx[-1] + 1
for l in net.first_layer:
  total_channels += named_modules[l].input_channels
net.end_first_layer = net.layer_dict[l].input_channel_idx[-1]
l = net.last_layer[0]
net.start_last_layer = net.layer_dict[l].output_channel_idx[0] + net.end_first_layer + 1

net.reshape()

print('Total number of channels to be considered for pruning: ', total_channels)

active_channel = list(range(total_channels))
# remove channels that would not have a saliency in some cases
if not args.output_channels:
  for i in range(net.start_last_layer, total_channels):
    active_channel.remove(i)
elif not args.input_channels:
  for i in range(net.end_first_layer + 1):
    active_channel.remove(i)
test_acc, ce_loss = test(saliency_solver, args.test_size)
initial_eval_loss = 0.0
initial_eval_acc = 0.0
for i in range(100):
  output = net.forward()
  initial_eval_loss += output['loss']
  initial_eval_acc += output['top-1']
initial_eval_loss /= 100.0
initial_eval_acc /= 100.0
print('Initial eval loss', initial_eval_loss)
print('Initial eval acc', initial_eval_acc)

# Train accuracy and loss
initial_train_acc = 0.0
initial_train_loss = 0.0
if args.characterise:
  current_loss = saliency_solver.net.forward()['loss']
  for i in range(100):
    output = saliency_solver.net.forward()
    initial_train_loss += output['loss']
    initial_train_acc += output['top-1']
  initial_train_loss /= 100.0
  initial_train_acc /= 100.0
  print('Initial train loss', initial_train_loss)
  print('Initial train acc', initial_train_acc)

if method in _caffe_saliencies_:
  for layer in net.convolution_list:
    named_modules[layer].blobs[named_modules[layer].saliency_pos_].data.fill(0) # reset saliency
    named_modules[layer].blobs[named_modules[layer].saliency_pos_+1].data.fill(0) # reset saliency
    if args.input_channels:
      named_modules[layer].saliency_input_channel_compute_ = True
    if args.output_channels:
      named_modules[layer].saliency_output_channel_compute_ = True
pruning_signal = np.array([])
input_pruning_signal = np.array([])
output_pruning_signal = np.array([])

# compute saliency    
evalset_size = args.eval_size;
current_eval_loss = 0.0
current_eval_acc = 0.0
for iter in range(evalset_size):
  if method == 'random':
    break
  net.clear_param_diffs()
  output = net.forward()
  net.backward()
  current_eval_loss += output['loss']
  current_eval_acc += output['top-1']
  if (method == 'WEIGHT_AVG') and (args.saliency_input == 'WEIGHT'):
    break   #no need to do multiple passes of the network

if method in _caffe_saliencies_:
  for layer in net.convolution_list:
    input_saliency_data = named_modules[layer].blobs[named_modules[layer].saliency_pos_+1].data[0]
    output_saliency_data = named_modules[layer].blobs[named_modules[layer].saliency_pos_].data[0]
    input_pruning_signal = np.hstack([input_pruning_signal, input_saliency_data])
    output_pruning_signal = np.hstack([output_pruning_signal, output_saliency_data])
  pruning_signal = layerwise_normalisation(net, input_pruning_signal, output_pruning_signal, args.normalisation, args.saliency_input)

if (method != 'WEIGHT_AVG') or (args.saliency_input != 'WEIGHT'):
  pruning_signal /= float(evalset_size) # get approximate change in loss using taylor expansions

if method == 'random':
  pruning_signal = np.zeros(total_channels)
  pruning_signal[random.sample(active_channel, 1)] = -1

prune_channel_idx = np.argmin(pruning_signal[active_channel])
prune_channel = active_channel[prune_channel_idx]
#if prune_channel < net.end_first_layer + 1 :
#  PruneChannel(net, prune_channel, final=True, is_input_channel=True)
#else:
#  PruneChannel(net, prune_channel - (net.end_first_layer + 1), final=True, is_input_channel=False)

caffe.set_mode_cpu()

