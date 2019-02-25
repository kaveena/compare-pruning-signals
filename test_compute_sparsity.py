import scipy
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from plot_util import *
from triNNity.frontend.graph import IRGraphBuilder
from triNNity.util.transformers import DataInjector
import caffe
import triNNity
from functools import reduce

def update_param_shape(list_modules):
  for l in list_modules.keys(): 
    if list_modules[l].kind == 'Convolution': 
      #check m 
      list_modules[l].params_shape[0][0] = list_modules[l].output_shape.channels 
      if len(list_modules[l].params_shape) > 1 : 
        list_modules[l].params_shape[1][0] = list_modules[l].output_shape.channels 
      #check c 
      list_modules[l].params_shape[0][1] = list_modules[l].get_only_parent().output_shape.channels 
    if list_modules[l].kind == 'InnerProduct': 
      list_modules[l].params_shape[0][1] = list_modules[l].get_only_parent().output_shape.channels * list_modules[l].get_only_parent().output_shape.height * list_modules[l].get_only_parent().output_shape.width 
      if len(list_modules[l].params_shape) > 1 : 
        list_modules[l].params_shape[1][0] = list_modules[l].params_shape[0][1] 

def compute_num_param(list_modules):
  update_param_shape(list_modules)
  conv_param = 0 
  fc_param = 0
  for l in list_modules.keys():
    if list_modules[l].kind == 'Convolution':
      conv_param += reduce(lambda u, v: u + v, list(map(lambda x: reduce(lambda a, b: a * b, x), list_modules[l].params_shape)))
    if list_modules[l].kind == 'InnerProduct':
      fc_param += reduce(lambda u, v: u + v, list(map(lambda x: reduce(lambda a, b: a * b, x), list_modules[l].params_shape)))
  return conv_param, fc_param

def remove_channel(pruned_channel, convolution_list, channels, list_modules):
  idx = np.where(channels>pruned_channel)[0][0]
  idx_convolution = convolution_list[idx]
  idx_channel = (pruned_channel - channels[idx-1]) if idx > else pruned_channel
  list_modules[idx_convolution].layer.parameters.num_output -= 1


to_torch_arch = {'LeNet-5-CIFAR10': 'LeNet_5', 'AlexNet-CIFAR10': 'AlexNet', 'NIN-CIFAR10': 'NIN'}

parser = argparse.ArgumentParser()
parser.add_argument('--arch-caffe', action='store', default='LeNet-5-CIFAR10')
parser.add_argument('--retrain', action='store_true', default=False)

args = parser.parse_args()  
args.arch = to_torch_arch[args.arch_caffe]

prototxt = 'caffe-pruned-models/'+ args.arch_caffe + '/test.prototxt'  
caffemodel = 'caffe-pruned-models/'+ args.arch_caffe + '/original.caffemodel'
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

graph = IRGraphBuilder(prototxt, 'test').build()
graph = DataInjector(prototxt, caffemodel)(graph)

graph.compute_output_shapes()
list_modules = graph.node_lut

for l in list_modules.keys():
  if list_modules[l].data is not None:
    list_modules[l].params_shape = list(map(lambda x: list(x.shape), list_modules[l].data))

convolution_list = list(filter(lambda x: 'Convolution' in net.layer_dict[x].type, net.layer_dict.keys()))

channels = []
total_channels = 0
for layer in convolution_list:
  total_channels += list_modules[layer].layer.parameters.num_output
  channels.append(list_modules[layer].layer.parameters.num_output)
channels = np.array(channels)
channels = np.cumsum(channels)

initial_num_param = 0
initial_conv_param = 0
initial_fc_param = 0

initial_conv_param, initial_fc_param = compute_num_param(list_modules) 

initial_num_param = initial_conv_param + initial_fc_param






