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

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def choose_linewidth(method):
    if 'oracle' in method:
        return 2.0
    else:
        return 1.0

def choose_color(method):
    if 'fisher' in method:
        return colors[1]    #orange
    elif 'taylor-abbs-norm' in method:
        return colors[0]    #blue
    elif 'min-weight' in method:
        return colors[2]    #green
    elif 'mean-act' in method:
        return colors[3]    #red
    elif 'apoz' in method:
        return colors[4]    #purple
    elif 'l1-norm-weight' in method:
        return colors[5]    #brown
    elif 'random' in method:
        return colors[6]    #pink
    elif 'rms-grad-weight' in method:
        return colors[8]    #
    elif 'hybrid' in method:
        return colors[9]    #cyan
    elif 'oracle' in method:
        return '#ffffff'    #black
    else:
        return colors[7]    #grey

def choose_label(method):
    if '_retrained' in method:
        method = method.split('_retrained')[0]
    if 'augmented' in method:
        label = method.split('augmented_')[1].split('_top')[0]
        if 'hybrid' not in label:
            label = label + ' + oracle'
    else:
        label = method
    return label

def choose_network(arch):
    if arch=='vgg19':
        return 'VGG-19'
    if 'resnet' in arch:
        return 'ResNet-' + arch.split('resnet')[1]
    if arch=='LeNet_5':
        return 'LeNet-5'
    return arch

def choose_linestyle(method):
    if 'augmented' in method:
        return '-'
    else:
        return '--'

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
  idx_channel = (pruned_channel - channels[idx-1]) if idx > 0 else pruned_channel
  list_modules[idx_convolution].layer.parameters.num_output -= 1

def correct_sparsity(summary, convolution_list, graph, channels, arch, total_channels, stop_itr): 
  list_modules = graph.node_lut
  graph.compute_output_shapes()
  initial_conv_param, initial_fc_param = compute_num_param(list_modules)
  new_num_param = np.zeros(total_channels)
  for i in range(total_channels):
    remove_channel(summary['pruned_channel'][i], convolution_list, channels, list_modules)
    graph.compute_output_shapes()
    update_param_shape(list_modules)
    conv_param, fc_param = compute_num_param(list_modules)
    new_num_param[i] = conv_param
    if (stop_itr>0) and (i==stop_itr):
      break 
  return new_num_param  
