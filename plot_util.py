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
import matplotlib.colors as mcolors

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def choose_linewidth(method):
    if 'oracle' in method:
        return 2.0
    else:
        return 1.0

def get_color(norm, normalisation):
  plot_colors = dict()
  norm_to_int = {'no_normalisation':0, 'l0_normalisation':1, 'l1_normalisation':2, 'l2_normalisation': 3, 'l0_normalisation_adjusted': 4, 'weights_removed': 5}
  plot_colors['none_norm'] = ['rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown']
  plot_colors['l1_norm'] = ['olive', 'yellowgreen', 'lawngreen', 'lightgreen', 'darkgreen', 'mediumseagreen']
  plot_colors['l2_norm'] = ['teal', 'cyan', 'lightblue', 'steelblue', 'cornflowerblue', 'blue']
  plot_colors['abs_sum_norm'] = ['pink', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'hotpink']
  plot_colors['sqr_sum_norm'] = ['indigo', 'darkorchid', 'darkviolet', 'rebeccapurple', 'purple', 'darkmagenta']
  return mcolors.CSS4_COLORS[plot_colors[norm][norm_to_int[normalisation]]]

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

def convert_label(label):
  labels = label.split('-')
  l = ' f(x) '
  l_prefix = ''
  l_suffix = ''
  if 'none_norm' in labels:
    l_prefix = ' \sum'
    l_suffix = ' '
  elif 'l1_norm' in labels:
    l_prefix = ' \sum | '
    l_suffix = ' | '
  elif 'l2_norm' in labels:
    l_prefix = ' \sum ( '
    l_suffix = ' )^2 '
  elif 'abs_sum_norm' in labels:
    l_prefix = ' | \sum '
    l_suffix = ' | '
  elif 'sqr_sum_norm' in labels:
    l_prefix = ' ( \sum  '
    l_suffix = ' )^2'

  if 'weight_avg' in labels:
    l = ' x '
  elif 'diff_avg' in labels:
    l = ' \\frac{d\mathcal{L}}{dx}'
  elif 'taylor_2nd_approx2' in labels:
    l = ' -x \\frac{d\mathcal{L}}{dx} + \\frac{x^2}{2}\\frac{d^2\mathcal{L}}{dx^2}_{app. 2} (x)'
  elif 'taylor_2nd' in labels:
    l = ' -x \\frac{d\mathcal{L}}{dx} + \\frac{x^2}{2}\\frac{d^2\mathcal{L}}{dx^2}_{app. 1} (x)'
  elif 'taylor' in labels:
    l = ' -x \\frac{d\mathcal{L}}{dx} (x)'
  elif 'hessian_diag_approx2' in labels:
    l = ' \\frac{x^2}{2} \\frac{d^2\mathcal{L}}{dx^2}_{app. 2}(x)'
  elif 'hessian_diag' in labels:
    l = ' \\frac{x^2}{2} \\frac{d^2\mathcal{L}}{dx^2}_{app. 1} (x)'

  S_x = l_prefix + l + l_suffix

  if 'l0_normalisation' in labels:
    l_prefix = ' \\frac{1}{card(x)}' + l_prefix
  elif 'l1_normalisation' in labels:
    l_prefix = ' \\frac{1}{ \| f(x) \| _1}' + l_prefix
  elif 'l2_normalisation' in labels:
    l_prefix = ' \\frac{1}{ \| f(x) \| _2}' + l_prefix
  elif 'weights_removed' in labels:
    l_prefix = ' \\frac{1}{n.weights}' + l_prefix
  elif 'l0_normalisation_adjusted' in labels:
    l_prefix = ' \\frac{1}{card(f(x)_{pruned})}' + l_prefix

  l_prefix = '$' + l_prefix
  l_suffix = l_suffix + '$'

  if 'activation' in labels:
    l_prefix = l_prefix.replace('x', 'a')
    l_suffix = l_suffix.replace('x', 'a')
    l = l.replace('x', 'a')
  elif 'weight' in labels:
    l_prefix = l_prefix.replace('x', 'w')
    l_suffix = l_suffix.replace('x', 'w')
    l = l.replace('x', 'w')
  return l_prefix + l + l_suffix


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

def remove_channel(pruned_channel, convolution_list, channels, list_modules, input=False):
  idx = np.where(channels>pruned_channel)[0][0]
  idx_convolution = convolution_list[idx]
  idx_channel = (pruned_channel - channels[idx-1]) if idx > 0 else pruned_channel
  if input:
    parent = list_modules[idx_convolution].get_only_parent() # Conv layers have one input blob and one output blob for our networks
    while(not(parent.kind=='Convolution' or parent.kind=='Input')):
      if len(parent.parents) == 1:
        parent = parent.get_only_parent()
      elif (parent.kind == 'Concat'): #continue moving up the correct branch
        cummulated_parent_output = 0
        for p in parent.parents:
          cummulated_parent_output += p.output_shape[1] #channel axis
          if idx_channel < cummulated_parent_output:
            break
          parent = p
      else:
        sys.stderr.write("remove_channel parent.kind not implemented\n")
        sys.exit(-1)
    if parent.kind == 'Convolution':
      parent.layer.parameters.num_output -= 1
    elif parent.kind == 'Input':
      parent.output_shape = (parent.output_shape.batch_size, parent.output_shape.channels-1, parent.output_shape.height, parent.output_shape.width)
  else:
    list_modules[idx_convolution].layer.parameters.num_output -= 1

def correct_sparsity(summary, convolution_list, graph, channels, arch, total_channels, stop_itr, input=False): 
  list_modules = graph.node_lut
  graph.compute_output_shapes()
  initial_conv_param, initial_fc_param = compute_num_param(list_modules)
  new_conv_param = np.zeros(total_channels)
  new_fc_param = np.zeros(total_channels)
  for i in range(total_channels):
    remove_channel(summary['pruned_channel'][i], convolution_list, channels, list_modules, input)
    graph.compute_output_shapes()
    update_param_shape(list_modules)
    conv_param, fc_param = compute_num_param(list_modules)
    new_conv_param[i] = conv_param
    new_fc_param[i] = fc_param
    if (stop_itr>0) and (i==stop_itr):
      break 
  return new_conv_param, new_fc_param
