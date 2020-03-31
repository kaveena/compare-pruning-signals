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
from utils import *

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def convert_label(label):
  labels = label.split('-')
  l = ' f(x) '
  l_prefix = ''
  l_suffix = ''
  if ('none_norm' in labels) or ('N' in labels):
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
  elif 'apoz' in labels:
    l = 'APoZ'
  
  if 'l0_normalisation' in labels:
    l_prefix = ' \\frac{1}{card(x)}' + l_prefix
  elif 'l1_normalisation' in labels:
    l_prefix = ' \\frac{1}{ \| f(x) \| _1}' + l_prefix
  elif 'l2_normalisation' in labels:
    l_prefix = ' \\frac{1}{ \| f(x) \| _2}' + l_prefix
  elif 'weights_removed' in labels:
    l_prefix = ' \\frac{1}{n.weights}' + l_prefix
  elif 'l0_normalisation_adjusted' in labels:
    l_prefix = ' \\frac{1}{card(x_{pruned})}' + l_prefix
  
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
  if 'random' in labels:
    l = 'Random'
    l_prefix = ''
    l_suffix = ''
  return l_prefix + l + l_suffix

def UpdateActiveChannelsOnly(net, idx_channel, conv_module, is_input_channel=False, is_convolution=True):
  fill = 0
  if conv_module.type == 'Convolution':
    bias = conv_module.bias_term_
    if is_input_channel:
      conv_module.active_ifm[int(idx_channel)] = fill
      if conv_module.group == 1:
        conv_module.active_input_channels[int(idx_channel)] = fill
      else:
        can_prune = True
        for g in range(len(conv_module.groups)):
          if (conv_module.active_ifm[int(g*(conv_module.input_channels) + idx_channel)] != 0) and prune:
            can_prune = False
        if can_prune:
          weight_index = idx_channel / conv_module.group
          conv_module.active_input_channels[int(weight_index)] = fill
    else:
      conv_module.active_output_channels[int(idx_channel)] = fill
      conv_module.active_ofm[int(idx_channel)] = fill
  elif conv_module.type == 'InnerProduct':
    conv_module.active_input_channels[int(idx_channel)] = 0

def choose_linewidth(method):
    if 'oracle' in method:
        return 2.0
    else:
        return 1.0

def RemoveChannel(net, pruned_channel, prune=True, final=True, is_input_channel=False, remove_all_nodes=False):
  """
  Prune or unprune a channel from the network.
  
  Parameters
  ----------
  net         : caffe._caffe.Net
    CNN considered
  pruned_channel: int
    global input or output index of the channel to remove
  prune         : bool
    if set to True, removes a channel
    if set to False resets the mask of the channel to ones.  If a channel has
    been pruned with the option final then it cannot b unpruned
  final         : 
    if set to True, prunes a channel by also zeroing its weights
    if set to False, only the mask of the weights are zeroed out
  is_input_channel:
    set to True if pruned_channel is a global input channel index 
  remove_all_nodes:
    set to True to remove dependency branch of that channel
    set to False to remove only pruned_channel and the weights that are inactive
    following the removal of pruned_channel
  """
  channels = net.input_channels_boundary if is_input_channel else net.output_channels_boundary
  fill = 0 if prune else 1
  # find convolution and local channel index
  idx = np.where(channels>pruned_channel)[0][0]
  idx_convolution = net.convolution_list[idx]
  idx_channel = (pruned_channel - channels[idx-1]) if idx > 0 else pruned_channel
  conv_module = net.layer_dict[idx_convolution]
  # remove local weights
  UpdateActiveChannelsOnly(net, idx_channel, conv_module, is_input_channel)
  # update depend channels
  if is_input_channel: # remove previous conv's output channel
    for global_source_output_idx in conv_module.input_channel_output_idx[int(idx_channel)]:
      if global_source_output_idx != -1:
        idx_c, c = get_channel_from_global_channel_idx(net, global_source_output_idx, False)
        source = net.layer_dict[c]
        if source.type == 'Convolution':
          can_prune = True
          local_source_output_idx = source.output_channel_idx.index(global_source_output_idx)
          if remove_all_nodes:
            UpdateActiveChannelsOnly(net, local_source_output_idx, source, is_input_channel=False, is_convolution=True)
            for j in range(len(source.output_channel_input_idx[local_source_output_idx])):
              other_sink = net.layer_dict[source.sinks[j]]
              other_local_input_idx = np.where(np.array(other_sink.input_channel_output_idx) == global_source_output_idx)[0][0]
              UpdateActiveChannelsOnly(net, other_local_input_idx, other_sink, is_input_channel=True, is_convolution=True)
          else:
            for j in range(len(source.output_channel_input_idx[local_source_output_idx])):
              if source.output_channel_input_idx[local_source_output_idx][j] == pruned_channel:
                continue
              else:
                other_sink = net.layer_dict[source.sinks[j]]
                other_local_input_idx = np.where(np.array(other_sink.input_channel_output_idx) == global_source_output_idx)[0][0]
                if other_sink.active_ifm[other_local_input_idx] == 1 :
                  can_prune = False
            if can_prune:
              UpdateActiveChannelsOnly(net, local_source_output_idx, source, is_input_channel=False, is_convolution=True)
  # if an output channel then update depencies of consumer channels
  else:
    for global_sink_input_idx in conv_module.output_channel_input_idx[int(idx_channel)]:
      if global_sink_input_idx != -1 :
        idx_c, c = get_channel_from_global_channel_idx(net, global_sink_input_idx, True)
        sink = net.layer_dict[c]
        if sink.type == 'Convolution':
          local_sink_input_idx = sink.input_channel_idx.index(global_sink_input_idx)
          # if all sources of this consumer channel have been removed, safely
          # remove this consumer channel
          can_prune = True
          if remove_all_nodes:
            UpdateActiveChannelsOnly(net, local_sink_input_idx, sink, is_input_channel=True, is_convolution=True)
            for j in range(len(sink.input_channel_output_idx[local_sink_input_idx])):
              other_source = net.layer_dict[sink.sources[j]]
              other_local_output_idx = np.where(np.array(other_source.output_channel_input_idx) == global_sink_input_idx)[0][0]
              UpdateActiveChannelsOnly(net, other_local_output_idx, other_source, is_input_channel=False, is_convolution=True)
          else:
            for j in range(len(sink.input_channel_output_idx[local_sink_input_idx])):
              if sink.input_channel_output_idx[local_sink_input_idx][j] == pruned_channel:
                continue
              else:
                other_source = net.layer_dict[sink.sources[j]]
                other_local_output_idx = np.where(np.array(other_source.output_channel_input_idx) == global_sink_input_idx)[0][0]
                if other_source.active_ofm[other_local_output_idx] == 1 :
                  can_prune = False
            if can_prune:
              UpdateActiveChannelsOnly(net, local_sink_input_idx, sink, is_input_channel=True, is_convolution=True)
      else:
        for c in conv_module.sinks:
          sink = net.layer_dict[c]
          if sink.type == 'InnerProduct':
            UpdateActiveChannelsOnly(net, idx_channel,sink, is_input_channel=True, is_convolution=False)

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

def correct_sparsity(summary, net, stop_itr, is_input=False, use_old_method=False): 
  if use_old_method and is_input:
    total_channels = net.input_channels_boundary[-1]
  elif use_old_method:
    total_channels = net.output_channels_boundary[-1]
  else:
    total_channels = net.layer_dict[net.last_layer[-1]].output_channel_idx[-1] + 1 + net.end_first_layer + 1 
  new_conv_param = np.zeros(total_channels)
  new_fc_param = np.zeros(total_channels)
  for i in range(total_channels):
    if use_old_method:
      RemoveChannel(net, summary['pruned_channel'][i], is_input_channel=is_input)
    else:
      channel_to_remove = summary['pruned_channel'][i] 
      if channel_to_remove < net.end_first_layer + 1:
        RemoveChannel(net, summary['pruned_channel'][i], is_input_channel=True)
      else:
        RemoveChannel(net, summary['pruned_channel'][i] - (net.end_first_layer + 1), is_input_channel=False)
    conv_param, fc_param = compute_num_param(net)
    new_conv_param[i] = conv_param
    new_fc_param[i] = fc_param
    if (stop_itr>0) and (i==stop_itr):
      break
  return new_conv_param, new_fc_param
