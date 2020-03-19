
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

sys.dont_write_bytecode = True

def str2bool(v):
  """
    To parse bool from script argument
  """
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def saliency_scaling(net, input_pruning_signal, output_pruning_signal, scaling , input_saliency_type):
  """
  Apply scaling.
  
  Parameters
  ----------
  net         : caffe._caffe.Net
    CNN considered
  input_pruning_signal: array of floats
    saliency of input channels
  output_pruning_signal: array of floats
    saliency of output channels
  scaling     : type of scaling. options are
                  "l0_norm":  number of points used to compute channel saliency (does not consider updated/pruned network)
                  "l0_norm_adjusted":  number of points used to compute channel saliency (considering updated/pruned network)
                  "l1_norm":  layerwise L1-Norm of saliencies
                  "l2_norm":  layerwise L2-Norm of saliencies
                  "weights_removed":  number of weights to be removed if that channel is pruned
                  "no_scaling":  1.0
  """
  channel_saliencies = np.zeros(net.output_channels_boundary[-1] + net.end_first_layer + 1)
  for conv in net.convolution_list:
    conv_module = net.layer_dict[conv]
    # saliency of channels that are only input channels
    if conv in net.first_layer:
      for c in range(conv_module.input_channels):
        s = 0.0
        weights_removed = 0.0
        l0_norm = 0.0
        l0_norm_adjusted = 0.0
        i_c = conv_module.input_channel_idx[c]
        s = input_pruning_signal[i_c]
        weights_removed += conv_module.active_output_channels.sum() * conv_module.kernel_size
        if input_saliency_type == 'WEIGHT':
          l0_norm += conv_module.output_channels * conv_module.kernel_size
          l0_norm_adjusted += conv_module.active_output_channels.sum() * conv_module.kernel_size
        else:
          l0_norm += conv_module.height * conv_module.width
          l0_norm_adjusted += conv_module.height * conv_module.width
        if (scaling == 'weights_removed') and (weights_removed != 0):
          s /= weights_removed
        elif (scaling == 'l0_normalisation') and (l0_norm != 0):
          s /= l0_norm
        elif (scaling == 'l0_normalisation_adjusted') and (l0_norm_adjusted != 0):
          s /= l0_norm_adjusted
        channel_saliencies[i_c] = s
      L = 1.0
      if (scaling == 'l1_normalisation'):
        L = np.abs(channel_saliencies[conv_module.input_channel_idx[0]:conv_module.input_channel_idx[-1] + 1]).sum()
      elif (scaling == 'l2_normalisation'):
        L = (channel_saliencies[conv_module.input_channel_idx[0]:conv_module.input_channel_idx[-1] + 1] ** 2).sum()
      if (L != 0):
        channel_saliencies[conv_module.input_channel_idx[0]:conv_module.input_channel_idx[-1] + 1] = channel_saliencies[conv_module.input_channel_idx[0]:conv_module.input_channel_idx[-1] + 1] / L
    # saliency for output channels
    for c in range(conv_module.output_channels):
      s = 0.0
      weights_removed = 0.0
      l0_norm = 0.0
      l0_norm_adjusted = 0.0
      o_c = conv_module.output_channel_idx[c]
      s = output_pruning_signal[o_c]
      weights_removed += conv_module.active_input_channels.sum() * conv_module.kernel_size
      if conv_module.bias_term_:
        weights_removed += 1
        if input_saliency_type == 'WEIGHT':
          l0_norm += 1
          l0_norm_adjusted += 1
      if input_saliency_type == 'WEIGHT':
        l0_norm += conv_module.input_channels * conv_module.kernel_size
        l0_norm_adjusted += conv_module.active_input_channels.sum() * conv_module.kernel_size
      else:
        l0_norm += conv_module.height * conv_module.width
      for i_c in conv_module.output_channel_input_idx[c]:
        if i_c != -1:
          idx_channel, i_sink = get_channel_from_global_channel_idx(net, i_c, is_input_channel_idx=True)
          sink = net.layer_dict[i_sink]
          # if any add saliency of input channels
          if sink.type == 'Convolution':
            s += input_pruning_signal[i_c]
            weights_removed += sink.active_output_channels.sum() * sink.kernel_size
            if input_saliency_type == 'WEIGHT':
              l0_norm += sink.output_channels * sink.kernel_size
              l0_norm_adjusted += sink.active_output_channels.sum() * sink.kernel_size
            else:
              l0_norm += sink.height * sink.width
              l0_norm_adjusted += sink.height * sink.width
        # if sink is not a convolution
        elif len(conv_module.sinks) != 0:
          for i_s in conv_module.sinks:
            sink = net.layer_dict[i_s]
            if sink.type == 'InnerProduct':
              weights_removed += sink.active_output_channels.sum() * sink.output_size * sink.input_size
      if (scaling == 'weights_removed') and (weights_removed != 0):
        s /= weights_removed
      elif (scaling == 'l0_normalisation') and (l0_norm != 0):
        s /= l0_norm
      elif (scaling == 'l0_normalisation_adjusted') and (l0_norm_adjusted != 0):
        s /= l0_norm_adjusted
      channel_saliencies[o_c + net.end_first_layer + 1] = s
    L = 1.0
    if (scaling == 'l1_normalisation'):
      L = np.abs(channel_saliencies[conv_module.output_channel_idx[0] + net.end_first_layer + 1:conv_module.output_channel_idx[-1] + 1 + net.end_first_layer + 1]).sum()
    elif (scaling == 'l2_normalisation'):
      L = (channel_saliencies[conv_module.output_channel_idx[0] + net.end_first_layer + 1:conv_module.output_channel_idx[-1] + 1 + net.end_first_layer + 1] ** 2).sum()
    if (L != 0):
      channel_saliencies[conv_module.output_channel_idx[0] + net.end_first_layer + 1:conv_module.output_channel_idx[-1] + 1 + net.end_first_layer + 1] = channel_saliencies[conv_module.output_channel_idx[0] + net.end_first_layer + 1:conv_module.output_channel_idx[-1] + 1 + net.end_first_layer + 1] / L
  return channel_saliencies

def get_channel_from_global_channel_idx(net, global_idx, is_input_channel_idx=False):
  if is_input_channel_idx:
    channels = net.input_channels_boundary
  else:
    channels = net.output_channels_boundary
  idx = np.where(channels>global_idx)[0][0]
  idx_convolution = net.convolution_list[idx]
  idx_channel = (global_idx - channels[idx-1]) if idx > 0 else global_idx
  return idx_channel, idx_convolution

def get_producer_convolution(net, initial_parents, producer_conv, conv_type, conv_index): 
  for i in range(len(initial_parents)): 
    initial_parent = initial_parents[i] 
    if '_split_' in initial_parent: 
      split_names = initial_parent.split('_split_') 
      initial_parent = split_names[0] + '_split' 
      conv_type.append('Split')
    if net.layer_dict[initial_parent].type == 'Concat':
      for j in range(len(net.bottom_names[initial_parent])):
        conv_type.append('Concat')
        conv_index.append(j)
    if ('Convolution' not in net.layer_dict[initial_parent].type) and ('Data' not in net.layer_dict[initial_parent].type): 
      get_producer_convolution(net, net.bottom_names[initial_parent], producer_conv, conv_type,
 conv_index) 
    else: 
      producer_conv.append(initial_parent) 

def get_children(net, layer): 
  children = [] 
  if '_split_' in layer: 
    parent_split = layer.split('_split_')[0] + '_split' 
    next_layers = list(net.layer_dict.keys())[(list(net.layer_dict.keys()).index(parent_split)+
1):] 
  else: 
    next_layers = list(net.layer_dict.keys())[(list(net.layer_dict.keys()).index(layer)+1):] 
  for each in next_layers: 
    if ((layer in net.bottom_names[each]) and (layer not in net.top_names[each])): 
      children.append(each) 
  if ((layer in net.top_names.keys() and net.layer_dict[layer].type == 'Split')): 
    children = list(net.top_names[layer]) 
  return children 

def get_consumer_convolution_or_fc(net, initial_consumers, consumer_sink, junction_type, conv_index, previous_layer): 
  for i in range(len(initial_consumers)): 
    initial_consumer = initial_consumers[i] 
    if '_split_' in initial_consumer: 
      split_names = initial_consumer.split('_split_') 
      parent_split = split_names[0] + '_split' 
      consumer_type = 'Split'
      junction_type.append(consumer_type)
      conv_index.append(split_names[1])
    else: 
      consumer_type = net.layer_dict[initial_consumer].type 
    if consumer_type == 'Concat': 
        junction_type.append('Concat') 
        conv_index.append(net.bottom_names[initial_consumer].index(previous_layer)) 
    if ('Convolution' not in consumer_type) and ('InnerProduct' not in consumer_type): 
      get_consumer_convolution_or_fc(net, get_children(net, initial_consumer), consumer_sink, junction_type, conv_index, initial_consumer) 
    else: 
      consumer_sink.append(initial_consumer) 

def get_sources(net, conv, producer_conv, conv_type, conv_index): 
  get_producer_convolution(net, net.bottom_names[conv], producer_conv, conv_type, conv_index)
 
def get_sinks(net, conv, consumer_conv, conv_type, conv_index): 
  get_consumer_convolution_or_fc(net, get_children(net, conv), consumer_conv, conv_type, conv_index, conv) 

def test(solver, itr):
  accuracy = dict()
  for i in range(itr):
    output = solver.test_nets[0].forward()
    for j in output.keys():
      if j in accuracy.keys():
        accuracy[j] = accuracy[j] + output[j]
      else:
        accuracy[j] = output[j].copy()
  for j in accuracy.keys():
    accuracy[j] /= float(itr)
  return accuracy['top-1']*100.0, accuracy['loss']

def UpdateMask(net, idx_channel, conv_module, fill, final=True, is_input_channel=False, is_convolution=True):
  prune = fill == 0
  if conv_module.type == 'Convolution':
    bias = conv_module.bias_term_
    prune = fill == 0
    if is_input_channel:
      conv_module.active_ifm[idx_channel] = fill
      if conv_module.group == 1:
        if final and prune: 
          conv_module.blobs[0].data[:,idx_channel,:,:].fill(0)
        conv_module.blobs[conv_module.mask_pos_].data[:,idx_channel,:,:].fill(fill)
        conv_module.active_input_channels[idx_channel] = fill
      else:
        can_prune = True
        for g in range(len(conv_module.groups)):
          if (conv_module.active_ifm[g*(conv_module.input_channels) + idx_channel] != 0) and prune:
            can_prune = False
        if can_prune:
          weight_index = idx_channel / conv_module.group
          if final and prune: 
            conv_module.blobs[0].data[:,weight_index,:,:].fill(0)
          conv_module.blobs[conv_module.mask_pos_].data[:,weight_index,:,:].fill(fill)
          conv_module.active_input_channels[weight_index] = fill
    else:
      if final and prune:
        conv_module.blobs[0].data[idx_channel].fill(0)
      if final and prune and bias:
        conv_module.blobs[1].data[idx_channel] = 0
      conv_module.blobs[conv_module.mask_pos_].data[idx_channel].fill(fill)
      if bias:
        conv_module.blobs[conv_module.mask_pos_+1].data[idx_channel] = 0
      conv_module.active_output_channels[idx_channel] = fill
      conv_module.active_ofm[idx_channel] = fill
  elif conv_module.type == 'InnerProduct':
    conv_module.active_input_channels[idx_channel] = 0
    if prune:
      conv_module.blobs[0].data[:, idx_channel].fill(0)
    
def PruneChannel(net, pruned_channel, prune=True, final=True, is_input_channel=False, remove_all_nodes=False):
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
  UpdateMask(net, idx_channel, conv_module, fill, final, is_input_channel)
  # update depend channels
  if is_input_channel: # remove previous conv's output channel
    for i in range(len(conv_module.sources)):
      c = conv_module.sources[i]
      source = net.layer_dict[c]
      if source.type == 'Convolution':
        can_prune = True
        global_source_output_idx = conv_module.input_channel_output_idx[idx_channel][i]
        local_source_output_idx = source.output_channel_idx.index(global_source_output_idx)
        if remove_all_nodes:
          UpdateMask(net, local_source_output_idx, source, fill, final, is_input_channel=False, is_convolution=True)
          for j in range(len(source.output_channel_input_idx[local_source_output_idx])):
            other_sink = net.layer_dict[source.sinks[j]]
            other_local_input_idx = np.where(np.array(other_sink.input_channel_output_idx) == global_source_output_idx)[0][0]
            UpdateMask(net, other_local_input_idx, other_sink, fill, final, is_input_channel=True, is_convolution=True)
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
            UpdateMask(net, local_source_output_idx, source, fill, final, is_input_channel=False, is_convolution=True)
  # if an output channel then update depencies of consumer channels
  else:
    for i in range(len(conv_module.sinks)):
      c = conv_module.sinks[i]
      sink = net.layer_dict[c]
      if sink.type == 'Convolution':
        global_sink_input_idx = conv_module.output_channel_input_idx[idx_channel][i]
        local_sink_input_idx = sink.input_channel_idx.index(global_sink_input_idx)
        # if all sources of this consumer channel have been removed, safely
        # remove this consumer channel
        can_prune = True
        if remove_all_nodes:
          UpdateMask(net, local_sink_input_idx, sink, fill, final, is_input_channel=True, is_convolution=True)
          for j in range(len(sink.input_channel_output_idx[local_sink_input_idx])):
            other_source = net.layer_dict[sink.sources[j]]
            other_local_output_idx = np.where(np.array(other_source.output_channel_input_idx) == global_sink_input_idx)[0][0]
            UpdateMask(net, other_local_output_idx, other_source, fill, final, is_input_channel=False, is_convolution=True)
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
            UpdateMask(net, local_sink_input_idx, sink, fill, final, is_input_channel=True, is_convolution=True)
      elif sink.type == 'InnerProduct':
        UpdateMask(net, idx_channel,sink, fill, final, is_input_channel=True, is_convolution=False)

def channel_output_dependency(idx_channel, idx_convolution, net):
  """
  Find which input channels are going to consume the output channel

  Parameters
  ----------
  idx_channel  : int
    Local index of output channel
  idx_convolution : string
    name of convolution channel
  net         : caffe._caffe.Net
    CNN considered
  """
  conv_module = net.layer_dict[idx_convolution]
  sink_channel = list([])
  for i in range(len(conv_module.sinks)):
    c = conv_module.sinks[i]
    if net.layer_dict[c].type == 'Convolution':
      if len(conv_module.sink_junction_type) == len(conv_module.sinks):
        if conv_module.sink_junction_type[i] == 'Concat':
          c_offset = 0
          for j in range(len(net.layer_dict[c].sources)):
            if j == conv_module.sink_conv_index[i]:
              break
            c_offset += net.layer_dict[net.layer_dict[c].sources[j]].output_channels
          sink_channel.append(net.layer_dict[c].input_channel_idx[c_offset + idx_channel])
        if conv_module.sink_junction_type[i] == 'Split':
          sink_channel.append(net.layer_dict[c].input_channel_idx[idx_channel])
      else:
        sink_channel.append(net.layer_dict[c].input_channel_idx[idx_channel])
    else:
      sink_channel = list([-1])
  if len(conv_module.sinks) == 0:
    sink_channel = list([-1])
  return sink_channel

def channel_input_dependency(idx_channel, idx_convolution, net):
  """
  Find which output channels are going to produce the input channel

  Parameters
  ----------
  idx_channel  : int
    Local index of input channel
  idx_convolution : string
    name of convolution channel
  net         : caffe._caffe.Net
    CNN considered
  """
  conv_module = net.layer_dict[idx_convolution]
  source_channel = list([])
  global_input_idx = conv_module.input_channel_idx[idx_channel]
  if len(conv_module.sources) == 0:
    source_channel = list([-1])
  elif len(conv_module.sources) == len(conv_module.source_junction_type):
    for s in conv_module.sources:
      src = net.layer_dict[s]
      if conv_module.source_junction_type[0] == 'Concat':
        if global_input_idx in np.array(src.output_channel_input_idx).reshape(-1):
          if len(src.output_channel_input_idx[0]) == 1:
            local_output_channel = src.output_channel_input_idx.index([global_input_idx])
            source_channel.append(src.output_channel_idx[local_output_channel])
      if conv_module.source_junction_type[0] == 'Split':
        source_channel.append(src.output_channel_idx[idx_channel])
  else:
    src = net.layer_dict[conv_module.sources[0]]
    if src.type == 'Convolution':
      source_channel.append(src.output_channel_idx[idx_channel])
    else:
      source_channel = list([-1])
  return source_channel

