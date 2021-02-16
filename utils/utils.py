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
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

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

def get_solver_proto_from_file(filename):
  solver_proto = caffe_pb2.SolverParameter()
  with open(filename) as f:
      s = f.read()
      txtf.Merge(s, solver_proto)
  return solver_proto

def get_prototxt_from_file(filename):
  prototxt_net = caffe_pb2.NetParameter()
  with open(filename) as f:
      s = f.read()
      txtf.Merge(s, prototxt_net)
  return prototxt_net

def save_prototxt_to_file(prototxt_net, filename):
  with open(filename, 'w') as f:
      f.write(str(prototxt_net))

def add_mask_to_prototxt(original_prototxt):
  for l in original_prototxt.layer:
    if l.type == 'Convolution':
      l.convolution_mask_param.Clear()
      l.convolution_mask_param.mask_term = True
      l.convolution_mask_param.mask_filler.type = 'constant'
      l.convolution_mask_param.mask_filler.value = 1
      expected_len_param = 1
      final_len_param = 2
      if l.convolution_param.bias_term:
        expected_len_param += 1
        final_len_param += 2
      current_len_param = len(l.param)
      for i in range(current_len_param, expected_len_param):
        l.param.add()
        l.param[i].lr_mult = 1
        l.param[i].decay_mult = 1
      current_len_param = len(l.param)
      for i in range(current_len_param, final_len_param):
        l.param.add()
        l.param[i].lr_mult = 0
        l.param[i].decay_mult = 0

def add_saliency_to_prototxt(original_prototxt, pointwise_saliency, saliency_input, saliency_reduction, output_channel=True, input_channel=False):
  if not (len(pointwise_saliency) == len(saliency_input) == len(saliency_reduction)):
    os.exit("Provided Saliencies Do Not Match")
  enum_saliency = caffe_pb2.ConvolutionSaliencyParameter.SALIENCY
  enum_saliency_input = caffe_pb2.ConvolutionSaliencyParameter.INPUT
  enum_saliency_norm = caffe_pb2.ConvolutionSaliencyParameter.NORM
  for l in original_prototxt.layer:
    if l.type == 'Convolution':
      l.convolution_saliency_param.Clear()
      l.convolution_saliency_param.saliency_term = True
      l.convolution_saliency_param.output_channel_compute = output_channel
      l.convolution_saliency_param.input_channel_compute = input_channel
      l.convolution_saliency_param.accum = True
      for i in range(len(pointwise_saliency)):
        l.convolution_saliency_param.saliency.append(enum_saliency.Value(pointwise_saliency[i]))
        l.convolution_saliency_param.saliency_input.append(enum_saliency_input.Value(saliency_input[i]))
        l.convolution_saliency_param.saliency_norm.append(enum_saliency_norm.Value(saliency_reduction[i]))

def saliency_scaling(pruning_net, output_saliency=True, input_saliency=False, input_saliency_type='WEIGHT', scaling='L0'):
  if input_saliency and not(output_saliency):
    final_saliency = np.zeros(pruning_net.total_input_channels)
    if scaling == 'no_normalisation':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        final_saliency[graph_layer.input_channel_idx] = graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data
    elif scaling == 'l1_normalisation':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        layer_scale_factor = np.abs(graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data).sum()
        if layer_scale_factor <= 0.0:
          layer_scale_factor = 1.0
        final_saliency[graph_layer.input_channel_idx] = graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data / layer_scale_factor
    elif scaling == 'l2_normalisation':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        layer_scale_factor = (graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data**2).sum()
        if layer_scale_factor <= 0.0:
          layer_scale_factor = 1.0
        else:
          layer_scale_factor = layer_scale_factor**0.5
        final_saliency[graph_layer.input_channel_idx] = graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data / layer_scale_factor
    elif scaling == 'l0_normalisation_adjusted':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        if input_saliency_type == 'ACTIVATION':
          bottom_layer = pruning_net.caffe_net.blobs[pruning_net.caffe_net.bottom_names[l][0]]
          layer_scale_factor = bottom_layer.height * bottom_layer.width
        elif input_saliency_type == 'WEIGHT':
          layer_scale_factor = graph_layer.active_output_channels.sum() * graph_layer.kernel_size
        else:
          sys.exit("Input saliency type not valid")
        final_saliency[graph_layer.input_channel_idx] = graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data / layer_scale_factor
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        layer_scale_factor = np.ones(graph_layer.input_channels)
        for i in range(graph_layer.input_channels):
          w = 0
          global_channel_idx = graph_layer.input_channel_idx[i]
          # Get number of local parameters
          w += graph_layer.active_output_channels.sum() * graph_layer.kernel_size
          # Get other linked sinks and sources
          sinks, sources = pruning_net.GetAllSinksSources(global_channel_idx, True)
          # Get their parameters
          for i_s in sinks:
            idx_c_sink, idx_conv_sink = pruning_net.GetChannelFromGlobalChannelIdx(i_s, True)
            sink_layer = pruning_net.graph[idx_conv_sink]
            if sink_layer.type == 'InnerProduct':
              w += sink_layer.active_output_channels.sum() * sink_layer.output_size * sink_layer.input_size
            elif sink_layer.type == 'Convolution':
              w += sink_layer.active_output_channels.sum() * sink_layer.kernel_size
          for i_s in sources:
            idx_c_source, idx_conv_source = pruning_net.GetChannelFromGlobalChannelIdx(i_s, False)
            source_layer = pruning_net.graph[idx_conv_source]
            if source_layer.type == 'Convolution':
              w += source_layer.active_input_channels.sum() * source_layer.kernel_size
          # scale saliency
          if w > 0 :
            layer_scale_factor[i] = w
          final_saliency[graph_layer.input_channel_idx] = graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data / layer_scale_factor
  elif output_saliency and not(input_saliency):
    final_saliency = np.zeros(pruning_net.total_output_channels)
    if scaling == 'no_normalisation':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        final_saliency[graph_layer.output_channel_idx] = graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data
    elif scaling == 'l1_normalisation':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        layer_scale_factor = np.abs(graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data).sum()
        if layer_scale_factor <= 0.0:
          layer_scale_factor = 1.0
        final_saliency[graph_layer.output_channel_idx] = graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data / layer_scale_factor
    elif scaling == 'l2_normalisation':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        layer_scale_factor = (graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data**2).sum()
        if layer_scale_factor <= 0.0:
          layer_scale_factor = 1.0
        else:
          layer_scale_factor = layer_scale_factor**0.5
        final_saliency[graph_layer.output_channel_idx] = graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data / layer_scale_factor
    elif scaling == 'l0_normalisation':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        if input_saliency_type == 'ACTIVATION':
          layer_scale_factor = graph_layer.height * graph_layer.width
        elif input_saliency_type == 'WEIGHT':
          layer_scale_factor = graph_layer.input_channels * graph_layer.kernel_size
        else:
          sys.exit("Input saliency type not valid")
        final_saliency[graph_layer.output_channel_idx] = graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data / layer_scale_factor
    elif scaling == 'l0_normalisation_adjusted':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        if input_saliency_type == 'ACTIVATION':
          layer_scale_factor = graph_layer.height * graph_layer.width
        elif input_saliency_type == 'WEIGHT':
          layer_scale_factor = graph_layer.active_input_channels.sum() * graph_layer.kernel_size
        else:
          sys.exit("Input saliency type not valid")
        final_saliency[graph_layer.output_channel_idx] = graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data / layer_scale_factor
    elif scaling == 'weights_removed':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        layer_scale_factor = np.ones(graph_layer.output_channels)
        for i in range(graph_layer.output_channels):
          w = 0
          global_channel_idx = graph_layer.output_channel_idx[i]
          # Get number of local parameters
          w += graph_layer.active_input_channels.sum() * graph_layer.kernel_size
          # Get other linked sinks and sources
          sinks, sources = pruning_net.GetAllSinksSources(global_channel_idx, False)
          # Get their parameters
          for i_s in sinks:
            idx_c_sink, idx_conv_sink = pruning_net.GetChannelFromGlobalChannelIdx(i_s, True)
            sink_layer = pruning_net.graph[idx_conv_sink]
            if sink_layer.type == 'InnerProduct':
              w += sink_layer.active_output_channels.sum() * sink_layer.output_size * sink_layer.input_size
            elif sink_layer.type == 'Convolution':
              w += sink_layer.active_output_channels.sum() * sink_layer.kernel_size
          for i_s in sources:
            idx_c_source, idx_conv_source = pruning_net.GetChannelFromGlobalChannelIdx(i_s, False)
            source_layer = pruning_net.graph[idx_conv_source]
            if source_layer.type == 'Convolution':
              w += source_layer.active_input_channels.sum() * source_layer.kernel_size
          # scale saliency
          if w > 0 :
            layer_scale_factor[i] = w
          final_saliency[graph_layer.output_channel_idx] = graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data / layer_scale_factor
    elif scaling == 'l0_and_weights_removed':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        layer_scale_factor = np.ones(graph_layer.output_channels)
        for i in range(graph_layer.output_channels):
          w = 0
          global_channel_idx = graph_layer.output_channel_idx[i]
          # Get number of local parameters
          w += graph_layer.active_input_channels.sum() * graph_layer.kernel_size
          # Get other linked sinks and sources
          sinks, sources = pruning_net.GetAllSinksSources(global_channel_idx, False)
          # Get their parameters
          for i_s in sinks:
            idx_c_sink, idx_conv_sink = pruning_net.GetChannelFromGlobalChannelIdx(i_s, True)
            sink_layer = pruning_net.graph[idx_conv_sink]
            if sink_layer.type == 'InnerProduct':
              w += sink_layer.active_output_channels.sum() * sink_layer.output_size * sink_layer.input_size
            elif sink_layer.type == 'Convolution':
              w += sink_layer.active_output_channels.sum() * sink_layer.kernel_size
          for i_s in sources:
            idx_c_source, idx_conv_source = pruning_net.GetChannelFromGlobalChannelIdx(i_s, False)
            source_layer = pruning_net.graph[idx_conv_source]
            if source_layer.type == 'Convolution':
              w += source_layer.active_input_channels.sum() * source_layer.kernel_size
          # scale saliency
          if w > 0 :
            if input_saliency_type == 'ACTIVATION':
              layer_scale_factor[i] = w * graph_layer.height * graph_layer.width
            elif input_saliency_type == 'WEIGHT':
              layer_scale_factor[i] = w* graph_layer.active_input_channels.sum() * graph_layer.kernel_size
            else:
              sys.exit("Input saliency type not valid")
          final_saliency[graph_layer.output_channel_idx] = graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data / layer_scale_factor
  elif output_saliency and input_saliency:
    final_saliency = np.zeros(pruning_net.total_output_channels)
    for l in pruning_net.convolution_list:
      graph_layer = pruning_net.graph[l]
      layer_saliency = np.ones(graph_layer.output_channels)
      scale_w = np.ones(graph_layer.output_channels)
      scale_a = np.ones(graph_layer.output_channels)
      for i in range(graph_layer.output_channels):
        s = 0
        w = 0
        a = 0
        global_channel_idx = graph_layer.output_channel_idx[i]
        # Get number of local parameters
        s += graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data[0][i]
        w += graph_layer.active_input_channels.sum() * graph_layer.kernel_size
        a += graph_layer.height * graph_layer.width
        # Get other linked sinks and sources
        sinks, sources = pruning_net.GetAllSinksSources(global_channel_idx, False)
        # Get their parameters
        for i_s in sinks:
          idx_c_sink, idx_conv_sink = pruning_net.GetChannelFromGlobalChannelIdx(i_s, True)
          sink_layer = pruning_net.graph[idx_conv_sink]
          if sink_layer.type == 'Convolution':
            s += sink_layer.caffe_layer.blobs[graph_layer.saliency_pos+1].data[0][idx_c_sink]
            w += sink_layer.active_output_channels.sum() * sink_layer.kernel_size
          elif sink_layer.type == 'InnerProduct':
            w += sink_layer.active_output_channels.sum() * sink_layer.output_size * sink_layer.input_size
        for i_s in sources:
          idx_c_source, idx_conv_source = pruning_net.GetChannelFromGlobalChannelIdx(i_s, False)
          source_layer = pruning_net.graph[idx_conv_source]
          if source_layer.type == 'Convolution':
            s += source_layer.caffe_layer.blobs[graph_layer.saliency_pos].data[0][idx_c_source]
            w += source_layer.active_input_channels.sum() * source_layer.kernel_size
        layer_saliency[i] = s
        scale_w[i] = w
      if scaling == 'no_normalisation':
        final_saliency[graph_layer.output_channel_idx] = layer_saliency
      elif scaling == 'l1_normalisation':
        layer_scale_factor = np.abs(graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data).sum()
        if layer_scale_factor <= 0.0:
          layer_scale_factor = 1.0
        final_saliency[graph_layer.output_channel_idx] = layer_saliency / layer_scale_factor
      elif scaling == 'l2_normalisation':
        layer_scale_factor = (graph_layer.caffe_layer.blobs[graph_layer.saliency_pos].data**2).sum()
        if layer_scale_factor <= 0.0:
          layer_scale_factor = 1.0
        else:
          layer_scale_factor = layer_scale_factor**0.5
        final_saliency[graph_layer.output_channel_idx] = layer_saliency / layer_scale_factor
      elif scaling == 'l0_normalisation':
        sys.exit("Not yet implemented")
      elif scaling == 'l0_normalisation_adjusted':
        sys.exit("Not yet implemented")
      elif scaling == 'weights_removed':
        final_saliency[graph_layer.output_channel_idx] = layer_saliency / scale_w
  return final_saliency

def saliency_python_scaling(pruning_net, output_saliencies, input_saliencies, output_saliency=True, input_saliency=False, input_saliency_type='WEIGHT', scaling='L0'):
  if input_saliency and not(output_saliency):
    final_saliency = np.zeros(pruning_net.total_input_channels)
    if scaling == 'no_normalisation':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        final_saliency[graph_layer.input_channel_idx] = input_saliencies[l].data
    elif scaling == 'l1_normalisation':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        layer_scale_factor = np.abs(input_saliencies[l]).sum()
        if layer_scale_factor <= 0.0:
          layer_scale_factor = 1.0
        final_saliency[graph_layer.input_channel_idx] = input_saliencies[l] / layer_scale_factor
    elif scaling == 'l2_normalisation':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        layer_scale_factor = (input_saliencies[l]**2).sum()
        if layer_scale_factor <= 0.0:
          layer_scale_factor = 1.0
        else:
          layer_scale_factor = layer_scale_factor**0.5
        final_saliency[graph_layer.input_channel_idx] = input_saliencies[l] / layer_scale_factor
    elif scaling == 'l0_normalisation_adjusted':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        if input_saliency_type == 'ACTIVATION':
          bottom_layer = pruning_net.caffe_net.blobs[pruning_net.caffe_net.bottom_names[l][0]]
          layer_scale_factor = bottom_layer.height * bottom_layer.width
        elif input_saliency_type == 'WEIGHT':
          layer_scale_factor = graph_layer.active_output_channels.sum() * graph_layer.kernel_size
        else:
          sys.exit("Input saliency type not valid")
        final_saliency[graph_layer.input_channel_idx] = input_saliencies[l] / layer_scale_factor
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        layer_scale_factor = np.ones(graph_layer.input_channels)
        for i in range(graph_layer.input_channels):
          w = 0
          global_channel_idx = graph_layer.input_channel_idx[i]
          # Get number of local parameters
          w += graph_layer.active_output_channels.sum() * graph_layer.kernel_size
          # Get other linked sinks and sources
          sinks, sources = pruning_net.GetAllSinksSources(global_channel_idx, True)
          # Get their parameters
          for i_s in sinks:
            idx_c_sink, idx_conv_sink = pruning_net.GetChannelFromGlobalChannelIdx(i_s, True)
            sink_layer = pruning_net.graph[idx_conv_sink]
            if sink_layer.type == 'InnerProduct':
              w += sink_layer.active_output_channels.sum() * sink_layer.output_size * sink_layer.input_size
            elif sink_layer.type == 'Convolution':
              w += sink_layer.active_output_channels.sum() * sink_layer.kernel_size
          for i_s in sources:
            idx_c_source, idx_conv_source = pruning_net.GetChannelFromGlobalChannelIdx(i_s, False)
            source_layer = pruning_net.graph[idx_conv_source]
            if source_layer.type == 'Convolution':
              w += source_layer.active_input_channels.sum() * source_layer.kernel_size
          # scale saliency
          if w > 0 :
            layer_scale_factor[i] = w
          final_saliency[graph_layer.input_channel_idx] = input_saliencies[l] / layer_scale_factor
  elif output_saliency and not(input_saliency):
    final_saliency = np.zeros(pruning_net.total_output_channels)
    if scaling == 'no_normalisation':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        final_saliency[graph_layer.output_channel_idx] = output_saliencies[l]
    elif scaling == 'l1_normalisation':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        layer_scale_factor = np.abs(output_saliencies[l]).sum()
        if layer_scale_factor <= 0.0:
          layer_scale_factor = 1.0
        final_saliency[graph_layer.output_channel_idx] = output_saliencies[l] / layer_scale_factor
    elif scaling == 'l2_normalisation':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        layer_scale_factor = (output_saliencies[l]**2).sum()
        if layer_scale_factor <= 0.0:
          layer_scale_factor = 1.0
        else:
          layer_scale_factor = layer_scale_factor**0.5
        final_saliency[graph_layer.output_channel_idx] = output_saliencies[l] / layer_scale_factor
    elif scaling == 'l0_normalisation':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        if input_saliency_type == 'ACTIVATION':
          layer_scale_factor = graph_layer.height * graph_layer.width
        elif input_saliency_type == 'WEIGHT':
          layer_scale_factor = graph_layer.input_channels * graph_layer.kernel_size
        else:
          sys.exit("Input saliency type not valid")
        final_saliency[graph_layer.output_channel_idx] = output_saliencies[l] / layer_scale_factor
    elif scaling == 'l0_normalisation_adjusted':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        if input_saliency_type == 'ACTIVATION':
          layer_scale_factor = graph_layer.height * graph_layer.width
        elif input_saliency_type == 'WEIGHT':
          layer_scale_factor = graph_layer.active_input_channels.sum() * graph_layer.kernel_size
        else:
          sys.exit("Input saliency type not valid")
        final_saliency[graph_layer.output_channel_idx] = output_saliencies[l] / layer_scale_factor
    elif scaling == 'weights_removed':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        layer_scale_factor = np.ones(graph_layer.output_channels)
        for i in range(graph_layer.output_channels):
          w = 0
          global_channel_idx = graph_layer.output_channel_idx[i]
          # Get number of local parameters
          w += graph_layer.active_input_channels.sum() * graph_layer.kernel_size
          # Get other linked sinks and sources
          sinks, sources = pruning_net.GetAllSinksSources(global_channel_idx, False)
          # Get their parameters
          for i_s in sinks:
            idx_c_sink, idx_conv_sink = pruning_net.GetChannelFromGlobalChannelIdx(i_s, True)
            sink_layer = pruning_net.graph[idx_conv_sink]
            if sink_layer.type == 'InnerProduct':
              w += sink_layer.active_output_channels.sum() * sink_layer.output_size * sink_layer.input_size
            elif sink_layer.type == 'Convolution':
              w += sink_layer.active_output_channels.sum() * sink_layer.kernel_size
          for i_s in sources:
            idx_c_source, idx_conv_source = pruning_net.GetChannelFromGlobalChannelIdx(i_s, False)
            source_layer = pruning_net.graph[idx_conv_source]
            if source_layer.type == 'Convolution':
              w += source_layer.active_input_channels.sum() * source_layer.kernel_size
          # scale saliency
          if w > 0 :
            layer_scale_factor[i] = w
          final_saliency[graph_layer.output_channel_idx] = output_saliencies[l] / layer_scale_factor
    elif scaling == 'l0_and_weights_removed':
      for l in pruning_net.convolution_list:
        graph_layer = pruning_net.graph[l]
        layer_scale_factor = np.ones(graph_layer.output_channels)
        for i in range(graph_layer.output_channels):
          w = 0
          global_channel_idx = graph_layer.output_channel_idx[i]
          # Get number of local parameters
          w += graph_layer.active_input_channels.sum() * graph_layer.kernel_size
          # Get other linked sinks and sources
          sinks, sources = pruning_net.GetAllSinksSources(global_channel_idx, False)
          # Get their parameters
          for i_s in sinks:
            idx_c_sink, idx_conv_sink = pruning_net.GetChannelFromGlobalChannelIdx(i_s, True)
            sink_layer = pruning_net.graph[idx_conv_sink]
            if sink_layer.type == 'InnerProduct':
              w += sink_layer.active_output_channels.sum() * sink_layer.output_size * sink_layer.input_size
            elif sink_layer.type == 'Convolution':
              w += sink_layer.active_output_channels.sum() * sink_layer.kernel_size
          for i_s in sources:
            idx_c_source, idx_conv_source = pruning_net.GetChannelFromGlobalChannelIdx(i_s, False)
            source_layer = pruning_net.graph[idx_conv_source]
            if source_layer.type == 'Convolution':
              w += source_layer.active_input_channels.sum() * source_layer.kernel_size
          # scale saliency
          if w > 0 :
            if input_saliency_type == 'ACTIVATION':
              layer_scale_factor[i] = w * graph_layer.height * graph_layer.width
            elif input_saliency_type == 'WEIGHT':
              layer_scale_factor[i] = w* graph_layer.active_input_channels.sum() * graph_layer.kernel_size
            else:
              sys.exit("Input saliency type not valid")
          final_saliency[graph_layer.output_channel_idx] = output_saliencies[l] / layer_scale_factor
  elif output_saliency and input_saliency:
    sys.exit("Not yet implemented")
  return final_saliency

def test_network(net, itr):
  accuracy = dict()
  for i in range(itr):
    output = net.forward()
    for j in output.keys():
      if j in accuracy.keys():
        accuracy[j] = accuracy[j] + output[j]
      else:
        accuracy[j] = output[j].copy()
  for j in accuracy.keys():
    accuracy[j] /= float(itr)
  return accuracy['top-1']*100.0, accuracy['loss']

def get_pruning_signals_python(net, output_saliencies, saliency_pointwise, saliency_input, saliency_reduction, avg_a=True, avg_diff=True):
  for l in net.convolution_list:
    graph_layer = net.graph[l]
    caffe_layer = graph_layer.caffe_layer
    batch_num = net.caffe_net.layer_dict[l].blobs[0].shape[0]
    if (saliency_input == 'WEIGHT'):
      if (saliency_pointwise == 'TAYLOR'):
        pointwise_saliency = -1 * caffe_layer.blobs[0].data * caffe_layer.blobs[0].diff
      if (saliency_pointwise == 'TAYLOR_2ND_APPROX2'):
        pointwise_saliency = -1 * caffe_layer.blobs[0].data * caffe_layer.blobs[0].diff + ((caffe_layer.blobs[0].data * caffe_layer.blobs[0].diff)**2)/(2) 
      if saliency_reduction == 'L1':
        channel_saliency = np.abs(pointwise_saliency).sum(axis=(1,2,3))
      elif saliency_reduction == 'L2':
        channel_saliency = (pointwise_saliency **2).sum(axis=(1,2,3))
      elif saliency_reduction == 'ABS_SUM':
        channel_saliency = np.abs(pointwise_saliency.sum(axis=(2,3))).sum(axis=1)
      elif saliency_reduction == 'SQR_SUM':
        channel_saliency = (pointwise_saliency.sum(axis=(2,3))**2).sum(axis=1)
      elif saliency_reduction == 'NONE':
        channel_saliency = pointwise_saliency.sum(axis=(1,2,3))
    elif (saliency_input == 'ACTIVATION'):
      if avg_a and avg_diff:
        if (saliency_pointwise == 'TAYLOR'):
          pointwise_saliency = -1 * net.caffe_net.blobs[l].data.sum(axis=0) * net.caffe_net.blobs[l].diff.sum(axis=0)
        if (saliency_pointwise == 'TAYLOR_2ND_APPROX2'):
          pointwise_saliency = ((-1 * net.caffe_net.blobs[l].data.sum(axis=0) * net.caffe_net.blobs[layer].diff.sum(axis=0)) + ((net.caffe_net.blobs[l].data.sum(axis=0) *net.caffe_net.blobs[layer].diff.sum(axis=0) * batch_num )**2)) / batch_num
      elif avg_a:
        if (saliency_pointwise == 'TAYLOR'):
          pointwise_saliency = -1 * net.caffe_net.blobs[l].data.sum(axis=0) * net.caffe_net.blobs[l].diff
        if (saliency_pointwise == 'TAYLOR_2ND_APPROX2'):
          pointwise_saliency = ((-1 * net.caffe_net.blobs[l].data.sum(axis=0) * net.caffe_net.blobs[layer].diff) + ((net.caffe_net.blobs[l].data.sum(axis=0) *net.caffe_net.blobs[layer].diff * batch_num )**2)) / batch_num
      elif avg_diff:
        if (saliency_pointwise == 'TAYLOR'):
          pointwise_saliency = -1 * net.caffe_net.blobs[l].data * net.caffe_net.blobs[l].diff.sum(axis=0)
        if (saliency_pointwise == 'TAYLOR_2ND_APPROX2'):
          pointwise_saliency = ((-1 * net.caffe_net.blobs[l].data * net.caffe_net.blobs[layer].diff.sum(axis=0)) + ((net.caffe_net.blobs[l].data *net.caffe_net.blobs[layer].diff.sum(axis=0) * batch_num )**2)) / batch_num
      else:
        if (saliency_pointwise == 'TAYLOR'):
          pointwise_saliency = -1 * net.caffe_net.blobs[l].data * net.caffe_net.blobs[l].diff
        if (saliency_pointwise == 'TAYLOR_2ND_APPROX2'):
          pointwise_saliency = ((-1 * net.caffe_net.blobs[l].data * net.caffe_net.blobs[layer].diff) + ((net.caffe_net.blobs[l].data *net.caffe_net.blobs[layer].diff * batch_num )**2)) / batch_num
      
      if saliency_reduction == 'L1':
        channel_saliency = np.abs(pointwise_saliency).sum(axis=(0,2,3))
      elif saliency_reduction == 'L2':
        channel_saliency = (pointwise_saliency **2).sum(axis=(0,2,3))
      elif saliency_reduction == 'ABS_SUM':
        channel_saliency = np.abs(pointwise_saliency.sum(axis=(2,3))).sum(axis=0)
      elif saliency_reduction == 'SQR_SUM':
        channel_saliency = (pointwise_saliency.sum(axis=(2,3))**2).sum(axis=0)
      elif saliency_reduction == 'NONE':
        channel_saliency = pointwise_saliency.sum(axis=(0,2,3))
    output_saliencies[l] += channel_saliency


