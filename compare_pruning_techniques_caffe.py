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

saliency_pos_ = 4
mask_pos_ = 2

_caffe_saliencies_ = caffe._caffe.SALIENCY.names
_caffe_saliency_input_ = caffe._caffe.SALIENCY_INPUT.names 
_caffe_saliency_norm_ = caffe._caffe.SALIENCY_NORM.names

def layerwise_normalisation(x, layer, net):
  conv_module = net.layer_dict[layer]
  if args.normalisation == 'l0_normalisation':
    if args.input_channels_only and (args.saliency_input == 'WEIGHT'):
      return x / float(conv_module.output_channels * conv_module.kernel_size)
    elif (args.saliency_input == 'WEIGHT'):
      return x / float(conv_module.input_channels * conv_module.kernel_size)
    elif args.input_channels_only:
      ifm = net.blobs[net.bottom_names[layer][0]]
      return x / float(conv_module.output_channels * ifm.width * ifm.height)
    else:
      ofm = net.blobs[layer]
      return x / float(conv_module.input_channels * ofm.width * ofm.height)
  if args.normalisation == 'l0_normalisation_adjusted':
    if args.input_channels_only and (args.saliency_input == 'WEIGHT'):
      return x / float(conv_module.active_output_channels.sum() * conv_module.kernel_size)
    elif (args.saliency_input == 'WEIGHT'):
      return x / float(conv_module.active_input_channels.sum() * conv_module.kernel_size)
    elif args.input_channels_only:
      ifm = net.blobs[net.bottom_names[layer][0]]
      return x / float(conv_module.active_output_channels.sum() * ifm.width * ifm.height)
    else:
      ofm = net.blobs[layer]
      return x / float(conv_module.active_input_channels.sum() * ofm.width * ofm.height)
  if args.normalisation == 'weights_removed':
    return x / float(weights_removed(net, 0, conv_module, args.input_channels_only))
  elif args.normalisation == 'l1_normalisation':
    return x/float(np.abs(x).sum()) if float(np.abs(x).sum()) != 0 else x
  elif args.normalisation == 'l2_normalisation':
    return x/np.sqrt(np.power(x,2).sum()) if float(np.sqrt(np.power(x,2).sum())) != 0 else x
  elif args.normalisation == 'no_normalisation':
    return x

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parser():
    parser = argparse.ArgumentParser(description='Caffe Channel Pruning Example')
    parser.add_argument('--arch', action='store', default=None,
            help='the caffe solver to use')
    parser.add_argument('--arch-saliency', action='store', default=None,
            help='saliency prototxt to use')
    parser.add_argument('--pretrained', action='store', default=None,
            help='pretrained caffemodel')
    parser.add_argument('--retrain', type=str2bool, nargs='?', default=False,
            help='retrain the pruned network')
    parser.add_argument('--characterise', type=str2bool, nargs='?', default=False,
            help='characterise the pruned network')
    parser.add_argument('--tolerance', type=float, default='1.0',
            help='Drop in train loss before retraining starts')
    parser.add_argument('--filename', action='store', default='summary_',
            help='prefix for storing pruning data')
    parser.add_argument('--stop-acc', type=float, default='10.0',
            help='Stop pruning when test accuracy drops below this value')
    parser.add_argument('--method', action='store', default='FISHER',
            help='saliency method')
    parser.add_argument('--saliency-norm', action='store', default='NONE',
            help='Caffe saliency_norm')
    parser.add_argument('--saliency-input', action='store', default='ACTIVATION',
            help='Caffe saliency_input')
    parser.add_argument('--normalisation', action='store', default='no_normalisation',
            help='Layer-wise normalisation to use for saliency')
    parser.add_argument('--test-size', type=int, default=80, 
            help='Number of batches to use for testing')
    parser.add_argument('--train-size', type=int, default=200, 
            help='Number of batches to use for training')
    parser.add_argument('--eval-size', type=int, default=40, 
            help='Number of batches to use for evaluating the saliency')
    parser.add_argument('--test-interval', type=int, default=1, 
            help='After how many pruning steps to test')
    parser.add_argument('--input-channels-only', type=str2bool, nargs='?', default=False,
            help='prune input channels only')
    parser.add_argument('--input-output-channels', type=str2bool, nargs='?', default=False,
            help='prune input and output channels')
    return parser

def get_producer_convolution(net, initial_parents, producer_conv): 
  for initial_parent in initial_parents: 
    if '_split_' in initial_parent: 
      split_names = initial_parent.split('_split_') 
      initial_parent = split_names[0] + '_split' 
    if ('Convolution' not in net.layer_dict[initial_parent].type) and ('Data' not in net.layer_dict[initial_parent].type): 
      get_producer_convolution(net, net.bottom_names[initial_parent], producer_conv)
    else: 
      producer_conv.append(initial_parent) 

def get_children(net, layer): 
  children = [] 
  if '_split_' in layer: 
    parent_split = layer.split('_split_')[0] + '_split' 
    next_layers = list(net.layer_dict.keys())[(list(net.layer_dict.keys()).index(parent_split)+1):] 
  else: 
    next_layers = list(net.layer_dict.keys())[(list(net.layer_dict.keys()).index(layer)+1):] 
  for each in next_layers: 
    if ((layer in net.bottom_names[each]) and (layer not in net.top_names[each])): 
      children.append(each) 
   
  if ((layer in net.top_names.keys() and net.layer_dict[layer].type == 'Split')): 
    children = list(net.top_names[layer]) 
  return children

def get_consumer_convolution_or_fc(net, initial_consumers, consumer_sink): 
  for initial_consumer in initial_consumers: 
    if '_split_' in initial_consumer: 
      split_names = initial_consumer.split('_split_') 
      parent_split = split_names[0] + '_split' 
      consumer_type = 'Split' 
    else: 
      consumer_type = net.layer_dict[initial_consumer].type 
    if ('Convolution' not in consumer_type) and ('InnerProduct' not in consumer_type): 
      get_consumer_convolution_or_fc(net, get_children(net, initial_consumer), consumer_sink) 
    else: 
      consumer_sink.append(initial_consumer) 

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

def UpdateMask(net, idx_channel, conv_module, fill, final=True, input=False):
  if 'Convolution' in conv_module.type:
    bias = conv_module.bias_term_
    prune = fill == 0
    if input:
      if final and prune:
        conv_module.blobs[0].data[:,idx_channel,:,:].fill(0)
      conv_module.blobs[conv_module.mask_pos_].data[:,idx_channel,:,:].fill(fill)
      conv_module.active_input_channels[idx_channel] = fill
    else:
      if final and prune:
        conv_module.blobs[0].data[idx_channel].fill(0)
      if final and prune and bias:
        conv_module.blobs[1].data[idx_channel] = 0
      conv_module.blobs[conv_module.mask_pos_].data[idx_channel].fill(fill)
      if bias:
        conv_module.blobs[conv_module.mask_pos_+1].data[idx_channel] = 0
        conv_module.active_output_channels[idx_channel] = fill

def PruneChannel(net, pruned_channel, convolution_list, channels, prune=True, final=True, input=False):
  fill = 0 if prune else 1
  idx = np.where(channels>pruned_channel)[0][0]
  idx_convolution = convolution_list[idx]
  idx_channel = (pruned_channel - channels[idx-1]) if idx > 0 else pruned_channel
  conv_module = net.layer_dict[idx_convolution]
  UpdateMask(net, idx_channel, conv_module, fill, final, input)
  if input: # remove previous conv's output channel
    for c in conv_module.sources:
      if net.layer_dict[c].type == 'Convolution':
        UpdateMask(net, idx_channel, net.layer_dict[c], fill, final, input=False)
  else: # remove next conv's input channel
    for c in conv_module.sinks:
      if net.layer_dict[c].type == 'Convolution':
        UpdateMask(net, idx_channel, net.layer_dict[c], fill, final, input=True)

def weights_removed(net, idx_channel, conv_module, input=False):
  num_weights = 0
  if input:
    num_weights += (conv_module.active_output_channels.sum() * conv_module.kernel_size)
    for l in conv_module.sources:
      c = net.layer_dict[l]
      if 'Convolution' in c.type:
        num_weights += (c.active_input_channels.sum() * c.kernel_size)
        if c.bias_term_:
          num_weights += 1
  else:
    num_weights += (conv_module.active_input_channels.sum() * conv_module.kernel_size)
    if conv_module.bias_term_:
      num_weights += 1
    for l in conv_module.sinks:
      c = net.layer_dict[l]
      if 'Convolution' in c.type:
        num_weights += (c.active_output_channels.sum() * c.kernel_size)
      elif 'InnerProduct' in c.type:
        num_weights += (c.input_size)
  return num_weights

if __name__=='__main__':
  start = time.time()
  args = parser().parse_args()
  if args.arch is None:
    print("Caffe solver needed")
    exit(1)
  #  Allocate two models : one for masking weights and retraining (weights + mask)
  #                          one for computing saliency (weights + masks + saliency)
  caffe.set_mode_gpu()
  net = caffe.Net(args.arch_saliency, caffe.TEST) # net used to compute saliency
  saliency_solver = caffe.SGDSolver(args.arch)
  saliency_solver.net.copy_from(args.pretrained)
  saliency_solver.test_nets[0].share_with(saliency_solver.net)
  net.share_with(saliency_solver.net) 
  
  convolution_list = list(filter(lambda x: 'Convolution' in net.layer_dict[x].type, net.layer_dict.keys()))
  named_modules = net.layer_dict
  
  #reset mask
  for layer in convolution_list:
    conv_module = named_modules[layer]
    conv_module.blobs[conv_module.mask_pos_].data.fill(1) # reset mask
    if conv_module.bias_term_:
      conv_module.blobs[conv_module.mask_pos_+1].data.fill(1) # reset mask
    conv_module.batch = net.blobs[layer].num
    conv_module.height = net.blobs[layer].height
    conv_module.width = net.blobs[layer].width
    conv_module.output_channels = conv_module.blobs[0].data.shape[0]
    conv_module.input_channels = conv_module.blobs[0].data.shape[1]
    conv_module.kernel_size = conv_module.blobs[0].data.shape[2] * conv_module.blobs[0].data.shape[3]
    conv_module.sources = []
    conv_module.sinks = []
    get_producer_convolution(net, net.bottom_names[layer], conv_module.sources)
    get_consumer_convolution_or_fc(net, get_children(net, layer), conv_module.sinks)
    input_layer = net.bottom_names[layer][0] 
    conv_module.group = int(net.blobs[input_layer].channels / conv_module.blobs[0].channels) 
    conv_module.active_input_channels = np.ones(conv_module.input_channels)
    conv_module.active_ifm = np.ones(conv_module.input_channels * conv_module.group)
    conv_module.active_output_channels = np.ones(conv_module.output_channels)
    for l in conv_module.sinks:
      if 'InnerProduct' in net.layer_dict[l].type:
        net.layer_dict[l].input_channels = conv_module.output_channels
        net.layer_dict[l].input_size = net.layer_dict[l].blobs[0].data.shape[0] / conv_module.output_channels
        net.layer_dict[l].active_input_channels = np.ones(conv_module.output_channels)

  if args.method in _caffe_saliencies_.keys():
    for layer in convolution_list:
      conv_module = named_modules[layer]
      conv_module.saliency_ = _caffe_saliencies_[args.method] 
      conv_module.saliency_norm_ = _caffe_saliency_norm_[args.saliency_norm]
      conv_module.saliency_input_ = _caffe_saliency_input_[args.saliency_input]

  method = args.method
  initial_density = 0
  total_param = 0 
  total_output_channels = 0
  total_input_channels = 0
  output_channels = []
  input_channels = []
  for layer in convolution_list:
    total_output_channels += named_modules[layer].blobs[0].num
    total_input_channels += named_modules[layer].blobs[0].channels
    output_channels.append(named_modules[layer].blobs[0].num)
    input_channels.append(named_modules[layer].blobs[0].channels)
    initial_density += named_modules[layer].blobs[0].num * named_modules[layer].blobs[0].height * named_modules[layer].blobs[0].width * named_modules[layer].blobs[0].channels
  output_channels = np.array(output_channels)
  output_channels = np.cumsum(output_channels)
  input_channels = np.array(input_channels)
  input_channels = np.cumsum(input_channels)
  if args.input_channels_only:
    total_channels = total_input_channels
    channels = input_channels
  else:
    total_channels = total_output_channels
    channels = output_channels
  print('Total number of channels to be considered for pruning: ', total_channels)

  summary = dict()
  summary['test_acc'] = np.zeros(total_channels)
  summary['test_loss'] = np.zeros(total_channels)
  summary['pruned_channel'] = np.zeros(total_channels)
  summary['method'] = method + '-' + args.normalisation
  active_channel = list(range(total_channels))
  summary['initial_param'] = initial_density
  test_acc, ce_loss = test(saliency_solver, args.test_size)
  summary['initial_test_acc'] = test_acc
  summary['initial_test_loss'] = ce_loss
  summary['eval_loss'] = np.zeros(total_channels)
  summary['eval_acc'] = np.zeros(total_channels)
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
  summary['initial_eval_loss'] = initial_eval_loss
  summary['initial_eval_acc'] = initial_eval_acc
  
  # Train accuracy and loss
  initial_train_acc = 0.0
  initial_train_loss = 0.0
  if args.characterise:
    summary['train_loss'] = np.zeros(total_channels)
    summary['train_acc'] = np.zeros(total_channels)
    current_loss = saliency_solver.net.forward()['loss']
    summary['retraining_loss'] = np.empty(total_channels, dtype=np.object)
    summary['retraining_acc'] = np.empty(total_channels, dtype=np.object)
    for i in range(100):
      output = saliency_solver.net.forward()
      initial_train_loss += output['loss']
      initial_train_acc += output['top-1']
    initial_train_loss /= 100.0
    initial_train_acc /= 100.0
    print('Initial train loss', initial_train_loss)
    print('Initial train acc', initial_train_acc)
    summary['initial_train_loss'] = initial_train_loss
    summary['initial_train_acc'] = initial_train_acc
  train_loss_upper_bound = (100 + args.tolerance) * initial_train_loss / 100.0
  train_acc_lower_bound = (100 - args.tolerance) * initial_train_acc / 100.0

  for j in range(total_channels): 
    if method in _caffe_saliencies_:
      for layer in convolution_list:
        named_modules[layer].blobs[named_modules[layer].saliency_pos_].data.fill(0) # reset saliency
        named_modules[layer].blobs[named_modules[layer].saliency_pos_+1].data.fill(0) # reset saliency
        if args.input_channels_only :
          named_modules[layer].saliency_input_channel_compute_ = True
          named_modules[layer].saliency_output_channel_compute_ = False
    pruning_signal = np.array([])
  
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
      if (method == 'apoz'):
        pruning_signal_partial = np.array([])
        if (args.saliency_input == 'ACTIVATION'):
          for layer in convolution_list:
            if args.input_channels_only:
              feature_map_name = net.bottom_names[layer][0]
            else:
              feature_map_name = layer
            saliency_data = (net.blobs[feature_map_name].data > 0.0).sum(axis=(0,2,3)) / float(conv_module.batch * conv_module.height * conv_module.width)
            saliency_normalised = layerwise_normalisation(saliency_data, layer, net)
            pruning_signal_partial = np.hstack([pruning_signal_partial, saliency_normalised])
          if iter == 0:
            pruning_signal = pruning_signal_partial.data
          else:
            pruning_signal += pruning_signal_partial
          
        else:
          print('Not implemented')
          sys.exit(-1)
    summary['eval_loss'][j] = current_eval_loss / float(iter+1)
    summary['eval_acc'][j] = current_eval_acc / float(iter+1)
    
    if (method != 'WEIGHT_AVG') or (args.saliency_input != 'WEIGHT'):
      pruning_signal /= float(evalset_size) # get approximate change in loss using taylor expansions
          

    if method in _caffe_saliencies_:
      for layer in convolution_list:
        if args.input_channels_only:
          saliency_data = named_modules[layer].blobs[named_modules[layer].saliency_pos_+1].data[0]
        else:
          saliency_data = named_modules[layer].blobs[named_modules[layer].saliency_pos_].data[0]
        saliency_normalised = layerwise_normalisation(saliency_data, layer, net)
        pruning_signal = np.hstack([pruning_signal, saliency_normalised])
    
    if method == 'random':
      pruning_signal = np.zeros(total_channels)
      pruning_signal[random.sample(active_channel, 1)] = -1

    prune_channel_idx = np.argmin(pruning_signal[active_channel])
    prune_channel = active_channel[prune_channel_idx]
    PruneChannel(net, prune_channel, convolution_list, channels, final=True, input=args.input_channels_only)
    
    if args.characterise:
      output_train = saliency_solver.net.forward()
      current_loss = output_train['loss']
      current_acc = output_train['top-1']
      summary['train_loss'][j] = current_loss
      summary['train_acc'][j] = current_acc
      retraining_loss = np.array([])
      retraining_acc = np.array([])
      for i in range(args.train_size):
#        if (current_loss <= train_loss_upper_bound):
        if ((i==0) and (current_acc >= train_acc_lower_bound)):
          break
        if (current_acc >= initial_train_acc):
          break
        saliency_solver.net.clear_param_diffs()
        output_train = saliency_solver.net.forward()
        current_loss = output_train['loss']
        current_acc = output_train['top-1']
        saliency_solver.net.backward()
        saliency_solver.apply_update()
        retraining_loss = np.hstack([retraining_loss, current_loss])
        retraining_acc = np.hstack([retraining_acc, current_acc])
      summary['retraining_loss'][j] = retraining_loss.copy()
      summary['retraining_acc'][j] = retraining_acc.copy()

    if args.retrain:
      saliency_solver.step(args.train_size)
    
    if (j % args.test_interval == 0):
      test_acc, ce_loss = test(saliency_solver, args.test_size)
    summary['test_acc'][j] = test_acc
    summary['test_loss'][j] = ce_loss
    summary['pruned_channel'][j] = prune_channel
    print(args.normalisation, method, ' Step: ', j +1,'  ||   Remove Channel: ', prune_channel, '  ||  Test Acc: ', test_acc)
    active_channel.remove(prune_channel)
    
    if test_acc < args.stop_acc:
        break
  # if (j % 100) == 0 :
  #      np.save(args.filename+'.partial', summary)

  end = time.time()
  summary['exec_time'] = end - start
  np.save(args.filename, summary)

  caffe.set_mode_cpu()

