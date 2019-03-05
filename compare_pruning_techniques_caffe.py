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

l0_normalisation = lambda x, n, m, h, w, c, k, input_type : x/float(c * k * k) if (input_type=="WEIGHT") else x/float(h * w)
l1_normalisation = lambda x, n, m, h, w, c, k, input_typ : x/float(np.abs(x).sum()) if float(np.abs(x).sum()) != 0 else x
l2_normalisation = lambda x, n, m, h, w, c, k, input_typ : x/np.sqrt(np.power(x,2).sum()) if float(np.sqrt(np.power(x,2).sum())) != 0 else x
no_normalisation = lambda x, n, m, h, w, c, k, input_typ : x

def parser():
    parser = argparse.ArgumentParser(description='Caffe Channel Pruning Example')
    parser.add_argument('--arch', action='store', default=None,
            help='the caffe solver to use')
    parser.add_argument('--arch-saliency', action='store', default=None,
            help='saliency prototxt to use')
    parser.add_argument('--pretrained', action='store', default=None,
            help='pretrained caffemodel')
    parser.add_argument('--retrain', action='store_true', default=False,
            help='retrain the pruned network')
    parser.add_argument('--characterise', action='store_true', default=False,
            help='characterise the pruned network')
    parser.add_argument('--tolerance', type=float, default='1.0',
            help='Drop in train loss before retraining starts')
    parser.add_argument('--prune', action='store_true', default=False,
            help='prune network')
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
    parser.add_argument('--input-channels-only', action='store_true', default=False,
            help='prune input channels only')
    parser.add_argument('--input-output-channels', action='store_true', default=False,
            help='prune input and output channels')
    return parser

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

def UpdateMask(net, pruned_channel, convolution_list, channels, prune=True, final=True, input=False):
  fill = 0 if prune else 1
  idx = np.where(channels>pruned_channel)[0][0]
  idx_convolution = convolution_list[idx]
  idx_channel = (pruned_channel - channels[idx-1]) if idx > 0 else pruned_channel
  conv_module = net.layer_dict[idx_convolution]
  bias = conv_module.bias_term_
  if input:
    if final and prune:
      conv_module.blobs[0].data[:,idx_channel,:,:].fill(0)
    conv_module.blobs[conv_module.mask_pos_].data[:,idx_channel,:,:].fill(fill)
  else:
    if final and prune:
      conv_module.blobs[0].data[idx_channel].fill(0)
    if final and prune and bias:
      conv_module.blobs[1].data[idx_channel] = 0
    conv_module.blobs[conv_module.mask_pos_].data[idx_channel].fill(fill)
    if bias:
      conv_module.blobs[conv_module.mask_pos_+1].data[idx_channel] = 0

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
    named_modules[layer].blobs[named_modules[layer].mask_pos_].data.fill(1) # reset saliency
    if named_modules[layer].bias_term_:
      named_modules[layer].blobs[named_modules[layer].mask_pos_+1].data.fill(1) # reset saliency

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
            caffe_layer = net.blobs[feature_map_name]
            n = caffe_layer.num
            m = caffe_layer.channels
            h = net.blobs[feature_map_name].height
            w = net.blobs[feature_map_name].width
            if (args.input_channels_only):
              c = named_modules[layer].blobs[0].data.shape[0]
            else:
              c = named_modules[layer].blobs[0].data.shape[1]
            k = named_modules[layer].blobs[0].data.shape[2]
            saliency_data = (net.blobs[feature_map_name].data > 0.0).sum(axis=(0,2,3)) / float(n * h * w)
            exec('saliency_normalised='+args.normalisation+'(saliency_data, n, m, h , w, c , k, args.saliency_input)')
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
        caffe_layer = net.blobs[layer]
        if args.input_channels_only:
          feature_map_name = net.bottom_names[layer][0]
        else:
          feature_map_name = layer
        caffe_layer = net.blobs[feature_map_name]
        n = caffe_layer.num
        m = caffe_layer.channels
        h = caffe_layer.height
        w = caffe_layer.width
        if (args.input_channels_only):
          c = named_modules[layer].blobs[0].data.shape[0]
        else:
          c = named_modules[layer].blobs[0].data.shape[1]
        k = named_modules[layer].blobs[0].data.shape[2]
        exec('saliency_normalised='+args.normalisation+'(saliency_data, n, m, h , w, c , k, args.saliency_input)')
        pruning_signal = np.hstack([pruning_signal, saliency_normalised])
    
    if method == 'random':
      pruning_signal = np.zeros(total_channels)
      pruning_signal[random.sample(active_channel, 1)] = -1

    prune_channel_idx = np.argmin(pruning_signal[active_channel])
    prune_channel = active_channel[prune_channel_idx]
    UpdateMask(net, prune_channel, convolution_list, channels, final=True, input=args.input_channels_only)
    
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

