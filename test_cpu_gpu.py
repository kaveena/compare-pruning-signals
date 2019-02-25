import caffe
import os
import struct
import sys
import random
import numpy as np
import caffe
import argparse
import random

sys.dont_write_bytecode = True

saliency_pos_ = 4
mask_pos_ = 2

_caffe_saliencies_ = ['fisher', 'taylor', 'hessian_diag', 'hessian_diag_approx2', 'min-weight']

def parser():
    parser = argparse.ArgumentParser(description='Caffe Channel Pruning Example')
    parser.add_argument('--arch', action='store', default='LeNet-5-CIFAR10',
            help='the caffe solver to use')
    return parser

args = parser().parse_args()

prototxt = '/home/kaveena/compare-pruning-signals/caffe-pruned-models/' + args.arch + '/one-image.prototxt' 
caffemodel = '/home/kaveena/compare-pruning-signals/caffe-pruned-models/' + args.arch + '/original.caffemodel' 

if args.arch is None:
  print("Caffe solver needed")
  exit(1)
#  Allocate two models:  one that cumputes saliencies (weights + masks + saliencies)
#                        one for training (weights + masks)
#  saliency_solver = caffe.SGDSolver(args.arch)
#  saliency_solver.net.copy_from(args.pretrained)
#  saliency_solver.test_nets[0].copy_from(args.pretrained)
#  net = saliency_solver.test_nets[0]
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

convolution_list = list(filter(lambda x: 'Convolution' in net.layer_dict[x].type, net.layer_dict.keys()))
named_modules = net.layer_dict

gpu = dict()
cpu = dict()
gpu_diff = dict()
cpu_diff = dict()
gpu_ddiff = dict()
cpu_ddiff = dict()

evalset_size = 1

for k in convolution_list:
  named_modules[k].saliency_ = caffe._caffe.SALIENCY.HESSIAN_DIAG

caffe.set_mode_cpu()
net.clear_param_diffs();
for iter in range(evalset_size):
  net.forward()
  net.backward()
for k in net.blobs.keys():
   cpu[k] = net.blobs[k].data.copy()
   cpu_diff[k] = net.blobs[k].diff.copy()
   cpu_ddiff[k] = net.blobs[k].ddiff.copy()

caffe.set_mode_gpu()
net.clear_param_diffs();
for iter in range(evalset_size):
  net.forward()
  net.backward()
for k in net.blobs.keys():
   gpu[k] = net.blobs[k].data.copy()
   gpu_diff[k] = net.blobs[k].diff.copy()
   gpu_ddiff[k] = net.blobs[k].ddiff.copy()

cpu_gpu_equal = True
for k in net.blobs.keys():
  cpu_gpu_equal *= np.allclose(cpu[k], gpu[k], rtol=1e-3)
if cpu_gpu_equal:
  print("Outputs match")
cpu_gpu_equal = True
for k in net.blobs.keys():
  cpu_gpu_equal *= np.allclose(cpu_diff[k], gpu_diff[k], rtol=1e-3)
if cpu_gpu_equal:
  print("Diff match")
cpu_gpu_equal = True
for k in net.blobs.keys():
  cpu_gpu_equal *= np.allclose(cpu_ddiff[k], gpu_ddiff[k], rtol=1e-3)
if cpu_gpu_equal:
  print("Ddiff match")

for saliency in caffe._caffe.SALIENCY.names.values():
  cpu_gpu_equal = True
  
  caffe.set_mode_cpu()
  if (saliency == caffe._caffe.SALIENCY.ALL):
    continue
  for k in convolution_list:
    named_modules[k].saliency_ = saliency
    named_modules[k].blobs[saliency_pos_].data.fill(0)
  net.clear_param_diffs();
  # compute saliency    
  for iter in range(evalset_size):
    net.forward()
    net.backward()
  for k in convolution_list:
     cpu[k] = named_modules[k].blobs[saliency_pos_].data[0].copy()
  
  caffe.set_mode_gpu()
  for k in convolution_list:
    named_modules[k].saliency_ = saliency
    named_modules[k].blobs[saliency_pos_].data.fill(0)
  net.clear_param_diffs();
  # compute saliency    
  for iter in range(evalset_size):
    net.forward()
    net.backward()
  for k in convolution_list:
     gpu[k] = named_modules[k].blobs[saliency_pos_].data[0].copy()

  print("Activation " , saliency)
  for k in convolution_list:
    cpu_gpu_equal *= np.allclose(cpu[k], gpu[k])
  if cpu_gpu_equal:
    print("Match")

# Weight based 
for k in convolution_list:
  named_modules[k].saliency_input_ = caffe._caffe.SALIENCY_INPUT.WEIGHT
for saliency in caffe._caffe.SALIENCY.names.values():
  cpu_gpu_equal = True
  
  caffe.set_mode_cpu()
  if (saliency == caffe._caffe.SALIENCY.ALL):
    continue
  for k in convolution_list:
    named_modules[k].saliency_ = saliency
    named_modules[k].blobs[saliency_pos_].data.fill(0)
  net.clear_param_diffs();
  # compute saliency    
  for iter in range(evalset_size):
    net.forward()
    net.backward()
  for k in convolution_list:
     cpu[k] = named_modules[k].blobs[saliency_pos_].data[0].copy()
  
  caffe.set_mode_gpu()
  for k in convolution_list:
    named_modules[k].saliency_ = saliency
    named_modules[k].blobs[saliency_pos_].data.fill(0)
  net.clear_param_diffs();
  # compute saliency    
  for iter in range(evalset_size):
    net.forward()
    net.backward()
  for k in convolution_list:
     gpu[k] = named_modules[k].blobs[saliency_pos_].data[0].copy()

  print("Weight " , saliency)
  for k in convolution_list:
    cpu_gpu_equal *= np.allclose(cpu[k], gpu[k])
  if cpu_gpu_equal:
    print("Match")

