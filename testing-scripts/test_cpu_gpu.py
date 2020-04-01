import caffe
import os
import struct
import sys
import random
import numpy as np
import caffe
import argparse
import random
import tempfile
from utils.utils import *
from utils.pruning_graph import *

sys.dont_write_bytecode = True

def parser():
    parser = argparse.ArgumentParser(description='Caffe Channel Pruning Example')
    parser.add_argument('--arch', action='store', default='AlexNet',
            help='the network architecture to use')
    parser.add_argument('--dataset', action='store', default='CIFAR10',
            help='the dataset to use')
    return parser

args = parser().parse_args()

saliency_prototxt_file, saliency_prototxt_filename = tempfile.mkstemp()
small_images_file, small_images_filename = tempfile.mkstemp()

arch_solver = 'trained-models/' + args.arch + '-' + args.dataset + '/solver-gpu.prototxt'
pretrained = 'trained-models/' + args.arch + '-' + args.dataset + '/original.caffemodel'

solver_prototxt = get_solver_proto_from_file(arch_solver)
original_prototxt_filename = solver_prototxt.net
# Create a new network that uses mask blobs to be used with the solver
saliency_prototxt = get_prototxt_from_file(original_prototxt_filename)
for l in saliency_prototxt.layer:
  if l.type == 'ImageData':
    original_image_source = l.image_data_param.source
    break
os.system("cat " + original_image_source + " | head -n 128 > " + small_images_filename)

# Create a prototxt that uses the validation set and has saliency computation

for pointwise_saliency in caffe_pb2.ConvolutionSaliencyParameter.SALIENCY.keys():
  saliency_prototxt = get_prototxt_from_file(original_prototxt_filename)
  add_mask_to_prototxt(saliency_prototxt)
  for l in saliency_prototxt.layer:
    if l.type == 'ImageData':
      l.image_data_param.source = small_images_filename
      l.image_data_param.shuffle = False
      l.image_data_param.batch_size = 128
      l.transform_param.mirror = False
  add_saliency_to_prototxt(saliency_prototxt, [pointwise_saliency], ['ACTIVATION'], ['NONE'])
  saliency_prototxt.compute_2nd_derivative = True

  save_prototxt_to_file(saliency_prototxt, saliency_prototxt_filename)

  caffe._caffe.set_allocate_2nd_derivative_memory(True)
  net = caffe.Net(saliency_prototxt_filename, pretrained, caffe.TEST)
  pruning_net = PruningGraph(net, saliency_prototxt)

  named_modules = net.layer_dict

  gpu = dict()
  cpu = dict()
  gpu_diff = dict()
  cpu_diff = dict()
  gpu_ddiff = dict()
  cpu_ddiff = dict()
  gpu_saliency = dict()
  cpu_saliency = dict()

  evalset_size = 1

  caffe.set_mode_cpu()
  pruning_net.ClearSaliencyBlobs()
  net.clear_param_diffs();
  for iter in range(evalset_size):
    net.forward()
    net.backward()
  for k in net.blobs.keys():
     cpu[k] = net.blobs[k].data.copy()
     cpu_diff[k] = net.blobs[k].diff.copy()
     cpu_ddiff[k] = net.blobs[k].ddiff.copy()
  for k in pruning_net.convolution_list:
     cpu_saliency[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0].copy()

  caffe.set_mode_gpu()
  pruning_net.ClearSaliencyBlobs()
  net.clear_param_diffs();
  for iter in range(evalset_size):
    net.forward()
    net.backward()
  for k in net.blobs.keys():
     gpu[k] = net.blobs[k].data.copy()
     gpu_diff[k] = net.blobs[k].diff.copy()
     gpu_ddiff[k] = net.blobs[k].ddiff.copy()
  for k in pruning_net.convolution_list:
     gpu_saliency[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0].copy()

  cpu_gpu_equal = True
  for k in net.blobs.keys():
    cpu_gpu_equal *= np.allclose(cpu[k], gpu[k], rtol=1e-3)
  if cpu_gpu_equal:
    print("Outputs match")
  else:
    print("Outputs DON'T match")
    
  cpu_gpu_equal = True
  for k in net.blobs.keys():
    cpu_gpu_equal *= np.allclose(cpu_diff[k], gpu_diff[k], rtol=1e-3)
  if cpu_gpu_equal:
    print("Diff match")
  else:
    print("Diff DON'T match")
  cpu_gpu_equal = True
  for k in net.blobs.keys():
    cpu_gpu_equal *= np.allclose(cpu_ddiff[k], gpu_ddiff[k], rtol=1e-3)
  if cpu_gpu_equal:
    print("Ddiff match")
  else:
    print("Ddiff DON'T match")

  cpu_gpu_equal = True
  
  print("Activation " , pointwise_saliency)
  for k in pruning_net.convolution_list:
    cpu_gpu_equal *= np.allclose(cpu_saliency[k], gpu_saliency[k])
  if cpu_gpu_equal:
    print("Match")
  else:
    print("DON'T Match")
