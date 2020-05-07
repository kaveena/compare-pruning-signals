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

_caffe_saliencies_ = ['fisher', 'taylor', 'hessian_diag', 'hessian_diag_approx2', 'min-weight']

def parser():
    parser = argparse.ArgumentParser(description='Caffe Channel Pruning Example')
    parser.add_argument('--arch', action='store', default='LeNet-5',
            help='the caffe solver to use')
    parser.add_argument('--dataset', action='store', default='CIFAR10',
            help='the dataset to use')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu')
    return parser

args = parser().parse_args()

saliency_prototxt_file, saliency_prototxt_filename = tempfile.mkstemp()
arch_solver = 'trained-models/' + args.arch + '-' + args.dataset + '/solver-gpu.prototxt'
pretrained = 'trained-models/' + args.arch + '-' + args.dataset + '/original.caffemodel'
if args.gpu:
  caffe.set_mode_gpu()
else:
  caffe.set_mode_cpu()

solver_prototxt = get_solver_proto_from_file(arch_solver)
original_prototxt_filename = solver_prototxt.net

hessian_diag = dict()
hessian_diag_caffe = dict()
hessian_diag_approx2 = dict()
hessian_diag_approx2_caffe = dict()
taylor_2nd = dict()
taylor_2nd_caffe = dict()

# Create a prototxt that uses the validation set and has saliency computation
for pointwise_saliency in ['HESSIAN_DIAG_APPROX1', 'HESSIAN_DIAG_APPROX2', 'TAYLOR_2ND_APPROX1']:
  saliency_prototxt = get_prototxt_from_file(original_prototxt_filename)
  add_mask_to_prototxt(saliency_prototxt)
  add_saliency_to_prototxt(saliency_prototxt, [pointwise_saliency], ['ACTIVATION'], ['NONE'])
  saliency_prototxt.compute_2nd_derivative = True

  save_prototxt_to_file(saliency_prototxt, saliency_prototxt_filename)

  caffe._caffe.set_allocate_2nd_derivative_memory(True)
  net = caffe.Net(saliency_prototxt_filename, pretrained, caffe.TEST)
  pruning_net = PruningGraph(net, saliency_prototxt)

  pruning_net.ClearSaliencyBlobs()
  net.clear_param_diffs();

  hessian_diag_correct = True
  hessian_diag_approx2_correct = True
  taylor_2nd_correct = True

  # compute saliency    
  evalset_size = 1;
  for iter in range(evalset_size):
    net.forward()
    net.backward()

#   print("Check Hessian Diag")
  for k in pruning_net.convolution_list:
     hessian_diag_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
     hessian_diag[k] = ((net.blobs[k].data**2) * net.blobs[k].ddiff).sum(axis=(0,2,3)) / (2*net.blobs[k].num)
     hessian_diag_correct*=np.allclose(hessian_diag_caffe[k], hessian_diag[k])

#   print("Check Hessian Diag Approx 2")
  for k in pruning_net.convolution_list:
     hessian_diag_approx2_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
     hessian_diag_approx2[k] = ((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)**2).sum(axis=(0,2,3)) / (2*net.blobs[k].num)
     hessian_diag_approx2_correct*=(np.allclose(hessian_diag_approx2_caffe[k], hessian_diag_approx2[k]))

#   print("Check Taylor 2nd")
  for k in pruning_net.convolution_list:
     taylor_2nd_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
     taylor_2nd[k] = ((-1*net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)  + ((net.blobs[k].data**2 * net.blobs[k].ddiff))/2).sum(axis=(0,2,3)) / (net.blobs[k].num)
     taylor_2nd_correct*=(np.allclose(taylor_2nd_caffe[k], taylor_2nd[k]))

  print(pointwise_saliency)
  if (hessian_diag_correct):
    print("Hessian Diag")
  if (hessian_diag_approx2_correct):
    print("Hessian Diag Approx2")
  if (taylor_2nd_correct):
    print("Taylor 2nd")

# Weight based 
for pointwise_saliency in ['HESSIAN_DIAG_APPROX1', 'HESSIAN_DIAG_APPROX2', 'TAYLOR_2ND_APPROX1']:
  saliency_prototxt = get_prototxt_from_file(original_prototxt_filename)
  add_mask_to_prototxt(saliency_prototxt)
  add_saliency_to_prototxt(saliency_prototxt, [pointwise_saliency], ['WEIGHT'], ['NONE'])
  saliency_prototxt.compute_2nd_derivative = True

  save_prototxt_to_file(saliency_prototxt, saliency_prototxt_filename)

  caffe._caffe.set_allocate_2nd_derivative_memory(True)
  net = caffe.Net(saliency_prototxt_filename, pretrained, caffe.TEST)
  pruning_net = PruningGraph(net, saliency_prototxt)

  pruning_net.ClearSaliencyBlobs()
  net.clear_param_diffs();

  hessian_diag_correct = True
  hessian_diag_approx2_correct = True
  taylor_2nd_correct = True

  # compute saliency    
  evalset_size = 1;
  for iter in range(evalset_size):
    net.forward()
    net.backward()

#   print("Check Hessian Diag")
  for k in pruning_net.convolution_list:
     hessian_diag_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
     hessian_diag[k] = (((net.layer_dict[k].blobs[0].data**2) * net.layer_dict[k].blobs[0].ddiff).sum(axis=(1,2,3))  + (net.layer_dict[k].blobs[1].data * net.layer_dict[k].blobs[1].ddiff))/ (2*net.blobs[k].num)
     hessian_diag_correct*=np.allclose(hessian_diag_caffe[k], hessian_diag[k])

#   print("Check Hessian Diag Approx 2")
  for k in pruning_net.convolution_list:
     hessian_diag_approx2_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
     hessian_diag_approx2[k] = ((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)**2).sum(axis=(0,2,3)) / (2*net.blobs[k].num)
     hessian_diag_approx2_correct*=(np.allclose(hessian_diag_approx2_caffe[k], hessian_diag_approx2[k]))

#   print("Check Taylor 2nd")
  for k in pruning_net.convolution_list:
     taylor_2nd_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
     taylor_2nd[k] = (-1 * net.layer_dict[k].blobs[0].data * net.layer_dict[k].blobs[0].diff * net.blobs[k].num).sum(axis=(1,2,3)) / (net.blobs[k].num) + (-1 * net.layer_dict[k].blobs[1].data * net.layer_dict[k].blobs[1].diff) + (net.layer_dict[k].blobs[0].data**2 * net.layer_dict[k].blobs[0].ddiff/2).sum(axis=(1,2,3)) / (net.blobs[k].num) + (net.layer_dict[k].blobs[1].data**2 * net.layer_dict[k].blobs[1].ddiff/2 ) / (net.blobs[k].num)
     taylor_2nd_correct*=(np.allclose(taylor_2nd_caffe[k], taylor_2nd[k], rtol=1e-4))
  
  print(pointwise_saliency)
  if (hessian_diag_correct):
    print("Hessian Diag")
  if (hessian_diag_approx2_correct):
    print("Hessian Diag Approx2")
  if (taylor_2nd_correct):
    print("Taylor 2nd")
