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
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu')
    parser.add_argument('--one', action='store_true', default=False, help='use one image for testing')
    parser.add_argument('--two', action='store_true', default=False, help='use one image for testing')
    return parser

args = parser().parse_args()

if args.one:
  prototxt = '/home/kaveena/compare-pruning-signals/caffe-pruned-models/' + args.arch + '/one-image.prototxt' 
elif args.two:
  prototxt = '/home/kaveena/compare-pruning-signals/caffe-pruned-models/' + args.arch + '/2-images.prototxt' 
else:
  prototxt = '/home/kaveena/compare-pruning-signals/caffe-pruned-models/' + args.arch + '/128-images.prototxt' 
caffemodel = '/home/kaveena/compare-pruning-signals/caffe-pruned-models/' + args.arch + '/original.caffemodel' 
if args.gpu:
  caffe.set_mode_gpu()
else:
  caffe.set_mode_cpu()

if args.arch is None:
  print("Caffe solver needed")
  exit(1)
  
#Allocate two models:  one that cumputes saliencies (weights + masks + saliencies)
#                      one for training (weights + masks)
#saliency_solver = caffe.SGDSolver(args.arch)
#saliency_solver.net.copy_from(args.pretrained)
#saliency_solver.test_nets[0].copy_from(args.pretrained)
#net = saliency_solver.test_nets[0]

net = caffe.Net(prototxt, caffemodel, caffe.TEST)

convolution_list = list(filter(lambda x: 'Convolution' in net.layer_dict[x].type, net.layer_dict.keys()))
named_modules = net.layer_dict
  
for k in convolution_list:
  layer = named_modules[k]
  layer.saliency_output_channel_compute_ = False
  layer.saliency_input_channel_compute_ = True
  input_layer = net.bottom_names[k][0] 
  named_modules[k].group = int(net.blobs[input_layer].channels / named_modules[k].blobs[0].channels) 

net.reshape()

fisher_caffe = dict()
fisher = dict()
taylor = dict()
taylor_caffe = dict()
hessian_diag = dict()
hessian_diag_caffe = dict()
hessian_diag_approx2 = dict()
hessian_diag_approx2_caffe = dict()
taylor_2nd = dict()
taylor_2nd_caffe = dict()
taylor_2nd_approx2 = dict()
taylor_2nd_approx2_caffe = dict()
weight_avg = dict()
weight_avg_caffe = dict()
diff_avg = dict()
diff_avg_caffe = dict()

for saliency in caffe._caffe.SALIENCY.names.values():
  if (saliency == caffe._caffe.SALIENCY.ALL):
    continue
  for k in convolution_list:
    layer = named_modules[k]
    layer.saliency_ = saliency
    layer.blobs[layer.saliency_pos_].data.fill(0)
    layer.blobs[layer.saliency_pos_+1].data.fill(0)

  net.clear_param_diffs();

  fisher_correct = True
  taylor_correct = True
  hessian_diag_correct = True
  hessian_diag_approx2_correct = True
  taylor_2nd_correct = True
  taylor_2nd_approx2_correct = True
  weight_avg_correct = True
  diff_avg_correct = True

  # compute saliency    
  evalset_size = 1;
  for iter in range(evalset_size):
    net.forward()
    net.backward()

#   print("Check Fisher Info")
  for k in convolution_list:
    layer = named_modules[k]
    input_blob = net.bottom_names[k][0] 
    fisher_caffe[k] = named_modules[k].blobs[saliency_pos_+1].data[0]
    fisher[k] = ((net.blobs[input_blob].data * net.blobs[input_blob].diff * net.blobs[input_blob].num ).sum(axis=(2,3)) ** 2).sum(axis=0) / (2*net.blobs[input_blob].num)
    fisher[k] = (fisher[k].reshape((named_modules[k].group, -1)).sum(axis=0))
    fisher_correct*=np.allclose(fisher_caffe[k], fisher[k])

#   print("Check 1st order Taylor")
  for k in convolution_list:
    layer = named_modules[k]
    input_blob = net.bottom_names[k][0] 
    taylor_caffe[k] = named_modules[k].blobs[saliency_pos_+1].data[0]
    taylor[k] = (-1 * net.blobs[input_blob].data * net.blobs[input_blob].diff * net.blobs[input_blob].num).sum(axis=(0,2,3)).reshape((named_modules[k].group, -1)).sum(axis=0) / (net.blobs[input_blob].num)
    taylor_correct*=np.allclose(taylor_caffe[k], taylor[k])

#   print("Check Hessian Diag")
  for k in convolution_list:
    layer = named_modules[k]
    input_blob = net.bottom_names[k][0] 
    hessian_diag_caffe[k] = named_modules[k].blobs[saliency_pos_+1].data[0]
    hessian_diag[k] = ((net.blobs[input_blob].data**2) * net.blobs[input_blob].ddiff).sum(axis=(0,2,3)).reshape((named_modules[k].group, -1)).sum(axis=0) / (2*net.blobs[input_blob].num)
    hessian_diag_correct*=np.allclose(hessian_diag_caffe[k], hessian_diag[k])

#   print("Check Hessian Diag Approx 2")
  for k in convolution_list:
    layer = named_modules[k]
    input_blob = net.bottom_names[k][0] 
    hessian_diag_approx2_caffe[k] = named_modules[k].blobs[saliency_pos_+1].data[0]
    hessian_diag_approx2[k] = ((net.blobs[input_blob].data * net.blobs[input_blob].diff * net.blobs[input_blob].num)**2).sum(axis=(0,2,3)).reshape((named_modules[k].group, -1)).sum(axis=0) / (2*net.blobs[input_blob].num)
    hessian_diag_approx2_correct*=(np.allclose(hessian_diag_approx2_caffe[k], hessian_diag_approx2[k]))

#   print("Check Taylor 2nd")
  for k in convolution_list:
    layer = named_modules[k]
    input_blob = net.bottom_names[k][0] 
    taylor_2nd_caffe[k] = named_modules[k].blobs[saliency_pos_+1].data[0]
    taylor_2nd[k] = ((-1*net.blobs[input_blob].data * net.blobs[input_blob].diff * net.blobs[input_blob].num)  + ((net.blobs[input_blob].data**2 * net.blobs[input_blob].ddiff))/2).sum(axis=(0,2,3)).reshape((named_modules[k].group, -1)).sum(axis=0) / (net.blobs[input_blob].num)
    taylor_2nd_correct*=(np.allclose(taylor_2nd_caffe[k], taylor_2nd[k]))

#   print("Check Taylor 2nd Approx2")
  for k in convolution_list:
    layer = named_modules[k]
    input_blob = net.bottom_names[k][0] 
    taylor_2nd_approx2_caffe[k] = named_modules[k].blobs[saliency_pos_+1].data[0]
    taylor_2nd_approx2[k] = ((-1*net.blobs[input_blob].data * net.blobs[input_blob].diff * net.blobs[input_blob].num)  + (((net.blobs[input_blob].data * net.blobs[input_blob].diff * net.blobs[input_blob].num)**2)/2)).sum(axis=(0,2,3)).reshape((named_modules[k].group, -1)).sum(axis=0) / (net.blobs[input_blob].num)
    taylor_2nd_approx2_correct*=(np.allclose(taylor_2nd_approx2_caffe[k], taylor_2nd_approx2[k]))

  for k in convolution_list:
    layer = named_modules[k]
    input_blob = net.bottom_names[k][0] 
    weight_avg_caffe[k] = named_modules[k].blobs[named_modules[k].saliency_pos_+1].data[0]
    weight_avg[k] = net.blobs[input_blob].data.sum(axis=(0,2,3)).reshape((named_modules[k].group, -1)).sum(axis=0) / (net.blobs[input_blob].num)
    weight_avg_correct*=(np.allclose(weight_avg_caffe[k], weight_avg[k], rtol=1e-3))

  for k in convolution_list:
    layer = named_modules[k]
    input_blob = net.bottom_names[k][0] 
    diff_avg_caffe[k] = named_modules[k].blobs[named_modules[k].saliency_pos_+1].data[0]
    diff_avg[k] = net.blobs[input_blob].diff.sum(axis=(0,2,3)).reshape((named_modules[k].group, -1)).sum(axis=0)
    diff_avg_correct*=(np.allclose(diff_avg_caffe[k], diff_avg[k], rtol=1e-3))
  
  print(saliency)
  if (fisher_correct):
    print("Fisher")
  if (taylor_correct):
    print("Taylor")
  if (hessian_diag_correct):
    print("Hessian Diag")
  if (hessian_diag_approx2_correct):
    print("Hessian Diag Approx2")
  if (taylor_2nd_correct):
    print("Taylor 2nd")
  if (taylor_2nd_approx2_correct):
    print("Taylor 2nd Approx2")
  if (weight_avg_correct):
    print("Weight Avg")
  if (diff_avg_correct):
    print("Diff Avg")

# Weight based 
print(convolution_list)
for k in convolution_list:
  named_modules[k].saliency_input_ = caffe._caffe.SALIENCY_INPUT.WEIGHT

for saliency in caffe._caffe.SALIENCY.names.values():
  if (saliency == caffe._caffe.SALIENCY.ALL):
    continue
  for k in convolution_list:
    named_modules[k].saliency_ = saliency
    named_modules[k].blobs[saliency_pos_].data.fill(0)
    named_modules[k].blobs[saliency_pos_+1].data.fill(0)
  
  net.clear_param_diffs();

  fisher_correct = True
  taylor_correct = True
  hessian_diag_correct = True
  hessian_diag_approx2_correct = True
  taylor_2nd_correct = True
  taylor_2nd_approx2_correct = True
  weight_avg_correct = True
  diff_avg_correct = True

  # compute saliency    
  evalset_size = 1;
  for iter in range(evalset_size):
    net.forward()
    net.backward()

#   print("Check Fisher Info")
  for k in convolution_list:
    fisher_caffe[k] = named_modules[k].blobs[saliency_pos_+1].data[0]
    fisher[k] = (((named_modules[k].blobs[0].data * named_modules[k].blobs[0].diff * net.blobs[k].num).sum(axis=(0,2,3)))**2) / (2*net.blobs[k].num)
    fisher_correct*=np.allclose(fisher_caffe[k], fisher[k])

#   print("Check 1st order Taylor")
  for k in convolution_list:
    taylor_caffe[k] = named_modules[k].blobs[saliency_pos_+1].data[0]
    taylor[k] = (-1 * named_modules[k].blobs[0].data * named_modules[k].blobs[0].diff ).sum(axis=(0,2,3)) 
    taylor_correct*=np.allclose(taylor_caffe[k], taylor[k], rtol=1e-1)

#   print("Check Hessian Diag")
  for k in convolution_list:
    hessian_diag_caffe[k] = named_modules[k].blobs[saliency_pos_+1].data[0]
    hessian_diag[k] = (((named_modules[k].blobs[0].data**2) * named_modules[k].blobs[0].ddiff).sum(axis=(0,2,3)))/ (2*net.blobs[k].num)
    hessian_diag_correct*=np.allclose(hessian_diag_caffe[k], hessian_diag[k], rtol=1e-4)

#   print("Check Hessian Diag Approx 2")
  for k in convolution_list:
    hessian_diag_approx2_caffe[k] = named_modules[k].blobs[saliency_pos_+1].data[0]
    hessian_diag_approx2[k] = ((named_modules[k].blobs[0].data * named_modules[k].blobs[0].diff * net.blobs[k].num)**2).sum(axis=(0,2,3)) / (2*net.blobs[k].num)
    hessian_diag_approx2_correct*=(np.allclose(hessian_diag_approx2_caffe[k], hessian_diag_approx2[k], rtol=1e-4))

#   print("Check Taylor 2nd")
  for k in convolution_list:
    taylor_2nd_caffe[k] = named_modules[k].blobs[saliency_pos_+1].data[0]
    taylor_2nd[k] = (-1 * named_modules[k].blobs[0].data * named_modules[k].blobs[0].diff * net.blobs[k].num).sum(axis=(0,2,3)) / (net.blobs[k].num) + (named_modules[k].blobs[0].data**2 * named_modules[k].blobs[0].ddiff/2).sum(axis=(0,2,3)) / (net.blobs[k].num) 
    taylor_2nd_correct*=(np.allclose(taylor_2nd_caffe[k], taylor_2nd[k], rtol=1e-4))

#   print("Check Taylor 2nd Approx2")
  for k in convolution_list:
    taylor_2nd_approx2_caffe[k] = named_modules[k].blobs[saliency_pos_+1].data[0]
    taylor_2nd_approx2[k] = ((-1*named_modules[k].blobs[0].data * named_modules[k].blobs[0].diff * net.blobs[k].num)  + (((named_modules[k].blobs[0].data * named_modules[k].blobs[0].diff * net.blobs[k].num)**2)/2)).sum(axis=(0,2,3)) / (net.blobs[k].num)
    taylor_2nd_approx2_correct*=(np.allclose(taylor_2nd_approx2_caffe[k], taylor_2nd_approx2[k]))

  for k in convolution_list:
    weight_avg_caffe[k] = named_modules[k].blobs[named_modules[k].saliency_pos_+1].data[0]
    weight_avg[k] = named_modules[k].blobs[0].data.sum(axis=(0,2,3))
    weight_avg_correct*=(np.allclose(weight_avg_caffe[k], weight_avg[k], rtol=1e-1))

  for k in convolution_list:
    diff_avg_caffe[k] = named_modules[k].blobs[named_modules[k].saliency_pos_+1].data[0]
    diff_avg[k] = named_modules[k].blobs[0].diff.sum(axis=(0,2,3))
    diff_avg_correct*=(np.allclose(diff_avg_caffe[k], diff_avg[k], rtol=1e-1))
  
  print(saliency)
  if (fisher_correct):
    print("Fisher")
  if (taylor_correct):
    print("Taylor")
  if (hessian_diag_correct):
    print("Hessian Diag")
  if (hessian_diag_approx2_correct):
    print("Hessian Diag Approx2")
  if (taylor_2nd_correct):
    print("Taylor 2nd")
  if (taylor_2nd_approx2_correct):
    print("Taylor 2nd Approx2")
  if (weight_avg_correct):
    print("Weight Avg")
  if (diff_avg_correct):
    print("Diff Avg")
