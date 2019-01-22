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
    parser.add_argument('--arch', action='store', default='/home/kaveena/compare-pruning-signals/caffe-pruned-models/LeNet-5-CIFAR10/masked-one-saliency',
            help='the caffe solver to use')
    parser.add_argument('--pretrained', action='store', default='/home/kaveena/compare-pruning-signals/caffe-pruned-models/LeNet-5-CIFAR10/original.caffemodel',
            help='pretrained model')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu')
    return parser

args = parser().parse_args()
if args.gpu:
  caffe.set_mode_gpu()
else:
  caffe.set_mode_cpu()

if args.arch is None:
  print("Caffe solver needed")
  exit(1)
#  Allocate two models:  one that cumputes saliencies (weights + masks + saliencies)
#                        one for training (weights + masks)
#  saliency_solver = caffe.SGDSolver(args.arch)
#  saliency_solver.net.copy_from(args.pretrained)
#  saliency_solver.test_nets[0].copy_from(args.pretrained)
#  net = saliency_solver.test_nets[0]
net = caffe.Net(args.arch+'.prototxt', args.pretrained, caffe.TEST)

convolution_list = list(filter(lambda x: 'Convolution' in net.layer_dict[x].type, net.layer_dict.keys()))
named_modules = net.layer_dict

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
    named_modules[k].saliency_ = saliency
    named_modules[k].blobs[saliency_pos_].data.fill(0)

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
     fisher_caffe[k] = named_modules[k].blobs[saliency_pos_].data[0]
     fisher[k] = ((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num ).sum(axis=(2,3)) ** 2).sum(axis=0) / (2*net.blobs[k].num)
     fisher_correct*=np.allclose(fisher_caffe[k], fisher[k])

#   print("Check 1st order Taylor")
  for k in convolution_list:
     taylor_caffe[k] = named_modules[k].blobs[saliency_pos_].data[0]
     taylor[k] = (-1 * net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num).sum(axis=(0,2,3)) / (net.blobs[k].num)
     taylor_correct*=np.allclose(taylor_caffe[k], taylor[k])

#   print("Check Hessian Diag")
  for k in convolution_list:
     hessian_diag_caffe[k] = named_modules[k].blobs[saliency_pos_].data[0]
     hessian_diag[k] = ((net.blobs[k].data**2) * net.blobs[k].ddiff).sum(axis=(0,2,3)) / (2*net.blobs[k].num)
     hessian_diag_correct*=np.allclose(hessian_diag_caffe[k], hessian_diag[k])

#   print("Check Hessian Diag Approx 2")
  for k in convolution_list:
     hessian_diag_approx2_caffe[k] = named_modules[k].blobs[saliency_pos_].data[0]
     hessian_diag_approx2[k] = ((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)**2).sum(axis=(0,2,3)) / (2*net.blobs[k].num)
     hessian_diag_approx2_correct*=(np.allclose(hessian_diag_approx2_caffe[k], hessian_diag_approx2[k]))

#   print("Check Taylor 2nd")
  for k in convolution_list:
     taylor_2nd_caffe[k] = named_modules[k].blobs[saliency_pos_].data[0]
     taylor_2nd[k] = ((-1*net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)  + ((net.blobs[k].data**2 * net.blobs[k].ddiff))/2).sum(axis=(0,2,3)) / (net.blobs[k].num)
     taylor_2nd_correct*=(np.allclose(taylor_2nd_caffe[k], taylor_2nd[k]))

#   print("Check Taylor 2nd Approx2")
  for k in convolution_list:
     taylor_2nd_approx2_caffe[k] = named_modules[k].blobs[saliency_pos_].data[0]
     taylor_2nd_approx2[k] = ((-1*net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)  + (((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)**2)/2)).sum(axis=(0,2,3)) / (net.blobs[k].num)
     taylor_2nd_approx2_correct*=(np.allclose(taylor_2nd_approx2_caffe[k], taylor_2nd_approx2[k]))

  for k in convolution_list:
    weight_avg_caffe[k] = named_modules[k].blobs[named_modules[k].saliency_pos_].data[0]
    weight_avg[k] = net.blobs[k].data.sum(axis=(0,2,3)) / (net.blobs[k].num)
    weight_avg_correct*=(np.allclose(weight_avg_caffe[k], weight_avg[k]))

  for k in convolution_list:
    diff_avg_caffe[k] = named_modules[k].blobs[named_modules[k].saliency_pos_].data[0]
    diff_avg[k] = net.blobs[k].diff.sum(axis=(0,2,3))
    diff_avg_correct*=(np.allclose(diff_avg_caffe[k], diff_avg[k]))
  
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
for k in convolution_list:
  named_modules[k].saliency_input_ = caffe._caffe.SALIENCY_INPUT.WEIGHT

for saliency in caffe._caffe.SALIENCY.names.values():
  if (saliency == caffe._caffe.SALIENCY.ALL):
    continue
  for k in convolution_list:
    named_modules[k].saliency_ = saliency
    named_modules[k].blobs[saliency_pos_].data.fill(0)
    named_modules[k].blobs[0].diff.fill(0)
    named_modules[k].blobs[1].diff.fill(0)
    named_modules[k].blobs[0].ddiff.fill(0)
    named_modules[k].blobs[1].ddiff.fill(0)

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
     fisher_caffe[k] = named_modules[k].blobs[saliency_pos_].data[0]
     fisher[k] = ((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num ).sum(axis=(2,3)) ** 2).sum(axis=0) / (2*net.blobs[k].num)
     fisher_correct*=np.allclose(fisher_caffe[k], fisher[k])

#   print("Check 1st order Taylor")
  for k in convolution_list:
     taylor_caffe[k] = named_modules[k].blobs[saliency_pos_].data[0]
     taylor[k] = (-1 * named_modules[k].blobs[0].data * named_modules[k].blobs[0].diff * net.blobs[k].num).sum(axis=(1,2,3)) / (net.blobs[k].num) + (-1 * named_modules[k].blobs[1].data * named_modules[k].blobs[1].diff )
     taylor_correct*=np.allclose(taylor_caffe[k], taylor[k])

#   print("Check Hessian Diag")
  for k in convolution_list:
     hessian_diag_caffe[k] = named_modules[k].blobs[saliency_pos_].data[0]
     hessian_diag[k] = (((named_modules[k].blobs[0].data**2) * named_modules[k].blobs[0].ddiff).sum(axis=(1,2,3))  + (named_modules[k].blobs[1].data * named_modules[k].blobs[1].ddiff))/ (2*net.blobs[k].num)
     hessian_diag_correct*=np.allclose(hessian_diag_caffe[k], hessian_diag[k])

#   print("Check Hessian Diag Approx 2")
  for k in convolution_list:
     hessian_diag_approx2_caffe[k] = named_modules[k].blobs[saliency_pos_].data[0]
     hessian_diag_approx2[k] = ((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)**2).sum(axis=(0,2,3)) / (2*net.blobs[k].num)
     hessian_diag_approx2_correct*=(np.allclose(hessian_diag_approx2_caffe[k], hessian_diag_approx2[k]))

#   print("Check Taylor 2nd")
  for k in convolution_list:
     taylor_2nd_caffe[k] = named_modules[k].blobs[saliency_pos_].data[0]
     taylor_2nd[k] = (-1 * named_modules[k].blobs[0].data * named_modules[k].blobs[0].diff * net.blobs[k].num).sum(axis=(1,2,3)) / (net.blobs[k].num) + (-1 * named_modules[k].blobs[1].data * named_modules[k].blobs[1].diff) + (named_modules[k].blobs[0].data**2 * named_modules[k].blobs[0].ddiff/2).sum(axis=(1,2,3)) / (net.blobs[k].num) + (named_modules[k].blobs[1].data**2 * named_modules[k].blobs[1].ddiff/2 ) / (net.blobs[k].num)
     taylor_2nd_correct*=(np.allclose(taylor_2nd_caffe[k], taylor_2nd[k], rtol=1e-4))

#   print("Check Taylor 2nd Approx2")
  for k in convolution_list:
     taylor_2nd_approx2_caffe[k] = named_modules[k].blobs[saliency_pos_].data[0]
     taylor_2nd_approx2[k] = ((-1*net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)  + (((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)**2)/2)).sum(axis=(0,2,3)) / (net.blobs[k].num)
     taylor_2nd_approx2_correct*=(np.allclose(taylor_2nd_approx2_caffe[k], taylor_2nd_approx2[k]))

  for k in convolution_list:
    weight_avg_caffe[k] = named_modules[k].blobs[named_modules[k].saliency_pos_].data[0]
    weight_avg[k] = named_modules[k].blobs[0].data.sum(axis=(1,2,3)) + named_modules[k].blobs[1].data
    weight_avg_correct*=(np.allclose(weight_avg_caffe[k], weight_avg[k]))

  for k in convolution_list:
    diff_avg_caffe[k] = named_modules[k].blobs[named_modules[k].saliency_pos_].data[0]
    diff_avg[k] = named_modules[k].blobs[0].diff.sum(axis=(1,2,3)) + named_modules[k].blobs[1].diff
    diff_avg_correct*=(np.allclose(diff_avg_caffe[k], diff_avg[k], rtol=1e-4))
  
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
  
"""  
for k in convolution_list:
  named_modules[k].blobs[saliency_pos_].data.fill(0)
  named_modules[k].saliency_norm_ = caffe._caffe.NORM.L1

# compute saliency    
evalset_size = 1;
for iter in range(evalset_size):
  net.forward()
  net.backward()
print("L1 NORM")
print("Check Fisher Info")
for k in convolution_list:
   fisher_caffe[k] = named_modules[k].blobs[saliency_pos_].data[0]
   fisher[k] = ((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num ).sum(axis=(2,3)) ** 2).sum(axis=0) / (2*net.blobs[k].num)
   print(np.allclose(fisher_caffe[k], fisher[k]))

print("Check 1st order Taylor")
for k in convolution_list:
   taylor_caffe[k] = named_modules[k].blobs[saliency_pos_].data[1]
   taylor[k] = np.abs((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)).sum(axis=(0,2,3)) / (net.blobs[k].num)
   print(np.allclose(taylor_caffe[k], taylor[k]))

print("Check Hessian Diag")
for k in convolution_list:
   hessian_diag_caffe[k] = named_modules[k].blobs[saliency_pos_].data[2]
   hessian_diag[k] = np.abs((net.blobs[k].data**2) * net.blobs[k].ddiff/2).sum(axis=(0,2,3)) / (net.blobs[k].num)
   print(np.allclose(hessian_diag_caffe[k], hessian_diag[k]))

print("Check Hessian Diag Approx 2")
for k in convolution_list:
   hessian_diag_approx2_caffe[k] = named_modules[k].blobs[saliency_pos_].data[3]
   hessian_diag_approx2[k] = np.abs((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)**2 / 2).sum(axis=(0,2,3)) / (net.blobs[k].num)
   print(np.allclose(hessian_diag_approx2_caffe[k], hessian_diag_approx2[k]))

taylor_2nd = dict()
taylor_2nd_caffe = dict()

print("Check Taylor 2nd")
for k in convolution_list:
   taylor_2nd_caffe[k] = named_modules[k].blobs[saliency_pos_].data[4]
   taylor_2nd[k] = np.abs((-1*net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)  + ((net.blobs[k].data**2 * net.blobs[k].ddiff))/2).sum(axis=(0,2,3)) / (net.blobs[k].num)
   print(np.allclose(taylor_2nd_caffe[k], taylor_2nd[k]))

taylor_2nd_approx2 = dict()
taylor_2nd_approx2_caffe = dict()

print("Check Taylor 2nd Approx2")
for k in convolution_list:
   taylor_2nd_approx2_caffe[k] = named_modules[k].blobs[saliency_pos_].data[5]
   taylor_2nd_approx2[k] = np.abs((-1*net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)  + (((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)**2)/2)).sum(axis=(0,2,3)) / (net.blobs[k].num)
   print(np.allclose(taylor_2nd_approx2_caffe[k], taylor_2nd_approx2[k]))

net = caffe.Net(args.arch+'-l2.prototxt', args.pretrained, caffe.TEST)

named_modules = net.layer_dict

for k in convolution_list:
  named_modules[k].blobs[saliency_pos_].data.fill(0)

# compute saliency    
evalset_size = 1;
for iter in range(evalset_size):
  net.forward()
  net.backward()
print("L2 NORM")
print("Check Fisher Info")
for k in convolution_list:
   fisher_caffe[k] = named_modules[k].blobs[saliency_pos_].data[0]
   fisher[k] = ((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num ).sum(axis=(2,3)) ** 2).sum(axis=0) / (2*net.blobs[k].num)
   print(np.allclose(fisher_caffe[k], fisher[k]))

print("Check 1st order Taylor")
for k in convolution_list:
   taylor_caffe[k] = named_modules[k].blobs[saliency_pos_].data[1]
   taylor[k] = np.power((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num), 2).sum(axis=(0,2,3)) / (net.blobs[k].num)
   print(np.allclose(taylor_caffe[k], taylor[k]))

print("Check Hessian Diag")
for k in convolution_list:
   hessian_diag_caffe[k] = named_modules[k].blobs[saliency_pos_].data[2]
   hessian_diag[k] = np.power((net.blobs[k].data**2) * net.blobs[k].ddiff/2, 2).sum(axis=(0,2,3)) / (net.blobs[k].num)
   print(np.allclose(hessian_diag_caffe[k], hessian_diag[k]))

print("Check Hessian Diag Approx 2")
for k in convolution_list:
   hessian_diag_approx2_caffe[k] = named_modules[k].blobs[saliency_pos_].data[3]
   hessian_diag_approx2[k] = np.power((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)**2 / 2, 2).sum(axis=(0,2,3)) / (net.blobs[k].num)
   print(np.allclose(hessian_diag_approx2_caffe[k], hessian_diag_approx2[k]))

taylor_2nd = dict()
taylor_2nd_caffe = dict()

print("Check Taylor 2nd")
for k in convolution_list:
   taylor_2nd_caffe[k] = named_modules[k].blobs[saliency_pos_].data[4]
   taylor_2nd[k] = np.power((-1*net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)  + ((net.blobs[k].data**2 * net.blobs[k].ddiff))/2, 2).sum(axis=(0,2,3)) / (net.blobs[k].num)
   print(np.allclose(taylor_2nd_caffe[k], taylor_2nd[k]))

taylor_2nd_approx2 = dict()
taylor_2nd_approx2_caffe = dict()

print("Check Taylor 2nd Approx2")
for k in convolution_list:
   taylor_2nd_approx2_caffe[k] = named_modules[k].blobs[saliency_pos_].data[5]
   taylor_2nd_approx2[k] = np.power((-1*net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)  + (((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)**2)/2), 2).sum(axis=(0,2,3)) / (net.blobs[k].num)
   print(np.allclose(taylor_2nd_approx2_caffe[k], taylor_2nd_approx2[k]))

"""
