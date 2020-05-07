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
    parser.add_argument('--arch', action='store', default='LeNet-5',
            help='the network architecture to use')
    parser.add_argument('--dataset', action='store', default='CIFAR10',
            help='the dataset to use')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu')
    parser.add_argument('--batch-size', type=int, default=128, help='number of images in each batch')
    return parser

args = parser().parse_args()

if args.gpu:
  caffe.set_mode_gpu()

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
os.system("cat " + original_image_source + " | head -n " + str(args.batch_size) + " > " + small_images_filename)

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
apoz = dict()
apoz_caffe = dict()

# Create a prototxt that uses the validation set and has saliency computation

for pointwise_saliency in caffe_pb2.ConvolutionSaliencyParameter.SALIENCY.keys():
  saliency_prototxt = get_prototxt_from_file(original_prototxt_filename)
  add_mask_to_prototxt(saliency_prototxt)
  for l in saliency_prototxt.layer:
    if l.type == 'ImageData':
      l.image_data_param.source = small_images_filename
      l.image_data_param.shuffle = False
      l.image_data_param.batch_size = args.batch_size
      l.transform_param.mirror = False
  add_saliency_to_prototxt(saliency_prototxt, [pointwise_saliency], ['ACTIVATION'], ['NONE'])
  saliency_prototxt.compute_2nd_derivative = True

  save_prototxt_to_file(saliency_prototxt, saliency_prototxt_filename)

  caffe._caffe.set_allocate_2nd_derivative_memory(True)
  net = caffe.Net(saliency_prototxt_filename, pretrained, caffe.TEST)
  pruning_net = PruningGraph(net, saliency_prototxt)

  named_modules = net.layer_dict

# compute saliency    
  evalset_size = 1;
  pruning_net.ClearSaliencyBlobs()
  net.clear_param_diffs();

  fisher_correct = True
  taylor_correct = True
  hessian_diag_correct = True
  hessian_diag_approx2_correct = True
  taylor_2nd_correct = True
  taylor_2nd_approx2_correct = True
  weight_avg_correct = True
  diff_avg_correct = True
  apoz_correct = True

  # compute saliency    
  for iter in range(evalset_size):
    net.forward()
    net.backward()

#   print("Check Fisher Info")
  for k in pruning_net.convolution_list:
     fisher_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
     fisher[k] = ((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num ).sum(axis=(2,3)) ** 2).sum(axis=0) / (2*net.blobs[k].num)
     fisher_correct*=np.allclose(fisher_caffe[k], fisher[k])

#   print("Check 1st order Taylor")
  for k in pruning_net.convolution_list:
     taylor_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
     taylor[k] = (-1 * net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num).sum(axis=(0,2,3)) / (net.blobs[k].num)
     taylor_correct*=np.allclose(taylor_caffe[k], taylor[k])

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

#   print("Check Taylor 2nd Approx2")
  for k in pruning_net.convolution_list:
    taylor_2nd_approx2_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
    taylor_2nd_approx2[k] = ((-1*net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)  + (((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)**2)/2)).sum(axis=(0,2,3)) / (net.blobs[k].num)
    taylor_2nd_approx2_correct*=(np.allclose(taylor_2nd_approx2_caffe[k], taylor_2nd_approx2[k]))

  for k in pruning_net.convolution_list:
    weight_avg_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
    weight_avg[k] = net.blobs[k].data.sum(axis=(0,2,3)) / (net.blobs[k].num)
    weight_avg_correct*=(np.allclose(weight_avg_caffe[k], weight_avg[k]))

  for k in pruning_net.convolution_list:
    diff_avg_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
    diff_avg[k] = net.blobs[k].diff.sum(axis=(0,2,3))
    diff_avg_correct*=(np.allclose(diff_avg_caffe[k], diff_avg[k], rtol=1e-3))

  for k in pruning_net.convolution_list:
    apoz_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
    apoz[k] = (net.blobs[k].data > 0).sum(axis=(0,2,3)) / (net.blobs[k].num)
    apoz_correct*=(np.allclose(apoz_caffe[k], apoz[k]))

  if (pointwise_saliency=='TAYLOR'):
    print("TAYLOR OK" if taylor_correct else "TAYLOR KO")
  if (pointwise_saliency=='HESSIAN_DIAG_APPROX1'):
    print("HESSIAN_DIAG OK" if hessian_diag_correct else "HESSIAN_DIAG KO")
  if (pointwise_saliency=='HESSIAN_DIAG_APPROX2'):
    print("HESSIAN_DIAG_APPROX2 OK" if hessian_diag_approx2_correct else "HESSIAN_DIAG_APPROX2 KO")
  if (pointwise_saliency=='TAYLOR_2ND'):
    print("TAYLOR_2ND OK" if taylor_2nd_correct else "TAYLOR_2ND KO")
  if (pointwise_saliency=='TAYLOR_2ND_APPROX2'):
    print("TAYLOR_2ND_APPROX2 OK" if taylor_2nd_approx2_correct else "TAYLOR_2ND_APPROX2 KO")
  if (pointwise_saliency=='AVERAGE_INPUT'):
    print("WEIGHT_AVG OK" if weight_avg_correct else "WEIGHT_AVG KO")
  if (pointwise_saliency=='AVERAGE_GRADIENT'):
    print("DIFF_AVG OK" if diff_avg_correct else "DIFF_AVG KO")
  if (pointwise_saliency=='APOZ'):
    print("APOZ OK" if apoz_correct else "APOZ KO")

# Weight based 
for pointwise_saliency in caffe_pb2.ConvolutionSaliencyParameter.SALIENCY.keys():
  saliency_prototxt = get_prototxt_from_file(original_prototxt_filename)
  add_mask_to_prototxt(saliency_prototxt)
  for l in saliency_prototxt.layer:
    if l.type == 'ImageData':
      l.image_data_param.source = small_images_filename
      l.image_data_param.shuffle = False
      l.image_data_param.batch_size = args.batch_size
      l.transform_param.mirror = False
  add_saliency_to_prototxt(saliency_prototxt, [pointwise_saliency], ['WEIGHT'], ['NONE'])
  saliency_prototxt.compute_2nd_derivative = True

  save_prototxt_to_file(saliency_prototxt, saliency_prototxt_filename)

  caffe._caffe.set_allocate_2nd_derivative_memory(True)
  net = caffe.Net(saliency_prototxt_filename, pretrained, caffe.TEST)
  pruning_net = PruningGraph(net, saliency_prototxt)

  named_modules = net.layer_dict

# compute saliency    
  evalset_size = 1;
  pruning_net.ClearSaliencyBlobs()
  net.clear_param_diffs();

  fisher_correct = True
  taylor_correct = True
  hessian_diag_correct = True
  hessian_diag_approx2_correct = True
  taylor_2nd_correct = True
  taylor_2nd_approx2_correct = True
  weight_avg_correct = True
  diff_avg_correct = True
  apoz_correct = True

  # compute saliency    
  for iter in range(evalset_size):
    net.forward()
    net.backward()

#   print("Check 1st order Taylor")
  for k in pruning_net.convolution_list:
    taylor_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
    taylor[k] = (-1 * named_modules[k].blobs[0].data * named_modules[k].blobs[0].diff * net.blobs[k].num).sum(axis=(1,2,3)) / (net.blobs[k].num) + (-1 * named_modules[k].blobs[1].data * named_modules[k].blobs[1].diff )
    taylor_correct*=np.allclose(taylor_caffe[k], taylor[k])

#   print("Check Hessian Diag")
  for k in pruning_net.convolution_list:
    hessian_diag_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
    hessian_diag[k] = (((named_modules[k].blobs[0].data**2) * named_modules[k].blobs[0].ddiff).sum(axis=(1,2,3))  + (named_modules[k].blobs[1].data * named_modules[k].blobs[1].ddiff))/ (2*net.blobs[k].num)
    hessian_diag_correct*=np.allclose(hessian_diag_caffe[k], hessian_diag[k])

#   print("Check Hessian Diag Approx 2")
  for k in pruning_net.convolution_list:
    hessian_diag_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
    hessian_diag_approx2[k] = ((net.blobs[k].data * net.blobs[k].diff * net.blobs[k].num)**2).sum(axis=(0,2,3)) / (2*net.blobs[k].num)
    hessian_diag_approx2[k] = (((named_modules[k].blobs[0].data * named_modules[k].blobs[0].diff * net.blobs[k].num)**2).sum(axis=(1,2,3)) / (2*net.blobs[k].num)) + (((named_modules[k].blobs[1].data * named_modules[k].blobs[1].diff * net.blobs[k].num)**2) / (2*net.blobs[k].num))
    hessian_diag_approx2_correct*=(np.allclose(hessian_diag_approx2_caffe[k], hessian_diag_approx2[k]))

#   print("Check Taylor 2nd")
  for k in pruning_net.convolution_list:
    taylor_2nd_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
    taylor_2nd[k] = (-1 * named_modules[k].blobs[0].data * named_modules[k].blobs[0].diff * net.blobs[k].num).sum(axis=(1,2,3)) / (net.blobs[k].num) + (-1 * named_modules[k].blobs[1].data * named_modules[k].blobs[1].diff) + (named_modules[k].blobs[0].data**2 * named_modules[k].blobs[0].ddiff/2).sum(axis=(1,2,3)) / (net.blobs[k].num) + (named_modules[k].blobs[1].data**2 * named_modules[k].blobs[1].ddiff/2 ) / (net.blobs[k].num)
    taylor_2nd_correct*=(np.allclose(taylor_2nd_caffe[k], taylor_2nd[k], rtol=1e-4))

#   print("Check Taylor 2nd Approx2")
  for k in pruning_net.convolution_list:
    taylor_2nd_approx2_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
    adEda = named_modules[k].blobs[0].data * named_modules[k].blobs[0].diff
    bdEdb = named_modules[k].blobs[1].data * named_modules[k].blobs[1].diff
    taylor_2nd_approx2[k] = (((-1*adEda * net.blobs[k].num)  + (((adEda * net.blobs[k].num)**2)/2)).sum(axis=(1,2,3)) / (net.blobs[k].num)) + (((-1*bdEdb *net.blobs[k].num)+(((bdEdb * net.blobs[k].num)**2)/2))/net.blobs[k].num)
    taylor_2nd_approx2_correct*=(np.allclose(taylor_2nd_approx2_caffe[k], taylor_2nd_approx2[k]))

  for k in pruning_net.convolution_list:
    weight_avg_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
    weight_avg[k] = named_modules[k].blobs[0].data.sum(axis=(1,2,3)) + named_modules[k].blobs[1].data
    weight_avg_correct*=(np.allclose(weight_avg_caffe[k], weight_avg[k], rtol=1e-3))

  for k in pruning_net.convolution_list:
    diff_avg_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
    diff_avg[k] = named_modules[k].blobs[0].diff.sum(axis=(1,2,3)) + named_modules[k].blobs[1].diff
    diff_avg_correct*=(np.allclose(diff_avg_caffe[k], diff_avg[k], rtol=1e-3))

  for k in pruning_net.convolution_list:
    apoz_caffe[k] = pruning_net.graph[k].caffe_layer.blobs[pruning_net.graph[k].saliency_pos].data[0]
    apoz[k] = (named_modules[k].blobs[0].data > 0).sum(axis=(1,2,3)) + (named_modules[k].blobs[1].data > 0)
    apoz_correct*=(np.allclose(apoz_caffe[k], apoz[k], rtol=1e-3))

  if (pointwise_saliency=='TAYLOR'):
    print("TAYLOR OK" if taylor_correct else "TAYLOR KO")
  if (pointwise_saliency=='HESSIAN_DIAG_APPROX1'):
    print("HESSIAN_DIAG OK" if hessian_diag_correct else "HESSIAN_DIAG KO")
  if (pointwise_saliency=='HESSIAN_DIAG_APPROX2'):
    print("HESSIAN_DIAG_APPROX2 OK" if hessian_diag_approx2_correct else "HESSIAN_DIAG_APPROX2 KO")
  if (pointwise_saliency=='TAYLOR_2ND'):
    print("TAYLOR_2ND OK" if taylor_2nd_correct else "TAYLOR_2ND KO")
  if (pointwise_saliency=='TAYLOR_2ND_APPROX2'):
    print("TAYLOR_2ND_APPROX2 OK" if taylor_2nd_approx2_correct else "TAYLOR_2ND_APPROX2 KO")
  if (pointwise_saliency=='AVERAGE_INPUT'):
    print("WEIGHT_AVG OK" if weight_avg_correct else "WEIGHT_AVG KO")
  if (pointwise_saliency=='AVERAGE_GRADIENT'):
    print("DIFF_AVG OK" if diff_avg_correct else "DIFF_AVG KO")
  if (pointwise_saliency=='APOZ'):
    print("APOZ OK" if apoz_correct else "APOZ KO")
