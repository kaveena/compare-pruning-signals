import scipy
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from plot_util import *
from triNNity.frontend.graph import IRGraphBuilder
from triNNity.util.transformers import DataInjector
import caffe
import triNNity
from functools import reduce
import sys
import gc

to_torch_arch = {'LeNet-5-CIFAR10': 'LeNet_5', 'AlexNet-CIFAR10': 'AlexNet', 'NIN-CIFAR10': 'NIN', 'CIFAR10-CIFAR10': 'CIFAR10', 'SqueezeNet-CIFAR10': 'SqueezeNet'}

parser = argparse.ArgumentParser()
parser.add_argument('--arch-caffe', action='store', default='LeNet-5-CIFAR10')
parser.add_argument('--method', action='store', default='activation-fisher-none_norm-no_normalisation')
parser.add_argument('--input', action='store_true', default=False)
parser.add_argument('--test-interval', type=int, action='store', default=1)

args = parser.parse_args()  

args.arch = to_torch_arch[args.arch_caffe]

stop_acc = 10.01
prototxt = 'caffe-pruned-models/'+ args.arch_caffe + '/test.prototxt'  
caffemodel = 'caffe-pruned-models/'+ args.arch_caffe + '/original.caffemodel'
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

original_graph = IRGraphBuilder(prototxt, 'test').build()
original_graph = DataInjector(prototxt, caffemodel)(original_graph)

original_graph.compute_output_shapes()
original_list_modules = original_graph.node_lut

for l in original_list_modules.keys():
  if original_list_modules[l].data is not None:
    original_list_modules[l].params_shape = list(map(lambda x: list(x.shape), original_list_modules[l].data))

convolution_list = list(filter(lambda x: 'Convolution' in net.layer_dict[x].type, net.layer_dict.keys()))

output_channels = []
total_output_channels = 0
for layer in convolution_list:
  total_output_channels += original_list_modules[layer].layer.parameters.num_output
  output_channels.append(original_list_modules[layer].layer.parameters.num_output)
output_channels = np.array(output_channels)
output_channels = np.cumsum(output_channels)

input_channels = []
total_input_channels = 0
for layer in convolution_list:
  total_input_channels += net.layer_dict[layer].blobs[0].channels
  input_channels.append(net.layer_dict[layer].blobs[0].channels)
input_channels = np.array(input_channels)
input_channels = np.cumsum(input_channels)

if args.input:
  channels = input_channels
  total_channels = total_input_channels
else:
  channels = output_channels
  total_channels = total_output_channels

initial_num_param = 0
initial_conv_param = 0
initial_fc_param = 0

update_param_shape(original_list_modules)
initial_conv_param, initial_fc_param = compute_num_param(original_list_modules) 

initial_num_param = initial_conv_param + initial_fc_param


all_eval_size = ['2', '4', '8', '16', '32', '64']
all_test_size = ['2', '4', '8', '16', '32', '64', '128']
all_train_size = ['2', '4', '8', '16', '32', '64', '128', '256', '512']

summary_pruning_strategies = dict()
all_methods=[]
#Single heuristics
for train_size in all_train_size:
  for test_size in all_test_size:
    for eval_size in all_eval_size:
      all_methods.append(args.method + '-' + eval_size + '-' + test_size + '-' + train_size)

methods = list(all_methods)
for method in all_methods:
  summary_file = args.arch_caffe+'/results/prune/summary_'+method+'_caffe.npy'
  if os.path.isfile(summary_file):
    summary_pruning_strategies[method] = dict(np.load(summary_file).item()) 
  else:
    print(summary_file+'was not found')
    methods.remove(method)

#plot test accuracy of using only one heuristic
all_pruning = list(set(methods))

for i in range(len(all_pruning)):
    method = all_pruning[i]
    if method not in summary_pruning_strategies.keys():
        continue 
    # Bound test accuracy 
    print(method)
    test_acc = summary_pruning_strategies[method]['test_acc']
    idx_test = np.where(test_acc>10.0)[0]
    idx_test = idx_test[len(idx_test)-1]
    new_test_acc = np.zeros(len(test_acc))
    new_test_acc.fill(10.0)
    for j in range(len(test_acc)):
        if j < idx_test:
            new_test_acc[j] = test_acc[j]
    test_acc = new_test_acc
    # Re-compute sparsity
    graph = IRGraphBuilder(prototxt, 'test').build()
    graph.compute_output_shapes()
    list_modules = graph.node_lut
    for l in list_modules.keys():
      if original_list_modules[l].data is not None:
        list_modules[l].params_shape = original_list_modules[l].params_shape.copy()
    
    new_num_param = correct_sparsity(summary_pruning_strategies[method], convolution_list, graph, channels, args.arch, total_channels, stop_itr=idx_test, input=args.input)
    
    sparsity = 100 - 100 *(new_num_param.astype(float) / initial_conv_param )
    # Add data for unpruned network
    test_acc = np.hstack([summary_pruning_strategies[method]['initial_test_acc'], test_acc])
    sparsity = np.hstack([0.0, sparsity])
#    axes.plot(test_acc, label=method)
    summary_pruning_strategies[method]['test_acc'] = test_acc
    summary_pruning_strategies[method]['sparsity'] = sparsity

fig, axes = plt.subplots(nrows=6, ncols=7, figsize=(60,70), sharey=True)
for i in range(len(all_test_size)):
  test_size = all_test_size[i]
  for j in range(len(all_eval_size)):
    eval_size = all_eval_size[j]
    for train_size in all_train_size:  
      method = args.method + '-' + eval_size + '-' + test_size + '-' + train_size
      if method in summary_pruning_strategies.keys():
        summary = summary_pruning_strategies[method]
      else:
        continue
      axes[j,i].plot(summary['sparsity'][::args.test_interval], summary['test_acc'][::args.test_interval], label=train_size)
    axes[j,i].grid()
    axes[j,i].set_title('test size: ' + test_size + ' eval size: ' + eval_size)
    axes[j,i].set_xlabel('Sparsity Level in Convolution Layers')
    axes[j,i].set_ylabel('Test Set Accuracy')
    axes[j,i].legend(loc = 'lower left',prop = {'size': 6})
plt.savefig(args.arch_caffe+'/results/graph/'+args.method+'_train_size.pdf', bbox_inches='tight')
