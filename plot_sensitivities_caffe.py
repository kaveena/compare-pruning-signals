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

to_torch_arch = {'LeNet-5-CIFAR10': 'LeNet_5', 'AlexNet-CIFAR10': 'AlexNet', 'NIN-CIFAR10': 'NIN', 'CIFAR10-CIFAR10': 'CIFAR10'}

parser = argparse.ArgumentParser()
parser.add_argument('--arch-caffe', action='store', default='LeNet-5-CIFAR10')
parser.add_argument('--retrain', action='store_true', default=False)
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

channels = []
total_channels = 0
for layer in convolution_list:
  total_channels += original_list_modules[layer].layer.parameters.num_output
  channels.append(original_list_modules[layer].layer.parameters.num_output)
channels = np.array(channels)
channels = np.cumsum(channels)

initial_num_param = 0
initial_conv_param = 0
initial_fc_param = 0

update_param_shape(original_list_modules)
initial_conv_param, initial_fc_param = compute_num_param(original_list_modules) 

initial_num_param = initial_conv_param + initial_fc_param


caffe_methods = ['fisher', 'hessian_diag', 'hessian_diag_approx2', 'taylor_2nd', 'taylor_2nd_approx2', 'taylor', 'weight_avg', 'diff_avg']
python_methods = ['apoz']
norms = ['l1_norm', 'l2_norm', 'none_norm']
normalisations = ['no_normalisation', 'l1_normalisation', 'l2_normalisation', 'l0_normalisation']
saliency_inputs = ['weight', 'activation']

methods = []
selected_methods = list(methods)
summary_pruning_strategies = dict()
oracle_augmented = []
#Single heuristics
for saliency_input in saliency_inputs:
  for method in caffe_methods:
    for norm in norms:
      for normalisation in normalisations:
        methods.append(saliency_input+'-'+ method + '-' + norm + '-' + normalisation)
for method in python_methods:
  for normalisation in normalisations:
    methods.append(method + '-' + normalisation)
methods.append('random')

for method in methods:
  summary_file = args.arch_caffe+'/results/prune/summary_'+method+'_caffe.npy'
  if args.retrain:
    summary_file = args.arch_caffe+'/results/prune/summary_retrain_'+method+'_caffe.npy'
  if os.path.isfile(summary_file):
    summary_pruning_strategies[method] = dict(np.load(summary_file).item()) 
  else:
    print(summary_file+'was not found')
    methods.remove(method)

#plot test accuracy of using only one heuristic
all_pruning = list(set(methods))
#fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,4), sharey=True)

for i in range(len(all_pruning)):
    method = all_pruning[i]
    if method not in summary_pruning_strategies.keys():
        continue 
    # Bound test accuracy 
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
    
    new_num_param = correct_sparsity(summary_pruning_strategies[method], convolution_list, graph, channels, args.arch, total_channels, stop_itr=idx_test)
    
    sparsity = 100 - 100 *(new_num_param.astype(float) / initial_conv_param )
    # Add data for unpruned network
    test_acc = np.hstack([summary_pruning_strategies[method]['initial_test_acc'], test_acc])
    sparsity = np.hstack([0.0, sparsity])
#    axes.plot(test_acc, label=method)
    summary_pruning_strategies[method]['test_acc'] = test_acc
    summary_pruning_strategies[method]['sparsity'] = sparsity

for saliency in caffe_methods:  
  for saliency_input in saliency_inputs:
    plt.figure()
    for norm in norms:
      for normalisation in normalisations:
        method = saliency_input + '-' + saliency + '-' + norm + '-' + normalisation
        if method in summary_pruning_strategies.keys():
          summary = summary_pruning_strategies[method]
        else:
          continue
        plt.plot(summary['sparsity'][::args.test_interval], summary['test_acc'][::args.test_interval], label=norm + '-' + normalisation)
        gc.collect()
    plt.title('Pruning Sensitivity for ' + args.arch_caffe + ', saliency: '+ saliency + ' using ' +saliency_input + 's' )
    plt.xlabel('Sparsity Level in Convolution Layers')        
    plt.ylabel('Test Set Accuracy')                                                    
    plt.legend(loc = 'lower left',prop = {'size': 6})
    if args.retrain:
      plt.savefig(args.arch_caffe+'/results/graph/'+saliency_input+'-'+saliency+'_pruning.pdf', bbox_inches='tight') 
    else:
      plt.savefig(args.arch_caffe+'/results/graph/'+saliency_input+'-'+saliency+'_sensitivity.pdf', bbox_inches='tight')

for saliency in python_methods:  
  plt.figure()
  for normalisation in normalisations:
    method = saliency + '-' + normalisation
    if method in summary_pruning_strategies.keys():
      summary = summary_pruning_strategies[method]
    else:
      continue
    summary = summary_pruning_strategies[method]
    plt.plot(summary['sparsity'][::args.test_interval], summary['test_acc'][::args.test_interval], label=method)
    plt.title('Pruning Sensitivity for ' + args.arch_caffe + ', ' + method)
    plt.xlabel('Sparsity Level in Convolution Layers')        
    plt.ylabel('Test Set Accuracy')                                                    
    plt.legend(loc = 'lower left',prop = {'size': 6})
    if args.retrain:
      plt.savefig(args.arch_caffe+'/results/graph/'+ saliency + '_pruning.pdf', bbox_inches='tight') 
    else:
      plt.savefig(args.arch_caffe+'/results/graph/'+ saliency +'_sensitivity.pdf', bbox_inches='tight') 
#axes.set_title('Pruning Sensitivity for ' + args.arch_caffe)
#plt.xlabel('Sparsity Level in Convolution Layers')        
#plt.ylabel('Test Set Accuracy')                                                     
#handles, labels = axes.get_legend_handles_labels()
#labels_part1 = tuple(label for label, handle in zip(labels, handles) if 'hybrid' not in label)
#handles_part1 = tuple(handle for label, handle in zip(labels, handles) if 'hybrid' not in label)
#labels_part1, handles_part1 = zip(*sorted(zip(labels_part1, handles_part1), key=lambda t: t[0]))
#labels_part2 = tuple(label for label, handle in zip(labels, handles) if 'hybrid' in label)
#handles_part2 = tuple(handle for label, handle in zip(labels, handles) if 'hybrid' in label)
#labels_part2, handles_part2 = zip(*sorted(zip(labels_part2, handles_part2), key=lambda t: t[0]))
#labels = labels_part1 + labels_part2
#handles = handles_part1 + handles_part2
#l = fig.legend(handles, labels, loc='right', ncol=1)
#fig.legend(handles, labels, bbox_to_anchor=(0.525, 0.0), ncol=4)
#plt.tight_layout(h_pad=1000.0)
#plt.subplots_adjust(bottom=0.4)
#plt.savefig(args.arch+'_pruning_.pdf') 
#plt.show() 

