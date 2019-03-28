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

global_initial_test_acc = 0.0
global_num_test_acc_sample = 0

def compute_sparsity(summary):
    # Bound test accuracy 
    test_acc = summary['test_acc']
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
    
    new_conv_param, new_fc_param = correct_sparsity(summary, convolution_list, graph, channels, args.arch, total_channels, stop_itr=idx_test, input=args.input)
    
    sparsity = 100 - 100 *(new_conv_param.astype(float) / initial_conv_param )
    overall_sparsity = 100 - 100 *((new_conv_param + new_fc_param).astype(float) / initial_num_param )
    # Add data for unpruned network
    test_acc = np.hstack([summary['initial_test_acc'], test_acc])
    sparsity = np.hstack([0.0, sparsity])
    overall_sparsity = np.hstack([0.0, overall_sparsity])
#    axes.plot(test_acc, label=method)
    summary['test_acc'] = test_acc
    summary['sparsity'] = sparsity
    summary['overall_sparsity'] = overall_sparsity
    global global_initial_test_acc
    global global_num_test_acc_sample
    global_initial_test_acc += summary['initial_test_acc']
    global_num_test_acc_sample += 1
    if 'initial_test_loss' in summary.keys():
      summary['test_loss'] = np.hstack([summary['initial_test_loss'], summary['test_loss']])
    if 'initial_train_loss' in summary.keys():
      summary['train_loss'] = np.hstack([summary['initial_train_loss'], summary['train_loss']])
    if 'initial_train_acc' in summary.keys():
      summary['train_acc'] = np.hstack([summary['initial_train_acc'], summary['train_acc']])
    if args.characterise:
      training_iterations = np.zeros(total_channels)
      training_loss = np.array([])
      for j in range(total_channels):
        if summary['train_loss'][j] > 0:
          training_iterations[j] = len(summary['retraining_loss'][j]) if summary['retraining_loss'][j] is not None else 0
          training_loss = np.hstack([training_loss, summary['train_loss'][j], summary['retraining_loss'][j]])
      summary['training_iterations'] = np.hstack([0,np.cumsum(training_iterations)])
      # sparsity at 1% test_acc tolerance
      valid_idx = np.where(summary['test_acc'] >= 0.99*summary['initial_test_acc'])[0]
      pruning_itr_test_acc_1 = valid_idx[len(valid_idx)-1]
      # sparsity at 2% test_acc tolerance
      valid_idx = np.where(summary['test_acc'] >= 0.98*summary['initial_test_acc'])[0]
      pruning_itr_test_acc_2 = valid_idx[len(valid_idx)-1]
      # sparsity at 5% test_acc tolerance
      valid_idx = np.where(summary['test_acc'] >= 0.95*summary['initial_test_acc'])[0]
      pruning_itr_test_acc_5 = valid_idx[len(valid_idx)-1]
      # sparsity at 10% test_acc tolerance
      valid_idx = np.where(summary['test_acc'] >= 0.90*summary['initial_test_acc'])[0]
      pruning_itr_test_acc_10 = valid_idx[len(valid_idx)-1]
      # 50% sparsity
      valid_idx = np.where(summary['sparsity'] >= 0.50)[0]
      pruning_itr_sparsity_50 = valid_idx[0]
      # 60% sparsity
      valid_idx = np.where(summary['sparsity'] >= 0.60)[0]
      pruning_itr_sparsity_60 = valid_idx[0]
      # 70% sparsity
      valid_idx = np.where(summary['sparsity'] >= 0.70)[0]
      pruning_itr_sparsity_70 = valid_idx[0]
      # 80% sparsity
      valid_idx = np.where(summary['sparsity'] >= 0.80)[0]
      pruning_itr_sparsity_80 = valid_idx[0]
      # 90% sparsity
      valid_idx = np.where(summary['sparsity'] >= 0.90)[0]
      pruning_itr_sparsity_90 = valid_idx[0]

def plot_trend(metric1, metric2):
  filename_prefix = args.arch_caffe+'/results/graph/'
  filename_suffix = '.pdf'
  if args.retrain:
    filename_suffix = '_pruning' + filename_suffix
  elif args.characterise:
    filename_suffix = '_characterise' + filename_suffix
  else:
    filename_suffix = '_sensitivity' + filename_suffix
  if args.input:
    filename_suffix = '_input' + filename_suffix
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
          summary['test_acc'][0] = global_initial_test_acc
          plt.plot(summary[metric1][::args.test_interval], summary[metric2][::args.test_interval], label=norm + '-' + normalisation, color = get_color(norm, normalisation))
          # sparsity at 0.1% test_acc tolerance
          valid_idx = np.where(summary['test_acc'] >= 0.999*global_initial_test_acc)[0]
          pruning_itr_test_acc_01 = valid_idx[len(valid_idx)-1]
          # sparsity at 1% test_acc tolerance
          valid_idx = np.where(summary['test_acc'] >= 0.99*global_initial_test_acc)[0]
          pruning_itr_test_acc_1 = valid_idx[len(valid_idx)-1]
          # sparsity at 2% test_acc tolerance
          valid_idx = np.where(summary['test_acc'] >= 0.98*global_initial_test_acc)[0]
          pruning_itr_test_acc_2 = valid_idx[len(valid_idx)-1]
          plt.plot(summary[metric1][pruning_itr_test_acc_1], summary[metric2][pruning_itr_test_acc_1], color = get_color(norm, normalisation), marker = 'v')
          gc.collect()
#      plt.ylim(0, 100.0)
#      plt.xlim(0, 100.0)
#      plt.xticks(np.arange(0, 100, 10))
#      plt.yticks(np.arange(0, 100, 10))
      plt.title('Pruning Sensitivity for ' + args.arch_caffe + ', saliency: '+ saliency + ' using ' +saliency_input + 's' )
      plt.xlabel(metric1)
      plt.ylabel(metric2)
      plt.legend(loc = 'lower left',prop = {'size': 6})
      plt.grid()
      plt.savefig(filename_prefix + saliency_input + '-' + saliency + '-' + metric1 + '-' + metric2 + filename_suffix, bbox_inches='tight') 
  for saliency in python_methods:  
    plt.figure()
    for normalisation in normalisations:
      method = saliency + '-' + normalisation
      if method in summary_pruning_strategies.keys():
        summary = summary_pruning_strategies[method]
      else:
        continue
      summary['test_acc'][0] = global_initial_test_acc
      summary = summary_pruning_strategies[method]
      plt.plot(summary['sparsity'][::args.test_interval], summary['test_acc'][::args.test_interval], label=normalisation, color=get_color('none_norm', normalisation))
    plt.title('Pruning Sensitivity for ' + args.arch_caffe + ', ' + saliency)
    plt.xlabel(metric1)
    plt.ylabel(metric2)
    plt.legend(loc = 'lower left',prop = {'size': 6})
    plt.grid()
    plt.savefig(filename_prefix + saliency + '-' + metric1 + '-' + metric2 + filename_suffix, bbox_inches='tight') 


to_torch_arch = {'LeNet-5-CIFAR10': 'LeNet_5', 'AlexNet-CIFAR10': 'AlexNet', 'NIN-CIFAR10': 'NIN', 'CIFAR10-CIFAR10': 'CIFAR10', 'SqueezeNet-CIFAR10': 'SqueezeNet'}

parser = argparse.ArgumentParser()
parser.add_argument('--arch-caffe', action='store', default='LeNet-5-CIFAR10')
parser.add_argument('--retrain', action='store_true', default=False)
parser.add_argument('--characterise', action='store_true', default=False)
parser.add_argument('--input', action='store_true', default=False)
parser.add_argument('--test-interval', type=int, action='store', default=1)
parser.add_argument('--metric1', action='store', default='sparsity')
parser.add_argument('--metric2', action='store', default='test_acc')

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


caffe_methods = ['fisher', 'hessian_diag', 'hessian_diag_approx2', 'taylor_2nd', 'taylor_2nd_approx2', 'taylor', 'weight_avg', 'diff_avg']
python_methods = ['apoz']
norms = ['l1_norm', 'l2_norm', 'none_norm']
normalisations = ['no_normalisation', 'l1_normalisation', 'l2_normalisation', 'l0_normalisation', 'l0_normalisation_adjusted', 'weights_removed']
saliency_inputs = ['weight', 'activation']

all_methods = []
summary_pruning_strategies = dict()
oracle_augmented = []
#Single heuristics
for saliency_input in saliency_inputs:
  for method in caffe_methods:
    for norm in norms:
      for normalisation in normalisations:
        all_methods.append(saliency_input+'-'+ method + '-' + norm + '-' + normalisation)
for method in python_methods:
  for normalisation in normalisations:
    all_methods.append(method + '-' + normalisation)
all_methods.append('random')

methods = list(all_methods)
filename_prefix = args.arch_caffe + '/results/prune/summary_'
filename_suffix = '_caffe.npy'
if args.retrain:
  filename_prefix = filename_prefix + 'retrain_'
elif args.characterise:
  filename_prefix = filename_prefix + 'characterise_'
else:
  filename_prefix = filename_prefix + 'sensitivity_'
if args.input:
  filename_prefix = filename_prefix + 'input_channels_'
for method in all_methods:
  summary_file = filename_prefix + method + filename_suffix
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
    compute_sparsity(summary_pruning_strategies[method])

global_initial_test_acc /= float(global_num_test_acc_sample)

plot_trend(args.metric1, args.metric2)

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

