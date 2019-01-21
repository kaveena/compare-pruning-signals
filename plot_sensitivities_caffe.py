import scipy
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from plot_util import *


to_torch_arch = {'LeNet-5-CIFAR10': 'LeNet_5', 'AlexNet-CIFAR10': 'AlexNet', 'NIN-CIFAR10': 'NIN'}

parser = argparse.ArgumentParser()
parser.add_argument('--arch-caffe', action='store', default='LeNet-5-CIFAR10')
parser.add_argument('--retrain', action='store_true', default=False)

args = parser.parse_args()  

args.arch = to_torch_arch[args.arch_caffe]

stop_acc = 10.01

convolution_list = np.load(args.arch + '.convolution_list.npy').tolist()
channels = np.load(args.arch + '.channels.npy')
convolution_next = dict(np.load(args.arch + '.convolution_next.npy').item())
convolution_previous = dict(np.load(args.arch + '.convolution_previous.npy').item())
convolution_summary = dict(np.load(args.arch + '.convolution_summary.npy').item())
convolution_original = dict(np.load(args.arch + '.convolution_summary.npy').item())

caffe_methods = ['fisher', 'hessian_diag', 'hessian_diag_approx2', 'taylor_2nd', 'taylor_2nd_approx2', 'taylor']
python_methods = ['random']
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

methods = methods + python_methods
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
    convolution_summary = dict(np.load(args.arch + '.convolution_summary.npy').item())
    new_num_param = correct_sparsity(summary_pruning_strategies[method], convolution_list, convolution_original, convolution_previous, convolution_next, convolution_summary, channels, args.arch, stop_itr=idx_test)
    sparsity = 100.0 - 100*(new_num_param/float(summary_pruning_strategies[method]['initial_param']))
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
        plt.plot(summary['sparsity'], summary['test_acc'], label=norm + '-' + normalisation)
        plt.title('Pruning Sensitivity for ' + args.arch_caffe + ', saliency: '+ saliency + ' using ' +saliency_input + 's' )
        plt.xlabel('Sparsity Level in Convolution Layers')        
  plt.figure()
        plt.ylabel('Test Set Accuracy')                                                    
    plt.legend(loc = 'lower left',prop = {'size': 6})
    if args.retrain:
      plt.savefig(args.arch_caffe+'/results/graph/'+saliency_input+'-'+saliency+'_pruning.pdf', bbox_inches='tight') 
    else:
      plt.savefig(args.arch_caffe+'/results/graph/'+saliency_input+'-'+saliency+'_sensitivity.pdf', bbox_inches='tight') 

plt.figure()
for method in python_methods:  
  summary = summary_pruning_strategies[method]
  plt.plot(summary['sparsity'], summary['test_acc'], label=method)
plt.title('Pruning Sensitivity for ' + args.arch_caffe + ', other methods')
plt.xlabel('Sparsity Level in Convolution Layers')        
plt.ylabel('Test Set Accuracy')                                                    
plt.legend(loc = 'lower left',prop = {'size': 6})
if args.retrain:
  plt.savefig(args.arch_caffe+'/results/graph/weight_pruning.pdf', bbox_inches='tight') 
else:
  plt.savefig(args.arch_caffe+'/results/graph/weight_sensitivity.pdf', bbox_inches='tight') 
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
plt.show() 

