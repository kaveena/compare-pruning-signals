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
import pandas as pd
from collections import namedtuple

global_initial_test_acc = 0.0
global_num_test_acc_samples = 0

def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if not os.path.exists(directory):
    os.makedirs(directory)

def get_info(method):
  if method == 'random':
    return 'N', 'random', 'N', 'N'
  method_info = method.split('-')
  if len(method_info) == 2:
    if method_info[0] == 'apoz':
      return 'activation', 'apoz', 'N', method_info[1]
  if len(method_info) == 4:
      return method_info[0], method_info[1], method_info[2], method_info[3]

def compute_sparsity(summary_dict, summary_method, characterise):
  summary = summary_dict[summary_method]
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
#  axes.plot(test_acc, label=method)
  summary['test_acc'] = test_acc
  summary['sparsity'] = sparsity
  summary['overall_sparsity'] = overall_sparsity
  if 'initial_test_loss' in summary.keys():
    summary['test_loss'] = np.hstack([summary['initial_test_loss'], summary['test_loss']])
  if 'initial_train_loss' in summary.keys() and 'train_loss' in summary.keys():
    summary['train_loss'] = np.hstack([summary['initial_train_loss'], summary['train_loss']])
  if 'initial_eval_loss' in summary.keys():
    summary['eval_loss'] = np.hstack([summary['initial_eval_loss'], summary['eval_loss']])
  if 'initial_eval_acc' in summary.keys():
    summary['eval_acc'] = np.hstack([summary['initial_eval_acc'], summary['eval_acc']])
  else:
    print(summary_method + " does not have eval acc")
  if 'initial_train_acc' in summary.keys():
    summary['train_acc'] = np.hstack([summary['initial_train_acc'], summary['train_acc']])
  if  not ('train_loss' in summary.keys() and ('retraining_loss' in summary.keys())) and ('characterise' in summary_method):
    print(summary_method + ' has no train loss')
  csv_prefix = args.arch_caffe + '/results/csv/'
  ensure_dir(csv_prefix)
  if '_characterise' in summary_method:
    csv_prefix = csv_prefix + 'characterise/'
    save_method = summary_method.replace('_characterise', '')
    training_iterations = np.zeros(total_channels)
    training_loss = np.array([])
    for j in range(total_channels):
      if summary['train_loss'][j] > 0:
        training_iterations[j] = len(summary['retraining_loss'][j]) if summary['retraining_loss'][j] is not None else 0
        training_loss = np.hstack([training_loss, summary['train_loss'][j], summary['retraining_loss'][j]])
    summary['training_iterations'] = np.hstack([0,np.cumsum(training_iterations)])
  else: 
    summary['train_loss'] = np.zeros(len(summary['test_acc']))
    summary['train_acc'] = np.zeros(len(summary['test_acc']))
    summary['training_iterations'] = np.zeros(len(summary['test_acc']))
    csv_prefix = csv_prefix + 'sensitivity/'
    save_method = summary_method
  ensure_dir(csv_prefix)
  summary['pruning_iterations'] = np.arange(len(summary['test_acc']))
  test_acc_to_save = np.hstack([summary['test_acc'][0], summary['test_acc'][1::args.test_interval]])
  test_loss_to_save = np.hstack([summary['test_loss'][0], summary['test_loss'][1::args.test_interval]])
  train_acc_to_save = np.hstack([summary['train_acc'][0], summary['train_acc'][1::args.test_interval]])
  train_loss_to_save = np.hstack([summary['train_loss'][0], summary['train_loss'][1::args.test_interval]])
  eval_acc_to_save = np.hstack([summary['eval_acc'][0], summary['eval_acc'][1::args.test_interval]])
  eval_loss_to_save = np.hstack([summary['eval_loss'][0], summary['eval_loss'][1::args.test_interval]])
  sparsity_to_save = np.hstack([summary['sparsity'][0], summary['sparsity'][1::args.test_interval]])
  overall_sparsity_to_save = np.hstack([summary['overall_sparsity'][0], summary['overall_sparsity'][1::args.test_interval]])
  training_iterations_to_save = np.hstack([summary['training_iterations'][0], summary['training_iterations'][1::args.test_interval]])
  pruning_iterations_to_save = np.hstack([summary['pruning_iterations'][0], summary['pruning_iterations'][1::args.test_interval]])
  
  num_conv_param_to_save = np.hstack([initial_conv_param, new_conv_param[0::args.test_interval]])
  num_fc_param_to_save = np.hstack([initial_fc_param, new_fc_param[0::args.test_interval]])
  num_param_to_save = num_conv_param_to_save + num_fc_param_to_save
  
  csv_ = pd.DataFrame(data = np.stack([ pruning_iterations_to_save, \
                                        test_acc_to_save, \
                                        sparsity_to_save, \
                                        overall_sparsity_to_save, \
                                        num_param_to_save, 
                                        num_conv_param_to_save, \
                                        num_fc_param_to_save, \
                                        test_loss_to_save, \
                                        eval_acc_to_save, \
                                        eval_loss_to_save, \
                                        train_acc_to_save, \
                                        train_loss_to_save, \
                                        training_iterations_to_save \
                                        ]).transpose(), \
                      columns=[         'pruning_iterations', \
                                        'test_acc', \
                                        'sparsity', \
                                        'overall_sparsity', \
                                        'num_param', \
                                        'conv_param', \
                                        'fc_param', \
                                        'test_loss', \
                                        'eval_acc', \
                                        'eval_loss', \
                                        'train_acc', \
                                        'train_loss', \
                                        'training_iterations', \
                                        ])
  ensure_dir(csv_prefix)
  csv_.to_csv(csv_prefix + save_method)
  
def extract_points(summary_dict, summary_method, characterise):
  summary = summary_dict[summary_method]
  summary['test_acc'][0] = global_initial_test_acc
  # sparsity at max test_acc
  valid_idx = np.where(summary['test_acc'] == np.max(summary['test_acc']))[0]
  pruning_itr_test_acc_max = valid_idx[len(valid_idx)-1]
  # sparsity at 0.1% test_acc tolerance
  valid_idx = np.where(summary['test_acc'] >= 0.999*global_initial_test_acc)[0]
  pruning_itr_test_acc_01 = valid_idx[len(valid_idx)-1]
  # sparsity at 1% test_acc tolerance
  valid_idx = np.where(summary['test_acc'] >= 0.99*global_initial_test_acc)[0]
  pruning_itr_test_acc_1 = valid_idx[len(valid_idx)-1]
  # sparsity at 2% test_acc tolerance
  valid_idx = np.where(summary['test_acc'] >= 0.98*global_initial_test_acc)[0]
  pruning_itr_test_acc_2 = valid_idx[len(valid_idx)-1]
  # sparsity at 5% test_acc tolerance
  valid_idx = np.where(summary['test_acc'] >= 0.95*global_initial_test_acc)[0]
  pruning_itr_test_acc_5 = valid_idx[len(valid_idx)-1]
  # sparsity at 10% test_acc tolerance
  valid_idx = np.where(summary['test_acc'] >= 0.90*global_initial_test_acc)[0]
  pruning_itr_test_acc_10 = valid_idx[len(valid_idx)-1]
  # sparsity at 0.1% test_acc drop
  valid_idx = np.where(summary['test_acc'] >= global_initial_test_acc - 0.1)[0]
  pruning_itr_test_acc_abs_01 = valid_idx[len(valid_idx)-1]
  # sparsity at 1% test_acc drop
  valid_idx = np.where(summary['test_acc'] >= global_initial_test_acc - 1.0)[0]
  pruning_itr_test_acc_abs_1 = valid_idx[len(valid_idx)-1]
  # sparsity at 2% test_acc drop
  valid_idx = np.where(summary['test_acc'] >= global_initial_test_acc - 2.0)[0]
  pruning_itr_test_acc_abs_2 = valid_idx[len(valid_idx)-1]
  # sparsity at 5% test_acc drop
  valid_idx = np.where(summary['test_acc'] >= global_initial_test_acc - 5.0)[0]
  pruning_itr_test_acc_abs_5 = valid_idx[len(valid_idx)-1]
  # sparsity at 10% test_acc drop
  valid_idx = np.where(summary['test_acc'] >= global_initial_test_acc - 10.0)[0]
  pruning_itr_test_acc_abs_10 = valid_idx[len(valid_idx)-1]
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
  return pruning_itr_test_acc_max, pruning_itr_test_acc_01, pruning_itr_test_acc_1, pruning_itr_test_acc_2, pruning_itr_test_acc_5, pruning_itr_test_acc_abs_01, pruning_itr_test_acc_abs_1, pruning_itr_test_acc_abs_2, pruning_itr_test_acc_abs_5, pruning_itr_sparsity_50, pruning_itr_sparsity_60, pruning_itr_sparsity_70, pruning_itr_sparsity_80, pruning_itr_sparsity_90

to_torch_arch = {'LeNet-5-CIFAR10': 'LeNet_5', 'AlexNet-CIFAR10': 'AlexNet', 'NIN-CIFAR10': 'NIN', 'CIFAR10-CIFAR10': 'CIFAR10', 'SqueezeNet-CIFAR10': 'SqueezeNet'}

parser = argparse.ArgumentParser()
parser.add_argument('--arch-caffe', action='store', default='LeNet-5-CIFAR10')
parser.add_argument('--retrain', action='store_true', default=False)
parser.add_argument('--characterise', action='store_true', default=False)
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


caffe_methods = ['hessian_diag', 'hessian_diag_approx2', 'taylor_2nd', 'taylor_2nd_approx2', 'taylor', 'weight_avg', 'diff_avg']
python_methods = ['apoz']
norms = ['l1_norm', 'l2_norm', 'none_norm', 'abs_sum_norm', 'sqr_sum_norm']
normalisations = ['no_normalisation', 'l1_normalisation', 'l2_normalisation', 'l0_normalisation', 'l0_normalisation_adjusted', 'weights_removed']
saliency_inputs = ['weight', 'activation']

all_methods = []
summary_pruning_strategies = dict()
oracle_augmented = []
#Single heuristics
for saliency_input in saliency_inputs:
  for method in caffe_methods:
    for norm in norms:
      if method == 'fisher' and norm != 'none_norm':
        continue
      for normalisation in normalisations:
        all_methods.append(saliency_input+'-'+ method + '-' + norm + '-' + normalisation)
for method in python_methods:
  for normalisation in normalisations:
    all_methods.append(method + '-' + normalisation)
all_methods.append('random')

methods = list(all_methods)
filename_prefix = args.arch_caffe + '/results/prune/summary_'
filename_suffix = '_caffe.npy'
filename_sensitivity_prefix = filename_prefix + 'sensitivity_'
filename_characterise_prefix = filename_prefix + 'characterise_'
if args.input:
  filename_sensitivity_prefix = filename_sensitivity_prefix + 'input_channels_'
  filename_characterise_prefix = filename_characterise_prefix + 'input_channels_'
for method in all_methods:
  summary_file_sensitivity = filename_sensitivity_prefix + method + filename_suffix
  summary_file_characterise = filename_characterise_prefix + method + filename_suffix
  if os.path.isfile(summary_file_sensitivity) and os.path.isfile(summary_file_characterise):
    summary_pruning_strategies[method] = dict(np.load(summary_file_sensitivity, allow_pickle=True).item()) 
    summary_pruning_strategies[method+ '_characterise'] = dict(np.load(summary_file_characterise, allow_pickle=True).item()) 
  else:
    if ('weights_removed' not in method) and ('l0_normalisation_adjusted' not in method):
      print(method + ' skipped')
    methods.remove(method)

#plot test accuracy of using only one heuristic
all_pruning = list(set(methods))
#fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,4), sharey=True)

column_names = ['network', \
'name', \
'saliency_input', \
'saliency_method', \
'saliency_norm', \
'layerwise_normalisation', \
'initial_test_accuracy', \
'initial_test_loss', \
'test_acc_max', \
'sparsity_test_acc_max', \
'overall_sparsity_test_acc_max', \
'test_loss_test_acc_max', \
'eval_acc_test_acc_max', \
'eval_loss_test_Acc_max', \
'pruning_iterations_test_acc_max', \
'test_acc_01', \
'sparsity_test_acc_01', \
'overall_sparsity_test_acc_01', \
'test_loss_test_acc_01', \
'eval_acc_test_acc_01', \
'eval_loss_test_Acc_01', \
'pruning_iterations_test_acc_01', \
'test_acc_1', \
'sparsity_test_acc_1', \
'overall_sparsity_test_acc_1', \
'test_loss_test_acc_1', \
'eval_acc_test_acc_1', \
'eval_loss_test_Acc_1', \
'pruning_iterations_test_acc_1', \
'test_acc_2', \
'sparsity_test_acc_2', \
'overall_sparsity_test_acc_2', \
'test_loss_test_acc_2', \
'eval_acc_test_acc_2', \
'eval_loss_test_Acc_2', \
'pruning_iterations_test_acc_2', \
'test_acc_5', \
'sparsity_test_acc_5', \
'overall_sparsity_test_acc_5', \
'test_loss_test_acc_5', \
'eval_acc_test_acc_5', \
'eval_loss_test_Acc_5', \
'pruning_iterations_test_acc_5', \
'characterise_test_acc_max', \
'characterise_sparsity_test_acc_max', \
'characterise_overall_sparsity_test_acc_max', \
'characterise_test_loss_test_acc_max', \
'characterise_eval_acc_test_acc_max', \
'characterise_eval_loss_test_Acc_max', \
'characterise_pruning_iterations_test_acc_max', \
'characterise_test_acc_01', \
'characterise_sparsity_test_acc_01', \
'characterise_overall_sparsity_test_acc_01', \
'characterise_test_loss_test_acc_01', \
'characterise_eval_acc_test_acc_01', \
'characterise_eval_loss_test_acc_01', \
'characterise_training_iterations_test_acc_01', \
'characterise_pruning_iterations_test_acc_01', \
'characterise_test_acc_1', \
'characterise_sparsity_test_acc_1', \
'characterise_overall_sparsity_test_acc_1', \
'characterise_test_loss_test_acc_1', \
'characterise_eval_acc_test_acc_1', \
'characterise_eval_loss_test_acc_1', \
'characterise_training_iterations_test_acc_1', \
'characterise_pruning_iterations_test_acc_1', \
'characterise_sparsity_test_acc_2', \
'characterise_overall_sparsity_test_acc_2', \
'characterise_test_loss_test_acc_2', \
'characterise_eval_acc_test_acc_2', \
'characterise_eval_loss_test_acc_2', \
'characterise_training_iterations_test_acc_2', \
'characterise_pruning_iterations_test_acc_2', \
'characterise_sparsity_test_acc_5', \
'characterise_overall_sparsity_test_acc_5', \
'characterise_test_loss_test_acc_5', \
'characterise_eval_acc_test_acc_5', \
'characterise_eval_loss_test_acc_5', \
'characterise_training_iterations_test_acc_5', \
'characterise_pruning_iterations_test_acc_5', \
'test_acc_abs_01', \
'sparsity_test_acc_abs_01', \
'overall_sparsity_test_acc_abs_01', \
'test_loss_test_acc_abs_01', \
'eval_acc_test_acc_abs_01', \
'eval_loss_test_acc_abs_01', \
'pruning_iterations_test_acc_abs_01', \
'test_acc_abs_1', \
'sparsity_test_acc_abs_1', \
'overall_sparsity_test_acc_abs_1', \
'test_loss_test_acc_abs_1', \
'eval_acc_test_acc_abs_1', \
'eval_loss_test_acc_abs_1', \
'pruning_iterations_test_acc_abs_1', \
'test_acc_abs_2', \
'sparsity_test_acc_abs_2', \
'overall_sparsity_test_acc_abs_2', \
'test_loss_test_acc_abs_2', \
'eval_acc_test_acc_abs_2', \
'eval_loss_test_acc_abs_2', \
'pruning_iterations_test_acc_abs_2', \
'test_acc_abs_5', \
'sparsity_test_acc_abs_5', \
'overall_sparsity_test_acc_abs_5', \
'test_loss_test_acc_abs_5', \
'eval_acc_test_acc_abs_5', \
'eval_loss_test_acc_abs_5', \
'pruning_iterations_test_acc_abs_5', \
'characterise_test_acc_abs_01', \
'characterise_sparsity_test_acc_abs_01', \
'characterise_overall_sparsity_test_acc_abs_01', \
'characterise_test_loss_test_acc_abs_01', \
'characterise_eval_acc_test_acc_abs_01', \
'characterise_eval_loss_test_acc_abs_01', \
'characterise_training_iterations_test_acc_abs_01', \
'characterise_pruning_iterations_test_acc_abs_01', \
'characterise_test_acc_abs_1', \
'characterise_sparsity_test_acc_abs_1', \
'characterise_overall_sparsity_test_acc_abs_1', \
'characterise_test_loss_test_acc_abs_1', \
'characterise_eval_acc_test_acc_abs_1', \
'characterise_eval_loss_test_acc_abs_1', \
'characterise_training_iterations_test_acc_abs_1', \
'characterise_pruning_iterations_test_acc_abs_1', \
'characterise_sparsity_test_acc_abs_2', \
'characterise_overall_sparsity_test_acc_abs_2', \
'characterise_test_loss_test_acc_abs_2', \
'characterise_eval_acc_test_acc_abs_2', \
'characterise_eval_loss_test_acc_abs_2', \
'characterise_training_iterations_test_acc_abs_2', \
'characterise_pruning_iterations_test_acc_abs_2', \
'characterise_sparsity_test_acc_abs_5', \
'characterise_overall_sparsity_test_acc_abs_5', \
'characterise_test_loss_test_acc_abs_5', \
'characterise_eval_acc_test_acc_abs_5', \
'characterise_eval_loss_test_acc_abs_5', \
'characterise_training_iterations_test_acc_abs_5', \
'characterise_pruning_iterations_test_acc_abs_5']

df = pd.DataFrame(columns=column_names)

global_initial_test_acc = 0.0
global_num_test_acc_samples = 0
for method in summary_pruning_strategies.keys():
  global_initial_test_acc += summary_pruning_strategies[method]['initial_test_acc']
  global_num_test_acc_samples += 1

global_initial_test_acc = global_initial_test_acc / float(global_num_test_acc_samples)

print(global_initial_test_acc)

for method in summary_pruning_strategies.keys():
  summary_pruning_strategies[method]['initial_test_acc'] = global_initial_test_acc

for i in range(len(all_pruning)):
  method = all_pruning[i]
  if method not in summary_pruning_strategies.keys():
    continue 
  summary_sensitivity = summary_pruning_strategies[method]
  summary_characterise = summary_pruning_strategies[method+'_characterise']
  compute_sparsity(summary_pruning_strategies, method, False)
  compute_sparsity(summary_pruning_strategies, method + '_characterise', True)


for i in range(len(all_pruning)):
  method = all_pruning[i]
  if method not in summary_pruning_strategies.keys():
    continue 
  summary_sensitivity = summary_pruning_strategies[method]
  summary_characterise = summary_pruning_strategies[method+'_characterise']
  pruning_itr_test_acc_max, pruning_itr_test_acc_01, pruning_itr_test_acc_1, pruning_itr_test_acc_2, pruning_itr_test_acc_5, pruning_itr_test_acc_abs_01, pruning_itr_test_acc_abs_1, pruning_itr_test_acc_abs_2, pruning_itr_test_acc_abs_5, pruning_itr_sparsity_50, pruning_itr_sparsity_60, pruning_itr_sparsity_70, pruning_itr_sparsity_80, pruning_itr_sparsity_90 = extract_points(summary_pruning_strategies, method, False)
  pruning_itr_characterise_test_acc_max, pruning_itr_characterise_test_acc_01, pruning_itr_characterise_test_acc_1, pruning_itr_characterise_test_acc_2, pruning_itr_characterise_test_acc_5, pruning_itr_characterise_test_acc_abs_01, pruning_itr_characterise_test_acc_abs_1, pruning_itr_characterise_test_acc_abs_2, pruning_itr_characterise_test_acc_abs_5, pruning_itr_sparsity_50, pruning_itr_sparsity_60, pruning_itr_sparsity_70, pruning_itr_sparsity_80, pruning_itr_sparsity_90 = extract_points(summary_pruning_strategies, method + '_characterise', True)
  saliency_input, saliency_method, saliency_norm, normalisation = get_info(method) 
  
  df.loc[len(df)] = {
    'network': args.arch_caffe,
    'name': method,
    'saliency_input': saliency_input,
    'saliency_method': saliency_method,
    'saliency_norm': saliency_norm,
    'layerwise_normalisation': normalisation,
    'initial_test_acc': global_initial_test_acc,
    'test_acc_max': summary_sensitivity['test_acc'][pruning_itr_test_acc_max], 
    'sparsity_test_acc_max': summary_sensitivity['sparsity'][pruning_itr_test_acc_max],
    'overall_sparsity_test_acc_max': summary_sensitivity['overall_sparsity'][pruning_itr_test_acc_max],
    'test_loss_test_acc_max': summary_sensitivity['test_loss'][pruning_itr_test_acc_max],
    'eval_acc_test_acc_max': summary_sensitivity['eval_acc'][pruning_itr_test_acc_max],
    'eval_loss_test_Acc_max': summary_sensitivity['eval_loss'][pruning_itr_test_acc_max],
    'pruning_iterations_test_acc_max': pruning_itr_test_acc_max,
    'test_acc_01': summary_sensitivity['test_acc'][pruning_itr_test_acc_01], 
    'sparsity_test_acc_01': summary_sensitivity['sparsity'][pruning_itr_test_acc_01],
    'overall_sparsity_test_acc_01': summary_sensitivity['overall_sparsity'][pruning_itr_test_acc_01],
    'test_loss_test_acc_01': summary_sensitivity['test_loss'][pruning_itr_test_acc_01],
    'eval_acc_test_acc_01': summary_sensitivity['eval_acc'][pruning_itr_test_acc_01],
    'eval_loss_test_Acc_01': summary_sensitivity['eval_loss'][pruning_itr_test_acc_01],
    'pruning_iterations_test_acc_01': pruning_itr_test_acc_01,
    'test_acc_1': summary_sensitivity['test_acc'][pruning_itr_test_acc_1], 
    'sparsity_test_acc_1': summary_sensitivity['sparsity'][pruning_itr_test_acc_1],
    'overall_sparsity_test_acc_1': summary_sensitivity['overall_sparsity'][pruning_itr_test_acc_1],
    'test_loss_test_acc_1': summary_sensitivity['test_loss'][pruning_itr_test_acc_1],
    'eval_acc_test_acc_1': summary_sensitivity['eval_acc'][pruning_itr_test_acc_1],
    'eval_loss_test_Acc_1': summary_sensitivity['eval_loss'][pruning_itr_test_acc_1],
    'pruning_iterations_test_acc_1': pruning_itr_test_acc_1,
    'test_acc_2': summary_sensitivity['test_acc'][pruning_itr_test_acc_2], 
    'sparsity_test_acc_2': summary_sensitivity['sparsity'][pruning_itr_test_acc_2],
    'overall_sparsity_test_acc_2': summary_sensitivity['overall_sparsity'][pruning_itr_test_acc_2],
    'test_loss_test_acc_2': summary_sensitivity['test_loss'][pruning_itr_test_acc_2],
    'eval_acc_test_acc_2': summary_sensitivity['eval_acc'][pruning_itr_test_acc_2],
    'eval_loss_test_Acc_2': summary_sensitivity['eval_loss'][pruning_itr_test_acc_2],
    'pruning_iterations_test_acc_2': pruning_itr_test_acc_2,
    'test_acc_5': summary_sensitivity['test_acc'][pruning_itr_test_acc_5], 
    'sparsity_test_acc_5': summary_sensitivity['sparsity'][pruning_itr_test_acc_5],
    'overall_sparsity_test_acc_5': summary_sensitivity['overall_sparsity'][pruning_itr_test_acc_5],
    'test_loss_test_acc_5': summary_sensitivity['test_loss'][pruning_itr_test_acc_5],
    'eval_acc_test_acc_5': summary_sensitivity['eval_acc'][pruning_itr_test_acc_5],
    'eval_loss_test_Acc_5': summary_sensitivity['eval_loss'][pruning_itr_test_acc_5],
    'pruning_iterations_test_acc_5': pruning_itr_test_acc_5,
    'characterise_test_acc_max': summary_characterise['test_acc'][pruning_itr_characterise_test_acc_max],
    'characterise_sparsity_test_acc_max': summary_characterise['sparsity'][pruning_itr_characterise_test_acc_max],
    'characterise_overall_sparsity_test_acc_max': summary_characterise['overall_sparsity'][pruning_itr_characterise_test_acc_max],
    'characterise_test_loss_test_acc_max': summary_characterise['test_loss'][pruning_itr_characterise_test_acc_max],
    'characterise_eval_acc_test_acc_max': summary_characterise['eval_acc'][pruning_itr_characterise_test_acc_max],
    'characterise_eval_loss_test_acc_max': summary_characterise['eval_loss'][pruning_itr_characterise_test_acc_max],
    'characterise_training_iterations_test_acc_max': summary_characterise['training_iterations'][pruning_itr_characterise_test_acc_max],
    'characterise_pruning_iterations_test_acc_max': pruning_itr_characterise_test_acc_max,
    'characterise_test_acc_01': summary_characterise['test_acc'][pruning_itr_characterise_test_acc_01],
    'characterise_sparsity_test_acc_01': summary_characterise['sparsity'][pruning_itr_characterise_test_acc_01],
    'characterise_overall_sparsity_test_acc_01': summary_characterise['overall_sparsity'][pruning_itr_characterise_test_acc_01],
    'characterise_test_loss_test_acc_01': summary_characterise['test_loss'][pruning_itr_characterise_test_acc_01],
    'characterise_eval_acc_test_acc_01': summary_characterise['eval_acc'][pruning_itr_characterise_test_acc_01],
    'characterise_eval_loss_test_acc_01': summary_characterise['eval_loss'][pruning_itr_characterise_test_acc_01],
    'characterise_training_iterations_test_acc_01': summary_characterise['training_iterations'][pruning_itr_characterise_test_acc_01],
    'characterise_pruning_iterations_test_acc_01': pruning_itr_characterise_test_acc_01,
    'characterise_test_acc_1': summary_characterise['test_acc'][pruning_itr_characterise_test_acc_1],
    'characterise_sparsity_test_acc_1': summary_characterise['sparsity'][pruning_itr_characterise_test_acc_1],
    'characterise_overall_sparsity_test_acc_1': summary_characterise['overall_sparsity'][pruning_itr_characterise_test_acc_1],
    'characterise_test_loss_test_acc_1': summary_characterise['test_loss'][pruning_itr_characterise_test_acc_1],
    'characterise_eval_acc_test_acc_1': summary_characterise['eval_acc'][pruning_itr_characterise_test_acc_1],
    'characterise_eval_loss_test_acc_1': summary_characterise['eval_loss'][pruning_itr_characterise_test_acc_1],
    'characterise_training_iterations_test_acc_1': summary_characterise['training_iterations'][pruning_itr_characterise_test_acc_1],
    'characterise_pruning_iterations_test_acc_1': pruning_itr_characterise_test_acc_1,
    'characterise_sparsity_test_acc_2': summary_characterise['sparsity'][pruning_itr_characterise_test_acc_2],
    'characterise_overall_sparsity_test_acc_2': summary_characterise['overall_sparsity'][pruning_itr_characterise_test_acc_2],
    'characterise_test_loss_test_acc_2': summary_characterise['test_loss'][pruning_itr_characterise_test_acc_2],
    'characterise_eval_acc_test_acc_2': summary_characterise['eval_acc'][pruning_itr_characterise_test_acc_2],
    'characterise_eval_loss_test_acc_2': summary_characterise['eval_loss'][pruning_itr_characterise_test_acc_2],
    'characterise_training_iterations_test_acc_2': summary_characterise['training_iterations'][pruning_itr_characterise_test_acc_2],
    'characterise_pruning_iterations_test_acc_2': pruning_itr_characterise_test_acc_2,
    'characterise_sparsity_test_acc_5': summary_characterise['sparsity'][pruning_itr_characterise_test_acc_5],
    'characterise_overall_sparsity_test_acc_5': summary_characterise['overall_sparsity'][pruning_itr_characterise_test_acc_5],
    'characterise_test_loss_test_acc_5': summary_characterise['test_loss'][pruning_itr_characterise_test_acc_5],
    'characterise_eval_acc_test_acc_5': summary_characterise['eval_acc'][pruning_itr_characterise_test_acc_5],
    'characterise_eval_loss_test_acc_5': summary_characterise['eval_loss'][pruning_itr_characterise_test_acc_5],
    'characterise_training_iterations_test_acc_5': summary_characterise['training_iterations'][pruning_itr_characterise_test_acc_5],
    'characterise_pruning_iterations_test_acc_5': pruning_itr_characterise_test_acc_5,
    'test_acc_abs_01': summary_sensitivity['test_acc'][pruning_itr_test_acc_abs_01], 
    'sparsity_test_acc_abs_01': summary_sensitivity['sparsity'][pruning_itr_test_acc_abs_01],
    'overall_sparsity_test_acc_abs_01': summary_sensitivity['overall_sparsity'][pruning_itr_test_acc_abs_01],
    'test_loss_test_acc_abs_01': summary_sensitivity['test_loss'][pruning_itr_test_acc_abs_01],
    'eval_acc_test_acc_abs_01': summary_sensitivity['eval_acc'][pruning_itr_test_acc_abs_01],
    'eval_loss_test_acc_abs_01': summary_sensitivity['eval_loss'][pruning_itr_test_acc_abs_01],
    'pruning_iterations_test_acc_abs_01': pruning_itr_test_acc_abs_01,
    'test_acc_abs_1': summary_sensitivity['test_acc'][pruning_itr_test_acc_abs_1], 
    'sparsity_test_acc_abs_1': summary_sensitivity['sparsity'][pruning_itr_test_acc_abs_1],
    'overall_sparsity_test_acc_abs_1': summary_sensitivity['overall_sparsity'][pruning_itr_test_acc_abs_1],
    'test_loss_test_acc_abs_1': summary_sensitivity['test_loss'][pruning_itr_test_acc_abs_1],
    'eval_acc_test_acc_abs_1': summary_sensitivity['eval_acc'][pruning_itr_test_acc_abs_1],
    'eval_loss_test_acc_abs_1': summary_sensitivity['eval_loss'][pruning_itr_test_acc_abs_1],
    'pruning_iterations_test_acc_abs_1': pruning_itr_test_acc_abs_1,
    'test_acc_abs_2': summary_sensitivity['test_acc'][pruning_itr_test_acc_abs_2], 
    'sparsity_test_acc_abs_2': summary_sensitivity['sparsity'][pruning_itr_test_acc_abs_2],
    'overall_sparsity_test_acc_abs_2': summary_sensitivity['overall_sparsity'][pruning_itr_test_acc_abs_2],
    'test_loss_test_acc_abs_2': summary_sensitivity['test_loss'][pruning_itr_test_acc_abs_2],
    'eval_acc_test_acc_abs_2': summary_sensitivity['eval_acc'][pruning_itr_test_acc_abs_2],
    'eval_loss_test_acc_abs_2': summary_sensitivity['eval_loss'][pruning_itr_test_acc_abs_2],
    'pruning_iterations_test_acc_abs_2': pruning_itr_test_acc_abs_2,
    'test_acc_abs_5': summary_sensitivity['test_acc'][pruning_itr_test_acc_abs_5], 
    'sparsity_test_acc_abs_5': summary_sensitivity['sparsity'][pruning_itr_test_acc_abs_5],
    'overall_sparsity_test_acc_abs_5': summary_sensitivity['overall_sparsity'][pruning_itr_test_acc_abs_5],
    'test_loss_test_acc_abs_5': summary_sensitivity['test_loss'][pruning_itr_test_acc_abs_5],
    'eval_acc_test_acc_abs_5': summary_sensitivity['eval_acc'][pruning_itr_test_acc_abs_5],
    'eval_loss_test_acc_abs_5': summary_sensitivity['eval_loss'][pruning_itr_test_acc_abs_5],
    'pruning_iterations_test_acc_abs_5': pruning_itr_test_acc_abs_5,
    'characterise_test_acc_abs_01': summary_characterise['test_acc'][pruning_itr_characterise_test_acc_abs_01],
    'characterise_sparsity_test_acc_abs_01': summary_characterise['sparsity'][pruning_itr_characterise_test_acc_abs_01],
    'characterise_overall_sparsity_test_acc_abs_01': summary_characterise['overall_sparsity'][pruning_itr_characterise_test_acc_abs_01],
    'characterise_test_loss_test_acc_abs_01': summary_characterise['test_loss'][pruning_itr_characterise_test_acc_abs_01],
    'characterise_eval_acc_test_acc_abs_01': summary_characterise['eval_acc'][pruning_itr_characterise_test_acc_abs_01],
    'characterise_eval_loss_test_acc_abs_01': summary_characterise['eval_loss'][pruning_itr_characterise_test_acc_abs_01],
    'characterise_training_iterations_test_acc_abs_01': summary_characterise['training_iterations'][pruning_itr_characterise_test_acc_abs_01],
    'characterise_pruning_iterations_test_acc_abs_01': pruning_itr_characterise_test_acc_abs_01,
    'characterise_test_acc_abs_1': summary_characterise['test_acc'][pruning_itr_characterise_test_acc_abs_1],
    'characterise_sparsity_test_acc_abs_1': summary_characterise['sparsity'][pruning_itr_characterise_test_acc_abs_1],
    'characterise_overall_sparsity_test_acc_abs_1': summary_characterise['overall_sparsity'][pruning_itr_characterise_test_acc_abs_1],
    'characterise_test_loss_test_acc_abs_1': summary_characterise['test_loss'][pruning_itr_characterise_test_acc_abs_1],
    'characterise_eval_acc_test_acc_abs_1': summary_characterise['eval_acc'][pruning_itr_characterise_test_acc_abs_1],
    'characterise_eval_loss_test_acc_abs_1': summary_characterise['eval_loss'][pruning_itr_characterise_test_acc_abs_1],
    'characterise_training_iterations_test_acc_abs_1': summary_characterise['training_iterations'][pruning_itr_characterise_test_acc_abs_1],
    'characterise_pruning_iterations_test_acc_abs_1': pruning_itr_characterise_test_acc_abs_1,
    'characterise_sparsity_test_acc_abs_2': summary_characterise['sparsity'][pruning_itr_characterise_test_acc_abs_2],
    'characterise_overall_sparsity_test_acc_abs_2': summary_characterise['overall_sparsity'][pruning_itr_characterise_test_acc_abs_2],
    'characterise_test_loss_test_acc_abs_2': summary_characterise['test_loss'][pruning_itr_characterise_test_acc_abs_2],
    'characterise_eval_acc_test_acc_abs_2': summary_characterise['eval_acc'][pruning_itr_characterise_test_acc_abs_2],
    'characterise_eval_loss_test_acc_abs_2': summary_characterise['eval_loss'][pruning_itr_characterise_test_acc_abs_2],
    'characterise_training_iterations_test_acc_abs_2': summary_characterise['training_iterations'][pruning_itr_characterise_test_acc_abs_2],
    'characterise_pruning_iterations_test_acc_abs_2': pruning_itr_characterise_test_acc_abs_2,
    'characterise_sparsity_test_acc_abs_5': summary_characterise['sparsity'][pruning_itr_characterise_test_acc_abs_5],
    'characterise_overall_sparsity_test_acc_abs_5': summary_characterise['overall_sparsity'][pruning_itr_characterise_test_acc_abs_5],
    'characterise_test_loss_test_acc_abs_5': summary_characterise['test_loss'][pruning_itr_characterise_test_acc_abs_5],
    'characterise_eval_acc_test_acc_abs_5': summary_characterise['eval_acc'][pruning_itr_characterise_test_acc_abs_5],
    'characterise_eval_loss_test_acc_abs_5': summary_characterise['eval_loss'][pruning_itr_characterise_test_acc_abs_5],
    'characterise_training_iterations_test_acc_abs_5': summary_characterise['training_iterations'][pruning_itr_characterise_test_acc_abs_5],
    'characterise_pruning_iterations_test_acc_abs_5': pruning_itr_characterise_test_acc_abs_5,
    }
df.to_csv(args.arch_caffe + '/results/summary_database.csv')
