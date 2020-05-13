import scipy
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from plot_utils.plot_util import *
from functools import reduce
import sys
import gc
from utils import *
import tempfile
import scipy.interpolate

global_initial_test_acc = 0.0
global_num_test_acc_sample = 0

def get_error(x, y_mean, y_std, y_point):
  y_b1 = y_mean - 2*y_std
  y_b2 = y_mean + 2*y_std
  x_point = scipy.interpolate.interp1d(y_mean, x)(y_point)
  x_error_1_itr = scipy.interpolate.interp1d(y_b1, x)(y_point)
  x_error_2_itr = scipy.interpolate.interp1d(y_b2, x)(y_point)
  x_error = max(np.abs(x_error_1_itr - x_point), np.abs(x_error_2_itr - x_point))
  return x_point, x_error

def plot_trend(metric1, metric2):
  filename_prefix = args.arch + '-' + args.dataset + '/results/graph/'
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
      fig = plt.figure(figsize=(8.0, 6.0))
      ax = plt.subplot(111)
      for norm in norms:
        for normalisation in normalisations:
          method = saliency_input + '-' + saliency + '-' + norm + '-' + normalisation
          if method in summary_pruning_strategies.keys():
            summary = summary_pruning_strategies[method]
          else:
            continue
          # sparsity at 1% test_acc tolerance
          x = summary[metric1][::args.test_interval]
          y = summary[metric2][::args.test_interval]
          y_std = summary[metric2+'_std'][::args.test_interval]
          y_point = global_initial_test_acc - 1.0
          x_point, x_error = get_error(x, y, y_std, global_initial_test_acc - 1.0)
          ax.plot(x, y, label=convert_label(norm + '-' + normalisation), color = get_color(norm, normalisation))
          ax.fill_between(x, y - 2*y_std, y + 2*y_std, color = get_color(norm, normalisation), alpha=0.2)
          ax.plot(x_point, y_point, color = get_color(norm, normalisation), marker = 'v')
          print(saliency, saliency_input, norm, normalisation, x_point, x_error)
          gc.collect()
      ax.set_title('Pruning Sensitivity for ' + args.arch + ', f(x) = '+ convert_label(saliency) + ' using ' +saliency_input + 's' )
      ax.set_xlabel(metric1)
      ax.set_ylabel(metric2)
      chartBox = ax.get_position()
      ax.legend(loc = 'upper center', bbox_to_anchor=(1.2, 1.0) , ncol=1, prop = {'size': 6})
      ax.grid()
      plt.savefig(filename_prefix + saliency_input + '-' + saliency + '-' + metric1 + '-' + metric2 + filename_suffix, bbox_inches='tight') 
      plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--arch', action='store', default='LeNet-5')
parser.add_argument('--dataset', action='store', default='CIFAR10')
parser.add_argument('--retrain', action='store_true', default=False)
parser.add_argument('--characterise', action='store_true', default=False)
parser.add_argument('--input', action='store_true', default=False)
parser.add_argument('--test-interval', type=int, action='store', default=1)
parser.add_argument('--iterations', type=int, action='store', default=1)
parser.add_argument('--metric1', action='store', default='sparsity')
parser.add_argument('--metric2', action='store', default='test_acc')

args = parser.parse_args()  

stop_acc = 10.01

caffe_methods = ['hessian_diag_approx1', 'hessian_diag_approx2', 'taylor_2nd_approx1', 'taylor_2nd_approx2', 'taylor', 'average_input', 'average_gradient']
#python_methods = ['apoz']
python_methods = ['']
norms = ['l1_norm', 'l2_norm', 'none_norm', 'abs_sum_norm', 'sqr_sum_norm']
normalisations = ['no_normalisation', 'l1_normalisation', 'l2_normalisation', 'l0_normalisation_adjusted', 'weights_removed']
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

methods = list(all_methods)
filename_prefix = args.arch + '-' + args.dataset + '/results/prune/summary_'
if args.retrain:
  filename_prefix = filename_prefix + 'retrain_'
elif args.characterise:
  filename_prefix = filename_prefix + 'characterise_'
else:
  filename_prefix = filename_prefix + 'sensitivity_'
if args.input:
  filename_prefix = filename_prefix + 'input_channels_'

for method in all_methods:
  summary_file = filename_prefix + method + '_caffe_iter1.npy'
  if os.path.isfile(summary_file):
    summary_pruning_strategies[method] = dict(np.load(summary_file, allow_pickle=True).item()) 
  else:
    print(summary_file+'was not found')
    methods.remove(method)

all_pruning = list(set(methods))

for i in range(len(all_pruning)):
  summary = summary_pruning_strategies[all_pruning[i]]
  global_initial_test_acc += summary['initial_test_acc']
  global_num_test_acc_sample += 1
global_initial_test_acc /= float(global_num_test_acc_sample)

x_signals = np.arange(0, 101, 1)

for method in all_pruning:
  num_iter_ = 0
  y_mean = np.zeros(len(x_signals))
  y_sqr_mean = np.zeros(len(x_signals))
  y_std = np.zeros(len(x_signals))
  for i in range(1, args.iterations + 1):
    summary_file = filename_prefix + method + '_caffe_iter' + str(i) + '.npy'
    if os.path.isfile(summary_file):
      num_iter_ += 1
      summary = dict(np.load(summary_file, allow_pickle=True).item())
      summary['conv_sparsity'] = 100 *(1 - summary['conv_param'] / float(summary['initial_conv_param']))
      summary['sparsity'] = 100 * (1 - (summary['conv_param'] + summary['fc_param']) / float(summary['initial_conv_param'] + summary['initial_fc_param']))
      y_inter = scipy.interpolate.interp1d(np.hstack([0.0, summary['sparsity'], 100]), np.hstack([global_initial_test_acc, summary['test_acc'], 10]))(x_signals)
      y_mean += y_inter
      y_sqr_mean += y_inter ** 2
  y_mean = y_mean / num_iter_
  y_sqr_mean /= num_iter_
  y_std = (y_sqr_mean - (y_mean**2))
  y_std = np.piecewise(y_std, [np.abs(y_std) < 1e-10, np.abs(y_std) > 1e-10], [0, lambda x: x])
  y_std = y_std ** 0.5
  summary_pruning_strategies[method]['sparsity'] = x_signals
  summary_pruning_strategies[method]['test_acc'] = y_mean 
  summary_pruning_strategies[method]['test_acc_std'] = y_std

plot_trend(args.metric1, args.metric2)
