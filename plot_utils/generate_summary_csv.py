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
import pandas as pd
from plot_utils.plot_data_util import *

caffe_methods = ['hessian_diag_approx1', 'hessian_diag_approx2', 'taylor_2nd_approx1', 'taylor_2nd_approx2', 'taylor', 'average_input', 'average_gradient']
#python_methods = ['apoz']
python_methods = ['']
norms = ['l1_norm', 'l2_norm', 'none_norm', 'abs_sum_norm', 'sqr_sum_norm']
normalisations = ['no_normalisation', 'l1_normalisation', 'l2_normalisation', 'l0_normalisation_adjusted', 'weights_removed']
saliency_inputs = ['weight', 'activation']

#max_sparsity = {'LeNet-5-CIFAR10':        84.3,
#                'CIFAR10-CIFAR10':        70.6,
#                'ResNet-20-CIFAR10':      22.1,
#                'NIN-CIFAR10':            10.0,
#                'AlexNet-CIFAR10':        0,
#                'ResNet-20-CIFAR100':     10.0,
#                'NIN-CIFAR100':           10.0,
#                'AlexNet-CIFAR100':       0,
#                'AlexNet-IMAGENET32x32':  0}

def get_x_point(x, y, y_point):
  valid_y = np.where(y > y_point)
  if len(valid_y) == 0:
    x_point = x[0]
  elif len(valid_y) == len(y):
    x_point = x[-1]
  else:
    max_i = valid_y[0][-1]
    x_point = scipy.interpolate.interp1d([y[max_i+1], y[max_i]], [x[max_i+1], x[max_i]])(y_point)
  return x_point

def get_x(x, y_mean, y_std, y_point):
  y_b1 = y_mean - 2*y_std
  y_b2 = y_mean + 2*y_std
  # find max x for y > y_point
  x_point = get_x_point(x, y_mean, y_point)
  x_error_1_itr = get_x_point(x, y_b1, y_point)
  x_error_2_itr = get_x_point(x, y_b2, y_point)
  x_error = max(np.abs(x_error_1_itr - x_point), np.abs(x_error_2_itr - x_point))
  return x_point, x_error

def get_y(x, y_mean, y_std, x_point):
  y_b1 = y_mean - 2*y_std
  y_b2 = y_mean + 2*y_std
  y_point = scipy.interpolate.interp1d(x, y_mean)(x_point)
  y_point_std = scipy.interpolate.interp1d(x, y_std)(x_point)
  y_error = 2*y_point_std
  return y_point, y_error

parser = argparse.ArgumentParser()
parser.add_argument('--retrain', action='store_true', default=False)
parser.add_argument('--characterise', action='store_true', default=False)
parser.add_argument('--input', action='store_true', default=False)
parser.add_argument('--test-interval', type=int, action='store', default=1)
parser.add_argument('--iterations', type=int, action='store', default=8)
parser.add_argument('--metric1', action='store', default='sparsity')
parser.add_argument('--metric2', action='store', default='test_acc')
parser.add_argument('--acc-drop', action='store', default=5.0)
parser.add_argument('--sparsity-drop', action='store', default=5.0)

args = parser.parse_args()  

if args.retrain:
  networks_dict = networks_dict_2
elif args.characterise:
  networks_dict = networks_dict_3
else:
  networks_dict = networks_dict_1

if args.characterise:
  dfObj = pd.DataFrame(columns=['pointwise_saliency', 'saliency_input', 'saliency_reduction', 'saliency_scaling', 'network', 'dataset', 'sparsity_mean', 'sparsity_err'])
else:
  dfObj = pd.DataFrame(columns=['pointwise_saliency', 'saliency_input', 'saliency_reduction', 'saliency_scaling', 'network', 'dataset', 'sparsity_mean', 'sparsity_err', 'retraining_steps', 'retraining_steps_err', 'pruning_steps', 'pruning_steps_err', 'pruning_steps_err', 'pruning_steps_err_2'])

for network in networks_dict.keys():
  for dataset in networks_dict[network]:
    all_methods = []
    summary_pruning_strategies = dict()
    #Single heuristics
    for saliency_input in saliency_inputs:
      for method in caffe_methods:
        for norm in norms:
          for normalisation in normalisations:
            skip=False
            if method == "hessian_diag_approx2":
              if norm == "none_norm" or norm == "l1_norm" or norm == "abs_sum_norm":
                skip=True
            if method == "taylor" and saliency_input == "weight":
              if norm =="none_norm" or norm == "abs_sum_norm" or norm == "sqr_sum_norm":
                if normalisation == "no_normalisation" or normalisation == "l1_normalisation" or normalisation == "l2_normalisation" or normalisation == "weights_removed" :
                  skip=True
            if method == "apoz":
              if norm == "abs_sum_norm" or norm == "l2_norm" or norm == "l1_norm":
                skip=True
            if not skip:
              all_methods.append(saliency_input+'-'+ method + '-' + norm + '-' + normalisation)

    methods = list(all_methods)
    filename_prefix = network + '-' + dataset + '/results/prune/summary_'
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

    x_signals = np.arange(0, 101, 1)

    for method in all_pruning:
      num_iter_ = 0
      y_mean = np.zeros(len(x_signals))
      y_sqr_mean = np.zeros(len(x_signals))
      y_std = np.zeros(len(x_signals))
      if args.characterise:
        y_mean_steps= np.zeros(len(x_signals))
        y_sqr_mean_steps = np.zeros(len(x_signals))
        y_std_steps = np.zeros(len(x_signals))
        y_mean_itr= np.zeros(len(x_signals))
        y_sqr_mean_itr = np.zeros(len(x_signals))
        y_std_itr = np.zeros(len(x_signals))
        x_itr = np.array(range(0, total_channels[network][dataset] + 2))
      x_min = 100
      x_max = 0
      for i in range(1, args.iterations + 1):
        summary_file = filename_prefix + method + '_caffe_iter' + str(i) + '.npy'
        if os.path.isfile(summary_file):
          num_iter_ += 1
          summary = dict(np.load(summary_file, allow_pickle=True).item())
          if args.characterise:
            f =  lambda x : 0 if x is None else len(x)
            retraining_steps = np.array([f(x) for x in summary['retraining_acc']])
            retraining_steps = np.cumsum(retraining_steps)
#          summary['conv_sparsity'] = 100 *(1 - summary['conv_param'] / float(summary['initial_conv_param']))
          summary['sparsity'] = 100 * (1 - (summary['conv_param'] + summary['fc_param']) / float(summary['initial_conv_param'] + summary['initial_fc_param']))
          y_inter = scipy.interpolate.interp1d(np.hstack([0.0, summary['sparsity'], 100]), np.hstack([accuracies[network + '-' + dataset], summary['test_acc'], 10]))(x_signals)
          y_mean += y_inter
          y_sqr_mean += y_inter ** 2
          y_point = accuracies[network + '-' + dataset] - args.acc_drop
          x_point = get_x_point(x_signals, y_inter, y_point)
          x_min = x_point if x_point < x_min else x_min
          x_max = x_point if x_point > x_max else x_max
          if args.characterise:
            y_inter_steps = scipy.interpolate.interp1d(np.hstack([0.0, summary['sparsity'], 100]), np.hstack([0, retraining_steps, retraining_steps[-1]]))(x_signals)
            y_mean_steps += y_inter_steps
            y_sqr_mean_steps += y_inter_steps ** 2
            y_inter_itr = scipy.interpolate.interp1d(np.hstack([0.0, summary['sparsity'], 100]), x_itr)(x_signals)
            y_mean_itr += y_inter_itr
            y_sqr_mean_itr += y_inter_itr ** 2
      y_mean = y_mean / num_iter_
      y_sqr_mean /= num_iter_
      y_std = (y_sqr_mean - (y_mean**2))
      y_std = np.piecewise(y_std, [np.abs(y_std) < 1e-10, np.abs(y_std) > 1e-10], [0, lambda x: x])
      y_std = y_std ** 0.5
      summary_pruning_strategies[method]['sparsity'] = x_signals
      summary_pruning_strategies[method]['test_acc'] = y_mean 
      summary_pruning_strategies[method]['test_acc_std'] = y_std
      summary_pruning_strategies[method]['sparsity_min'] = x_min
      summary_pruning_strategies[method]['sparsity_max'] = x_max
      if args.characterise:
        y_mean_steps = y_mean_steps / num_iter_
        y_sqr_mean_steps /= num_iter_
        y_std_steps = (y_sqr_mean_steps - (y_mean_steps**2))
        y_std_steps = np.piecewise(y_std_steps, [np.abs(y_std_steps) < 1e-10, np.abs(y_std_steps) > 1e-10], [0, lambda x: x])
        y_std_steps = y_std_steps ** 0.5
        summary_pruning_strategies[method]['retraining_steps'] = y_mean_steps
        summary_pruning_strategies[method]['retraining_steps_std'] = y_std_steps
        y_mean_itr = y_mean_itr / num_iter_
        y_sqr_mean_itr /= num_iter_
        y_std_itr = (y_sqr_mean_itr - (y_mean_itr**2))
        y_std_itr = np.piecewise(y_std_itr, [y_std_itr < 0, y_std_itr > 0], [0, lambda x: x])
#        y_std_itr = np.piecewise(y_std_itr, [np.abs(y_std_itr) < 1e-10, np.abs(y_std_itr) > 1e-10], [0, lambda x: x])
        y_std_itr = y_std_itr ** 0.5
        summary_pruning_strategies[method]['pruning_steps'] = y_mean_itr
        summary_pruning_strategies[method]['pruning_steps_std'] = y_std_itr
    for i_s in range(len(caffe_methods)):  
      for i_si in range(len(saliency_inputs)):
        pointwise_saliency = caffe_methods[i_s]
        saliency_input = saliency_inputs[i_si]
        y_bar = []
        x_bar = []
        y_bar_err = []
        y_bar_color = []
        for norm in norms:
          for normalisation in normalisations:
            method = saliency_input + '-' + pointwise_saliency + '-' + norm + '-' + normalisation
            if method in summary_pruning_strategies.keys():
              summary = summary_pruning_strategies[method]
            else:
              continue
            x = summary[args.metric1][::args.test_interval]
            y = summary[args.metric2][::args.test_interval]
            y_std = summary[args.metric2+'_std'][::args.test_interval]
            y_point = accuracies[network + '-' + dataset] - args.acc_drop
            x_point, x_error = get_x(x, y, y_std, y_point)
            if args.characterise:
              steps_point, steps_error = get_y(x, summary['retraining_steps'], summary['retraining_steps_std'], x_point)
              steps_point_2, steps_error_2 = get_y(x, summary['retraining_steps'], summary['retraining_steps_std'], max_sparsity[network+'-'+dataset] - args.sparsity_drop)
              itr_1, itr_err_1 = get_y(x, summary['pruning_steps'], summary['pruning_steps_std'], x_point)
              itr_2, itr_err_2 = get_y(x, summary['pruning_steps'], summary['pruning_steps_std'], max_sparsity[network+'-'+dataset] - args.sparsity_drop)
              dfObj = dfObj.append({
              'pointwise_saliency': pointwise_saliency, 
              'saliency_input': saliency_input, 
              'saliency_reduction': norm, 
              'saliency_scaling': normalisation, 
              'network': network, 
              'dataset': dataset, 
              'sparsity_mean': x_point, 
              'sparsity_min': summary['sparsity_min'], 
              'sparsity_max': summary['sparsity_max'], 
              'sparsity_err': x_error, 
              'retraining_steps': steps_point, 
              'retraining_steps_err': steps_error, 
              'retraining_steps_2': steps_point_2, 
              'retraining_steps_err_2': steps_error_2 , 
              'pruning_steps': itr_1, 
              'pruning_steps_err': itr_err_1, 
              'pruning_steps_2': itr_2, 
              'pruning_steps_err_2': itr_err_2
              }, ignore_index=True)
            else:
              dfObj = dfObj.append({
              'pointwise_saliency': pointwise_saliency, 
              'saliency_input': saliency_input, 
              'saliency_reduction': norm, 
              'saliency_scaling': normalisation, 
              'network': network, 
              'dataset': dataset, 
              'sparsity_mean': x_point, 
              'sparsity_min': summary['sparsity_min'], 
              'sparsity_max': summary['sparsity_max'], 
              'sparsity_err': x_error
              }, ignore_index=True)

if args.characterise:
  dfObj.to_csv('summary_characterise.csv')
elif args.retrain:
  dfObj.to_csv('summary_retrain.csv')
else:
  dfObj.to_csv('summary_no_retraining.csv')
