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

def get_y_point(x, y, x_point):
  y_point = scipy.interpolate.interp1d(x, y)(x_point)
  return y_point

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
  dfObj = pd.DataFrame(columns=[
  'network', 
  'dataset', 
  'pointwise_saliency', 
  'saliency_input', 
  'saliency_reduction', 
  'saliency_scaling', 
  'iteration',
  'sparsity', 
  'retraining_steps', 
  'retraining_steps_2', 
  'pruning_steps', 
  'pruning_steps_2' 
  ])
else:
  dfObj = pd.DataFrame(columns=[
  'network', 
  'dataset', 
  'pointwise_saliency', 
  'saliency_input', 
  'saliency_reduction', 
  'saliency_scaling',
  'iteration',
  'sparsity'
  ])

for network in networks_dict.keys():
  for dataset in networks_dict[network]:
    all_saliencies = []
    summary_pruning_strategies = dict()
    #Single heuristics
    for saliency_input in inputs:
      for pointwise_saliency in pointwise_saliencies:
        for saliency_reduction in reductions:
          for saliency_scaling in scalings:
            skip=False
            if pointwise_saliency == "hessian_diag_approx2":
              if saliency_reduction == "none_norm" or saliency_reduction == "l1_norm" or saliency_reduction == "abs_sum_norm":
                skip=True
            if pointwise_saliency == "taylor" and saliency_input == "weight":
              if saliency_reduction =="none_norm" or saliency_reduction == "abs_sum_norm" or saliency_reduction == "sqr_sum_norm":
                if saliency_scaling == "no_normalisation" or saliency_scaling == "l1_normalisation" or saliency_scaling == "l2_normalisation" or saliency_scaling == "weights_removed" :
                  skip=True
            if pointwise_saliency == "apoz":
              if saliency_reduction == "abs_sum_norm" or saliency_reduction == "l2_norm" or saliency_reduction == "l1_norm":
                skip=True
            if not skip:
              all_saliencies.append(saliency_input+'-'+ pointwise_saliency + '-' + saliency_reduction + '-' + saliency_scaling)

    saliencies = list(all_saliencies)
    filename_prefix = network + '-' + dataset + '/results/prune/summary_'
    if args.retrain:
      filename_prefix = filename_prefix + 'retrain_'
    elif args.characterise:
      filename_prefix = filename_prefix + 'characterise_'
    else:
      filename_prefix = filename_prefix + 'sensitivity_'
    if args.input:
      filename_prefix = filename_prefix + 'input_channels_'

    for saliency in all_saliencies:
      summary_file = filename_prefix + saliency + '_caffe_iter1.npy'
      if os.path.isfile(summary_file):
        summary_pruning_strategies[saliency] = dict(np.load(summary_file, allow_pickle=True).item()) 
      else:
        print(summary_file+'was not found')
        saliencies.remove(saliency)

    all_pruning = list(set(saliencies))

    x_signals = np.arange(0, 101, 1)

    for pointwise_saliency in pointwise_saliencies:  
      for saliency_input in inputs:
        for saliency_reduction in reductions:
          for saliency_scaling in scalings:
            saliency = '-'.join([saliency_input, pointwise_saliency, saliency_reduction, saliency_scaling])
            if not(saliency in all_saliencies):
              continue
            num_iter_ = 0
            y_mean = np.zeros(len(x_signals))
            y_sqr_mean = np.zeros(len(x_signals))
            y_std = np.zeros(len(x_signals))
            if args.characterise:
              x_itr = np.array(range(0, total_channels[network][dataset] + 2))
            x_min = 100
            x_max = 0
            for i in range(1, args.iterations + 1):
              summary_file = filename_prefix + saliency + '_caffe_iter' + str(i) + '.npy'
              if os.path.isfile(summary_file):
                summary = dict(np.load(summary_file, allow_pickle=True).item())
                if args.characterise:
                  f =  lambda x : 0 if x is None else len(x)
                  retraining_steps = np.array([f(x) for x in summary['retraining_acc']])
                  retraining_steps = np.cumsum(retraining_steps)
                summary['sparsity'] = 100 * (1 - (summary['conv_param'] + summary['fc_param']) / float(summary['initial_conv_param'] + summary['initial_fc_param']))
                y_inter = scipy.interpolate.interp1d(np.hstack([0.0, summary['sparsity'], 100]), np.hstack([accuracies[network + '-' + dataset], summary['test_acc'], 10]))(x_signals)
                if args.characterise:
                  y_inter_steps = scipy.interpolate.interp1d(np.hstack([0.0, summary['sparsity'], 100]), np.hstack([0, retraining_steps, retraining_steps[-1]]))(x_signals)
                  y_inter_itr = scipy.interpolate.interp1d(np.hstack([0.0, summary['sparsity'], 100]), x_itr)(x_signals)
                min_test_acc = accuracies[network + '-' + dataset] - args.acc_drop
                sparsity = get_x_point(x_signals, y_inter, min_test_acc)
                if args.characterise:
                  steps_point = get_y_point(x_signals, y_inter_steps, min_test_acc)
                  steps_point_2 = get_y_point(x_signals, y_inter_steps, max_sparsity[network+'-'+dataset] - args.sparsity_drop)
                  itr_point = get_y_point(x_signals, y_inter_itr, min_test_acc)
                  itr_point_2 = get_y_point(x_signals, y_inter_itr, max_sparsity[network+'-'+dataset] - args.sparsity_drop)
                  dfObj = dfObj.append({
                  'network': network, 
                  'dataset': dataset, 
                  'pointwise_saliency': pointwise_saliency, 
                  'saliency_input': saliency_input, 
                  'saliency_reduction': saliency_reduction, 
                  'saliency_scaling': saliency_scaling,
                  'iteration': i,
                  'sparsity': sparsity, 
                  'retraining_steps': steps_point, 
                  'retraining_steps_2': steps_point_2, 
                  'pruning_steps': itr_point, 
                  'pruning_steps_2': itr_point_2 
                  }, ignore_index=True)
                else:
                  dfObj = dfObj.append({
                  'network': network, 
                  'dataset': dataset, 
                  'pointwise_saliency': pointwise_saliency, 
                  'saliency_input': saliency_input, 
                  'saliency_reduction': saliency_reduction, 
                  'saliency_scaling': saliency_scaling,
                  'iteration': i,
                  'sparsity': sparsity
                  }, ignore_index=True)

if args.characterise:
  dfObj.to_csv('characterise.csv')
elif args.retrain:
  dfObj.to_csv('retrain.csv')
else:
  dfObj.to_csv('no_retraining.csv')
