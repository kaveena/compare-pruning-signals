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
from plot_utils.plot_util import *
import pandas as pd

ylim_sensitivity = dict()
ylim_sensitivity['LeNet-5'] = {'CIFAR10': 25}
ylim_sensitivity['CIFAR10'] = {'CIFAR10': 45}
ylim_sensitivity['ResNet-20'] = {'CIFAR10': 15, 'CIFAR100': 5}
ylim_sensitivity['NIN'] = {'CIFAR10': 35, 'CIFAR100': 45}
ylim_sensitivity['AlexNet'] = {'CIFAR10': 65, 'CIFAR100': 60, 'IMAGENET32x32': 55}

ylim_retrain = dict()
ylim_retrain['LeNet-5'] = {'CIFAR10': 85}
ylim_retrain['CIFAR10'] = {'CIFAR10': 90}
ylim_retrain['ResNet-20'] = {'CIFAR10': 25, 'CIFAR100': 10}
ylim_retrain['NIN'] = {'CIFAR10': 75, 'CIFAR100': 60}
ylim_retrain['AlexNet'] = {'CIFAR10': 75, 'CIFAR100': 65, 'IMAGENET32x32': 55}

ylim_characterise = dict()
ylim_characterise['LeNet-5'] = {'CIFAR10': 85}
ylim_characterise['CIFAR10'] = {'CIFAR10': 75}
ylim_characterise['ResNet-20'] = {'CIFAR10': 25, 'CIFAR100': 10}
ylim_characterise['NIN'] = {'CIFAR10': 75, 'CIFAR100': 60}
ylim_characterise['AlexNet'] = {'CIFAR10': 75, 'CIFAR100': 65, 'IMAGENET32x32': 55}

networks_dict_1 = { 'LeNet-5': ['CIFAR10'],
                  'CIFAR10': ['CIFAR10'],
                  'ResNet-20': ['CIFAR10', 'CIFAR100'],
                  'NIN': ['CIFAR10', 'CIFAR100'],
                  'AlexNet': ['CIFAR10', 'CIFAR100', 'IMAGENET32x32']}

networks_dict_2 = { 'LeNet-5': ['CIFAR10'],
                  'CIFAR10': ['CIFAR10'],
                  'ResNet-20': ['CIFAR10', 'CIFAR100'],
                  'NIN': ['CIFAR10', 'CIFAR100'],
                  'AlexNet': ['CIFAR10', 'CIFAR100']}

networks_dict_3 = { 'LeNet-5': ['CIFAR10'],
                  'CIFAR10': ['CIFAR10'],
                  'ResNet-20': ['CIFAR10']}

max_sparsity = {'LeNet-5-CIFAR10':        84.3,
                'CIFAR10-CIFAR10':        70.6,
                'ResNet-20-CIFAR10':      22.1,
                'NIN-CIFAR10':            0,
                'AlexNet-CIFAR10':        0,
                'ResNet-20-CIFAR100':     0,
                'NIN-CIFAR100':           0,
                'AlexNet-CIFAR100':       0,
                'AlexNet-IMAGENET32x32':  0}

parser = argparse.ArgumentParser()
parser.add_argument('--sparsity-drop', action='store', default=10.0)

args = parser.parse_args()  

df = pd.read_csv('summary_characterise.csv')
df_noretraining = pd.read_csv('summary_no_retraining.csv')
ylim = ylim_characterise
networks_dict = networks_dict_3

df['saliency_name'] = df['saliency_input'] + df['pointwise_saliency'] + df['saliency_reduction'] + df['saliency_scaling']
df_noretraining['saliency_name'] = df_noretraining['saliency_input'] + df_noretraining['pointwise_saliency'] + df_noretraining['saliency_reduction'] + df_noretraining['saliency_scaling']

all_metrics = ['network', 'dataset', 'pointwise_saliency', 'saliency_input', 'saliency_reduction', 'saliency_scaling']

df['saliency_computation'] = 2
df.loc[(df['saliency_input'] == 'weight') & (df['pointwise_saliency'] == 'average_input'), 'saliency_computation'] = 0
df.loc[(df['saliency_input'] == 'activation') & (df['pointwise_saliency'] == 'average_input'), 'saliency_computation'] = 1
df.loc[(df['pointwise_saliency'] == 'hessian_diag_approx1'), 'saliency_computation'] = 3
df.loc[(df['pointwise_saliency'] == 'taylor_2nd_approx1'), 'saliency_computation'] = 3

df['total_cost'] = (df['pruning_steps_2'] * df['saliency_computation']) + 2*df['retraining_steps_2']

for network in networks_dict.keys():
  datasets = networks_dict[network]
  fig, axs = plt.subplots(len(datasets), 1, figsize=(len(datasets)*6,10))
  fig.add_subplot(111, frameon=False)
  plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
  for i_dataset in range(len(datasets)):
    dataset = datasets[i_dataset]
    selected_df = df[(df['network'] == network) & (df['dataset'] == dataset)].sort_values('saliency_name')
    selected_df_noretraining = df_noretraining[(df_noretraining['network'] == network) & (df_noretraining['dataset'] == dataset)].sort_values('saliency_name')
    valid_idx=selected_df['sparsity_mean'] >= max_sparsity[network+'-CIFAR10'] - args.sparsity_drop
    retraining_steps = selected_df[valid_idx]['retraining_steps_2']
    sparsity_noretraining = selected_df_noretraining[valid_idx]['sparsity_mean']
    if len(datasets) > 1:
      axs[i_dataset].scatter(sparsity_noretraining, retraining_steps)

    else:
      axs.scatter(sparsity_noretraining, retraining_steps)
    if len(datasets) > 1:
      axs[i_dataset].set_ylabel('Retraining steps to reach {:.1f} % weights'.format(max_sparsity[network+'-CIFAR10']-10))
      axs[i_dataset].set_title(dataset)
    else:
      axs.set_ylabel('Retraining steps to reach {:.1f} % weights'.format(max_sparsity[network+'-CIFAR10']-10))
      axs.set_title(dataset)
  plt.xlabel('Weights removed (%) with no retraining')
  fig.suptitle(network)
  fig.tight_layout(pad=3.0)
  fig.savefig('graphs/retraining_steps_' + network + '.pdf') 

for network in networks_dict.keys():
  datasets = networks_dict[network]
  fig, axs = plt.subplots(len(datasets), 1, figsize=(10, len(datasets)*5))
  fig.add_subplot(111, frameon=False)
  plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
  for i_dataset in range(len(datasets)):
    dataset = datasets[i_dataset]
    selected_df = df[(df['network'] == network) & (df['dataset'] == dataset)].sort_values('saliency_name')
    selected_df_noretraining = df_noretraining[(df_noretraining['network'] == network) & (df_noretraining['dataset'] == dataset)].sort_values('saliency_name')
    valid_idx=selected_df['sparsity_mean'] >= max_sparsity[network+'-CIFAR10'] - args.sparsity_drop
    retraining_steps = selected_df[valid_idx]['total_cost']
    sparsity_noretraining = selected_df_noretraining[valid_idx]['sparsity_mean']
    if len(datasets) > 1:
      axs[i_dataset].scatter(sparsity_noretraining, retraining_steps)

    else:
      axs.scatter(sparsity_noretraining, retraining_steps)
    if len(datasets) > 1:
      axs[i_dataset].set_ylabel('Total steps to reach {:.1f} % weights'.format(max_sparsity[network+'-CIFAR10']-10))
      axs[i_dataset].set_title(dataset)
    else:
      axs.set_ylabel('Total steps to reach {:.1f} % weights'.format(max_sparsity[network+'-CIFAR10']-10))
      axs.set_title(dataset)
  plt.xlabel('Weights removed (%) with no retraining')
  fig.suptitle(network)
  fig.tight_layout(pad=3.0)
  fig.savefig('graphs/total_steps_' + network + '.pdf') 
