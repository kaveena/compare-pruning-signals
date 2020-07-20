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
from plot_utils.plot_data_util import *
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--sparsity-drop', action='store', default=5.0)

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
  fig, axs = plt.subplots(len(datasets), 1, figsize=(10, len(datasets)*6 - (len(datasets)-1)))
  fig.add_subplot(111, frameon=False)
  plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
  for i_dataset in range(len(datasets)):
    dataset = datasets[i_dataset]
    selected_df = df[(df['network'] == network) & (df['dataset'] == dataset)].sort_values('saliency_name')
    selected_df_noretraining = df_noretraining[(df_noretraining['network'] == network) & (df_noretraining['dataset'] == dataset)].sort_values('saliency_name')
    valid_idx=(selected_df['sparsity_mean'] >= max_sparsity[network+'-'+dataset] - args.sparsity_drop).to_numpy()
    retraining_steps = selected_df[valid_idx]['retraining_steps_2']
    sparsity_noretraining = selected_df_noretraining[valid_idx]['sparsity_mean']
    if len(datasets) > 1:
      axes_to_plot = axs[i_dataset]
    else:
      axes_to_plot = axs
    axes_to_plot.scatter(sparsity_noretraining, retraining_steps)
    print(network, dataset, scipy.stats.spearmanr(retraining_steps, sparsity_noretraining))
    axes_to_plot.set_ylabel('Retraining steps to remove \n {:.1f} % weights'.format(max_sparsity[network+'-'+dataset]- args.sparsity_drop), fontsize=20)
    axes_to_plot.set_title(dataset)
  plt.xlabel('Weights removed (%) with no retraining', fontsize=20)
#  fig.suptitle(network)
  fig.tight_layout(pad=3.0)
  plt.tick_params(axis='both', which='major', labelsize=20)
  plt.tick_params(axis='both', which='minor', labelsize=10)
  fig.savefig('graphs/retraining_steps_' + network + '.pdf') 

for network in networks_dict.keys():
  datasets = networks_dict[network]
  fig, axs = plt.subplots(len(datasets), 1, figsize=(10, len(datasets)*6 - (len(datasets)-1)))
  fig.add_subplot(111, frameon=False)
  plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
  for i_dataset in range(len(datasets)):
    dataset = datasets[i_dataset]
    selected_df = df[(df['network'] == network) & (df['dataset'] == dataset)].sort_values('saliency_name')
    selected_df_noretraining = df_noretraining[(df_noretraining['network'] == network) & (df_noretraining['dataset'] == dataset)].sort_values('saliency_name')
    valid_idx=(selected_df['sparsity_mean'] >= max_sparsity[network+'-'+dataset] - args.sparsity_drop).to_numpy()
    retraining_steps = selected_df[valid_idx]['total_cost']
    sparsity_noretraining = selected_df_noretraining[valid_idx]['sparsity_mean']
    if len(datasets) > 1:
      axes_to_plot = axs[i_dataset]
    else:
      axes_to_plot = axs
    axes_to_plot.scatter(sparsity_noretraining, retraining_steps)
    print(network, dataset, scipy.stats.spearmanr(retraining_steps, sparsity_noretraining))

    axes_to_plot.set_ylabel('Total steps to remove \n {:.1f} % weights'.format(max_sparsity[network+'-'+dataset]-args.sparsity_drop), fontsize=20)
    axes_to_plot.set_title(dataset)
  plt.xlabel('Weights removed (%) with no retraining')
  fig.suptitle(network)
  fig.tight_layout(pad=3.0)
  fig.savefig('graphs/total_steps_' + network + '.pdf')
  plt.close('all')
