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

parser = argparse.ArgumentParser()
parser.add_argument('--retrain', action='store_true', default=False)

args = parser.parse_args()  

df = pd.read_csv('summary_no_retraining.csv')


networks_dict = { 'LeNet-5': ['CIFAR10'],
                  'CIFAR10': ['CIFAR10'],
                  'ResNet-20': ['CIFAR10', 'CIFAR100'],
                  'NIN': ['CIFAR10', 'CIFAR100'],
                  'AlexNet': ['CIFAR10', 'CIFAR100', 'IMAGENET32x32']}


all_metrics = ['network', 'dataset', 'pointwise_saliency', 'saliency_input', 'saliency_reduction', 'saliency_scaling']

df['scaling_baseline'] = np.zeros(len(df))
df['reduction_baseline'] = np.zeros(len(df))

for index, row in df.iterrows():
  valid_idx = np.ones(len(df), dtype=bool)
  valid_idx = (df['network'] == row['network']) & (df['dataset'] == row['dataset'])
  if (row['pointwise_saliency'] == 'taylor') and (row['saliency_input'] == 'weight') and (row['saliency_reduction'] == 'none_norm' or row['saliency_reduction'] == 'abs_sum_norm' or row['saliency_reduction'] == 'sqr_sum_norm') :
    valid_idx = valid_idx & (df['pointwise_saliency'] == row['pointwise_saliency']) & (df['saliency_input'] == 'activation') & (df['saliency_reduction'] == row['saliency_reduction']) & (df['saliency_scaling'] == 'no_normalisation')
  else:
    valid_idx = valid_idx & (df['pointwise_saliency'] == row['pointwise_saliency']) & (df['saliency_input'] == row['saliency_input']) & (df['saliency_reduction'] == row['saliency_reduction']) & (df['saliency_scaling'] == 'no_normalisation')
  baseline_df = df[valid_idx]
  if len(baseline_df) != 1:
    print("no or more than 1 baseline found")
  else:
    baseline_row = baseline_df.iloc[0]
    df.loc[index, 'scaling_baseline'] = baseline_row['sparsity_mean']

for index, row in df.iterrows():
  valid_idx = np.ones(len(df), dtype=bool)
  valid_idx = (df['network'] == row['network']) & (df['dataset'] == row['dataset'])
  if (row['pointwise_saliency'] == 'taylor') and (row['saliency_input'] == 'weight') and (row['saliency_reduction'] == 'l1_norm' or row['saliency_reduction'] == 'l2_norm') and (row['saliency_scaling'] != 'l0_normalisation_adjusted') :
    valid_idx = valid_idx & (df['pointwise_saliency'] == row['pointwise_saliency']) & (df['saliency_input'] == 'activation') & (df['saliency_scaling'] == row['saliency_scaling']) & (df['saliency_reduction'] == 'none_norm')
  elif row['pointwise_saliency'] == 'hessian_diag_approx2':
    valid_idx = valid_idx & (df['pointwise_saliency'] == 'taylor') & (df['saliency_input'] == row['saliency_input']) & (df['saliency_scaling'] == row['saliency_scaling']) & (df['saliency_reduction'] == 'l2_norm')
  else:
    valid_idx = valid_idx & (df['pointwise_saliency'] == row['pointwise_saliency']) & (df['saliency_input'] == row['saliency_input']) & (df['saliency_scaling'] == row['saliency_scaling']) & (df['saliency_reduction'] == 'none_norm')
  baseline_df = df[valid_idx]
  if len(baseline_df) != 1:
    print("no or more than 1 baseline found", row)
  else:
    baseline_row = baseline_df.iloc[0]
    df.loc[index, 'reduction_baseline'] = baseline_row['sparsity_mean']

norms = ['l1_norm', 'l2_norm', 'none_norm', 'abs_sum_norm', 'sqr_sum_norm']
normalisations = ['no_normalisation', 'l1_normalisation', 'l2_normalisation', 'l0_normalisation_adjusted', 'weights_removed']
#for network in networks_dict.keys():
#  for dataset in networks_dict[network]:
#    for reduction in ['l1_norm', 'l2_norm', 'abs_sum_norm', 'sqr_sum_norm']:
#      print(network, dataset, reduction, df[(df['network'] == network) & (df['dataset'] == dataset) & (df['saliency_reduction'] == reduction)]['sparsity_mean'].mean())

df['reduction-scaling'] = df['saliency_reduction'] + '-' + df['saliency_scaling']

for network in networks_dict.keys():
  datasets = networks_dict[network]
  fig, axs = plt.subplots(len(datasets), 1, figsize=(len(datasets)*5,10))
  fig.add_subplot(111, frameon=False)
  plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
  for i_dataset in range(len(datasets)):
    dataset = datasets[i_dataset]
    selected_df = df[(df['network'] == network) & (df['dataset'] == dataset) & (df['saliency_input'] == 'weight') & (df['pointwise_saliency'] == 'average_input')]
    y_bar = selected_df['sparsity_mean'].to_list()
    y_bar_err = selected_df['sparsity_err'].to_list()
    y_scaling = selected_df['saliency_scaling'].to_list()
    x_bar = selected_df['reduction-scaling'].to_list()
    y_bar_color = [get_color_normalisation(i) for i in y_scaling]
    if len(datasets) > 1:
      #barlist = axs[i_dataset].bar([convert_label(i_x) for i_x in x_bar], y_bar, yerr=y_bar_err, color = y_bar_color, alpha=0.99)
      barlist = axs[i_dataset].bar([convert_label(i_x) for i_x in x_bar], y_bar, color = y_bar_color, alpha=0.99)
    else:
      #barlist = axs.bar([convert_label(i_x) for i_x in x_bar], y_bar, yerr=y_bar_err, color = y_bar_color, alpha=0.99)
      barlist = axs.bar([convert_label(i_x) for i_x in x_bar], y_bar, color = y_bar_color, alpha=0.99)
    if len(datasets) > 1:
      for tick in axs[i_dataset].get_xticklabels():
        tick.set_rotation(90)
#      axs[i_dataset].set_xlabel("Reduction and Scaling used")
#      axs[i_dataset].set_ylabel("Convolution weights removed ($\%$)")
      axs[i_dataset].set_title(dataset)
    else:
      for tick in axs.get_xticklabels():
        tick.set_rotation(90)
#      axs.set_xlabel("Reduction and Scaling used")
#      axs.set_ylabel("Convolution weights removed ($\%$)")
      axs.set_title(dataset)
    for i_bar in range(len(barlist)):
      bar = barlist[i_bar]
      bar.set_hatch(get_norm_hatch(x_bar[i_bar].split('-')[0]))
  plt.xlabel("Reduction and Scaling used", labelpad=100)
  plt.ylabel("Convolution weights removed ($\%$)")
  fig.suptitle(network)
  fig.tight_layout(pad=3.0)
  fig.savefig('graphs/weights_only_' + network + '.pdf') 
