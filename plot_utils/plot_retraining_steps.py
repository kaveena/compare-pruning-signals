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
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

parser = argparse.ArgumentParser()
parser.add_argument('--sparsity-drop', action='store', default=5.0)

args = parser.parse_args()  

df = pd.read_csv('summary_characterise.csv')
df_noretraining = pd.read_csv('summary_no_retraining.csv')
ylim = ylim_characterise
networks_dict = networks_dict_3

df['saliency_name'] = df['saliency_input'] + df['pointwise_saliency'] + df['saliency_reduction'] + df['saliency_scaling']
df_noretraining['saliency_name'] = df_noretraining['saliency_input'] + df_noretraining['pointwise_saliency'] + df_noretraining['saliency_reduction'] + df_noretraining['saliency_scaling']
df_noretraining = df_noretraining.rename(columns={'sparsity_mean': 'sparsity_noretraining'})

all_metrics = ['network', 'dataset', 'pointwise_saliency', 'saliency_input', 'saliency_reduction', 'saliency_scaling']

df['saliency_computation'] = 2
df.loc[(df['saliency_input'] == 'weight') & (df['pointwise_saliency'] == 'average_input'), 'saliency_computation'] = 0
df.loc[(df['saliency_input'] == 'activation') & (df['pointwise_saliency'] == 'average_input'), 'saliency_computation'] = 1
df.loc[(df['pointwise_saliency'] == 'hessian_diag_approx1'), 'saliency_computation'] = 3
df.loc[(df['pointwise_saliency'] == 'taylor_2nd_approx1'), 'saliency_computation'] = 3

df['info_cat'] = 'b'
df.loc[(df['saliency_input'] == 'weight') & (df['pointwise_saliency'] == 'average_input'), 'info_cat'] = 'darkgreen'
df.loc[(df['saliency_input'] == 'activation') & (df['pointwise_saliency'] == 'average_input'), 'info_cat'] = 'firebrick'
df.loc[(df['pointwise_saliency'] == 'average_gradient'), 'info_cat'] = 'olive'
df.loc[(df['pointwise_saliency'] == 'taylor'), 'info_cat'] = 'sandybrown'
df.loc[(df['pointwise_saliency'] == 'taylor_2nd_approx2'), 'info_cat'] = 'indigo'
df.loc[(df['pointwise_saliency'] == 'hessian_diag_approx2'), 'info_cat'] = 'teal'
df.loc[(df['pointwise_saliency'] == 'taylor_2nd_approx1'), 'info_cat'] = 'fuchsia'
df.loc[(df['pointwise_saliency'] == 'hessian_diag_approx1'), 'info_cat'] = 'pink'

df_max_sparsity = df[['network', 'dataset', 'sparsity_mean']].groupby(['network', 'dataset']).max().rename(columns={'sparsity_mean': 'max_sparsity'})

df = pd.merge(df, df_max_sparsity, on=['network', 'dataset'])

df = pd.merge(df, df_noretraining[['network', 'dataset', 'saliency_name', 'sparsity_noretraining']], on=['network', 'dataset', 'saliency_name'])

df=df[df['sparsity_mean'] >= df['max_sparsity'] - args.sparsity_drop]
df.loc[(df['sparsity_noretraining'] >= df['sparsity_mean'] - args.sparsity_drop), 'retraining_steps_2'] = 0

df['total_cost'] = (df['pruning_steps_2'] * df['saliency_computation'] * 2) + 2*df['retraining_steps_2']

#plot_legends = ['saliency cost: 0', 'saliency cost: $1 \\times N_{eval}$', 'saliency cost: $2 \\times N_{eval}$', 'saliency cost: $3 \\times N_{eval}$']
#plot_colors = ['g', 'r', 'b', 'c']
plot_legends = [
'$w$', 
'$a$', 
'$\\frac{d\\mathcal{L}}{dx}$', 
'$-x\\frac{d\\mathcal{L}}{dx}$', 
'$-x\\frac{d\\mathcal{L}}{dx} + \\frac{x^2}{2}\\frac{d^2\\mathcal{L}}{dx^2}_{app.2}$',
'$\\frac{x^2}{2}\\frac{d^2x\\mathcal{L}}{dx^2}_{app.2}$', 
'$-x\\frac{d\\mathcal{L}}{dx} + \\frac{x^2}{2}\\frac{d\\mathcal{L}}{dx^2}_{app.1}$', 
'$\\frac{x^2}{2}\\frac{d^2\\mathcal{L}}{dx^2}_{app.1}$' 
]
plot_colors = [
'darkgreen', 
'firebrick', 
'olive', 
'sandybrown', 
'indigo',
'teal', 
'fuchsia', 
'pink', 
]

legend_elements = []
legend_labels = []
for i_l in range(len(plot_legends)): 
  legend_elements.append(Line2D([0], [0], marker='o', color=plot_colors[i_l], label=plot_legends[i_l], markerfacecolor=plot_colors[i_l], linestyle='None')) 
  legend_labels.append(plot_legends[i_l])

networks_dict['NIN'] = ['CIFAR10']
networks_dict['AlexNet'] = ['CIFAR10', 'CIFAR100']

for network in networks_dict.keys():
  datasets = networks_dict[network]
  fig, axs = plt.subplots(len(datasets), 1, figsize=(10, len(datasets)*6 - (len(datasets)-1)))
  fig.add_subplot(111, frameon=False)
  plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
  for i_dataset in range(len(datasets)):
    dataset = datasets[i_dataset]
    selected_df = df[(df['network'] == network) & (df['dataset'] == dataset)].sort_values('saliency_name')
#    selected_df_noretraining = df_noretraining[(df_noretraining['network'] == network) & (df_noretraining['dataset'] == dataset)].sort_values('saliency_name')
#    valid_idx=(selected_df['sparsity_mean'] >= max_sparsity[network+'-'+dataset] - args.sparsity_drop).to_numpy()
    retraining_steps = selected_df['retraining_steps_2']
    colors = [mcolors.CSS4_COLORS[c] for c in selected_df['info_cat'].to_list()]
    sparsity_noretraining = selected_df['sparsity_noretraining']
    if len(datasets) > 1:
      axes_to_plot = axs[i_dataset]
    else:
      axes_to_plot = axs
    axes_to_plot.scatter(sparsity_noretraining, retraining_steps, color=colors)
    print(network, dataset, scipy.stats.spearmanr(retraining_steps, sparsity_noretraining))
    axes_to_plot.set_ylabel('Retraining steps to remove \n {:.1f} % weights'.format(max_sparsity[network+'-'+dataset]- args.sparsity_drop), fontsize=20)
    axes_to_plot.set_title(dataset)
#    axes_to_plot.legend(labels= legend_labels, loc = 'best', handles=legend_elements, ncol=1, prop = {'size': 15})
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
#    selected_df_noretraining = df_noretraining[(df_noretraining['network'] == network) & (df_noretraining['dataset'] == dataset)].sort_values('saliency_name')
#    valid_idx=(selected_df['sparsity_mean'] >= max_sparsity[network+'-'+dataset] - args.sparsity_drop).to_numpy()
#    retraining_steps = selected_df[valid_idx]['total_cost']
#    colors = selected_df[valid_idx]['info_cat'].to_list()
#    sparsity_noretraining = selected_df_noretraining[valid_idx]['sparsity_mean']
    retraining_steps = selected_df['total_cost']
    #colors = selected_df['info_cat'].to_list()
    colors = [mcolors.CSS4_COLORS[c] for c in selected_df['info_cat'].to_list()]
    sparsity_noretraining = selected_df['sparsity_noretraining']
    if len(datasets) > 1:
      axes_to_plot = axs[i_dataset]
    else:
      axes_to_plot = axs
    axes_to_plot.scatter(sparsity_noretraining, retraining_steps, color=colors)
    print(network, dataset, scipy.stats.spearmanr(retraining_steps, sparsity_noretraining))

    axes_to_plot.set_ylabel('Total steps to remove \n {:.1f} % weights'.format(max_sparsity[network+'-'+dataset]-args.sparsity_drop), fontsize=20)
    axes_to_plot.set_title(dataset)
#    axes_to_plot.legend(labels= legend_labels, loc = 'best', handles=legend_elements, ncol=1, prop = {'size': 15})

  plt.xlabel('Weights removed (%) with no retraining', fontsize=20)
  fig.suptitle(network)
  fig.tight_layout(pad=3.0)
  
  fig.savefig('graphs/total_steps_' + network + '.pdf')

plt.close('all')

figsize = (5, 1.5)
fig_leg = plt.figure(figsize=figsize)
ax_leg = fig_leg.add_subplot(111)
ax_leg.legend(labels=legend_labels, handles=legend_elements, loc='center', ncol=3)
ax_leg.axis('off')
fig_leg.savefig('graphs/total_steps_legend.pdf')

plt.close('all')
