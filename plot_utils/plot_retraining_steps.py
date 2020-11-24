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

def df_reset_index(df):
  df = df.reset_index()
  df.network = pd.Categorical(df.network, categories=networks)
  df.dataset = pd.Categorical(df.dataset, categories=datasets)
  df.saliency_input = pd.Categorical(df.saliency_input, categories=inputs)
  df.pointwise_saliency = pd.Categorical(df.pointwise_saliency, categories=pointwise_saliencies)
  df.saliency_reduction = pd.Categorical(df.saliency_reduction, categories=reductions)
  df.saliency_scaling = pd.Categorical(df.saliency_scaling, categories=scalings)
  df=df.set_index(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling', 'iteration'])
  return df

def df_mean_reset_index(df):
  df = df.reset_index()
  df.network = pd.Categorical(df.network, categories=networks)
  df.dataset = pd.Categorical(df.dataset, categories=datasets)
  df.saliency_input = pd.Categorical(df.saliency_input, categories=inputs)
  df.pointwise_saliency = pd.Categorical(df.pointwise_saliency, categories=pointwise_saliencies)
  df.saliency_reduction = pd.Categorical(df.saliency_reduction, categories=reductions)
  df.saliency_scaling = pd.Categorical(df.saliency_scaling, categories=scalings)
  df=df.set_index(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling'])
  return df

parser = argparse.ArgumentParser()
parser.add_argument('--sparsity-drop', action='store', default=5.0)

args = parser.parse_args()  

df = pd.read_csv('characterise.csv')
df_noretraining = pd.read_csv('no_retraining.csv')
ylim = ylim_characterise
networks_dict = networks_dict_3
df = df.drop(columns=['Unnamed: 0', 'retraining_steps', 'pruning_steps'])

df['saliency_computation'] = 2
df.loc[(df['saliency_input'] == 'weight') & (df['pointwise_saliency'] == 'average_input'), 'saliency_computation'] = 0
df.loc[(df['saliency_input'] == 'activation') & (df['pointwise_saliency'] == 'average_input'), 'saliency_computation'] = 1
df.loc[(df['pointwise_saliency'] == 'hessian_diag_approx1'), 'saliency_computation'] = 3
df.loc[(df['pointwise_saliency'] == 'taylor_2nd_approx1'), 'saliency_computation'] = 3

df['total_cost'] = (df['pruning_steps_2'] * df['saliency_computation'] * 2) + 2*df['retraining_steps_2']



df = df_reset_index(df)
df_noretraining = df_reset_index(df_noretraining)

df = df_reset_index(df.join(df_noretraining, how='inner', rsuffix='_noretraining'))

mean_df = df.groupby(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']).mean().dropna()

max_mean_sparsity_df = mean_df[['sparsity']].groupby(['network', 'dataset'])['sparsity'].max().dropna().to_frame()
mean_df = df_mean_reset_index(mean_df.join(max_mean_sparsity_df, how='inner', rsuffix='_max'))
mean_df=mean_df[mean_df['sparsity'] >= mean_df['sparsity_max'] - args.sparsity_drop]
mean_df.loc[(mean_df['sparsity_noretraining'] >= mean_df['sparsity'] - args.sparsity_drop), 'retraining_steps_2'] = 0
mean_df.loc[(mean_df['sparsity_noretraining'] >= mean_df['sparsity'] - args.sparsity_drop), 'total_cost'] = mean_df['pruning_steps_2'] * mean_df['saliency_computation']

mean_df['info_cat'] = 'b'
mean_df.loc[ np.logical_and.reduce((
          mean_df.index.get_level_values('saliency_input').isin(['weight']),
          mean_df.index.get_level_values('pointwise_saliency').isin(['average_input'])
            )), 'info_cat'] = 'darkgreen'
mean_df.loc[ np.logical_and.reduce((
          mean_df.index.get_level_values('saliency_input').isin(['activation']),
          mean_df.index.get_level_values('pointwise_saliency').isin(['average_input'])
            )), 'info_cat'] = 'firebrick'
mean_df.loc[ 
          mean_df.index.get_level_values('pointwise_saliency').isin(['average_gradient'])
            , 'info_cat'] = 'olive'
mean_df.loc[ 
          mean_df.index.get_level_values('pointwise_saliency').isin(['taylor'])
            , 'info_cat'] = 'sandybrown'
mean_df.loc[ 
          mean_df.index.get_level_values('pointwise_saliency').isin(['taylor_2nd_approx2'])
            , 'info_cat'] = 'indigo'
mean_df.loc[ 
          mean_df.index.get_level_values('pointwise_saliency').isin(['hessian_diag_approx2'])
            , 'info_cat'] = 'teal'
mean_df.loc[ 
          mean_df.index.get_level_values('pointwise_saliency').isin(['taylor_2nd_approx1'])
            , 'info_cat'] = 'fuchsia'
mean_df.loc[ 
          mean_df.index.get_level_values('pointwise_saliency').isin(['hessian_diag_approx1'])
            , 'info_cat'] = 'pink'

plot_legends = [
'$w$', 
'$a$', 
'$\\frac{d\\mathcal{L}}{dx}$', 
'$-x\\frac{d\\mathcal{L}}{dx}$', 
'$-x\\frac{d\\mathcal{L}}{dx} + \\frac{x^2}{2}\\frac{d^2\\mathcal{L}}{dx^2}_{GN}$',
'$\\frac{x^2}{2}\\frac{d^2x\\mathcal{L}}{dx^2}_{GN}$', 
'$-x\\frac{d\\mathcal{L}}{dx} + \\frac{x^2}{2}\\frac{d\\mathcal{L}}{dx^2}_{LM}$', 
'$\\frac{x^2}{2}\\frac{d^2\\mathcal{L}}{dx^2}_{LM}$' 
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

#for network in networks_dict.keys():
#  datasets = networks_dict[network]
#  for i_dataset in range(len(datasets)):
#    dataset = datasets[i_dataset]
#    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
#    fig.add_subplot(111, frameon=False)
#    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#    selected_df = mean_df[ np.logical_and.reduce((
#                    mean_df.index.get_level_values('network').isin([network]),
#                    mean_df.index.get_level_values('network').isin([network])))].sort_index(level=['saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling'])
#    retraining_steps = selected_df['retraining_steps_2']
#    colors = [mcolors.CSS4_COLORS[c] for c in selected_df['info_cat'].to_list()]
#    sparsity_noretraining = selected_df['sparsity_noretraining']
#    axes_to_plot = axs
#    axes_to_plot.scatter(sparsity_noretraining, retraining_steps, color=colors)
#    print(network, dataset, scipy.stats.spearmanr(retraining_steps, sparsity_noretraining))
#    axes_to_plot.set_ylabel('Retraining steps to remove \n {:.1f} % weights'.format(max_sparsity[network+'-'+dataset]- args.sparsity_drop), fontsize=20)
#    axes_to_plot.set_title('{} on {}'.format(network, dataset))
#    plt.xlabel('Weights removed (%) with no retraining', fontsize=20)
#    fig.tight_layout(pad=3.0)
#    plt.tick_params(axis='both', which='major', labelsize=20)
#    plt.tick_params(axis='both', which='minor', labelsize=10)
#    fig.savefig('graphs/retraining_steps_{}-{}.pdf'.format(network, dataset))

for network in networks_dict.keys():
  datasets = networks_dict[network]
  for i_dataset in range(len(datasets)):
    dataset = datasets[i_dataset]
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    selected_df = mean_df[ np.logical_and.reduce((
                    mean_df.index.get_level_values('network').isin([network]),
                    mean_df.index.get_level_values('dataset').isin([dataset])))].sort_index(level=['saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling'])
    total_cost = selected_df['total_cost']
    colors = [mcolors.CSS4_COLORS[c] for c in selected_df['info_cat'].to_list()]
    sparsity_noretraining = selected_df['sparsity_noretraining']
    axes_to_plot = axs
    axes_to_plot.scatter(sparsity_noretraining, total_cost, color=colors)
    print(network, dataset, scipy.stats.spearmanr(total_cost, sparsity_noretraining), len(selected_df))

    axes_to_plot.set_ylabel('Total steps to remove \n {:.1f} % weights'.format(max_sparsity[network+'-'+dataset]-args.sparsity_drop), fontsize=20)
    axes_to_plot.set_title('{} on {}'.format(network, dataset))

    plt.xlabel('Weights removed (%) with no retraining', fontsize=20)
#  fig.suptitle(network)
    fig.tight_layout(pad=3.0)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    fig.savefig('graphs/total_steps_{}-{}.pdf'.format(network, dataset))

plt.close('all')

figsize = (9.5, 0.5)
fig_leg = plt.figure(figsize=figsize)
ax_leg = fig_leg.add_subplot(111)
ax_leg.legend(labels=legend_labels, handles=legend_elements, loc='center', ncol=8)
ax_leg.axis('off')
fig_leg.tight_layout()
fig_leg.savefig('graphs/total_steps_legend.pdf')

plt.close('all')
