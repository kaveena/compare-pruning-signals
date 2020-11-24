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
from matplotlib.lines import Line2D
import pandas as pd



parser = argparse.ArgumentParser()
parser.add_argument('--retrain', action='store_true', default=False)
parser.add_argument('--characterise', action='store_true', default=False)

args = parser.parse_args()  

if args.retrain:
  df = pd.read_csv('retrain.csv')
  networks_dict = networks_dict_2
elif args.characterise:
  df = pd.read_csv('characterise.csv')
  networks_dict = networks_dict_3
else:
  df = pd.read_csv('no_retraining.csv')
  networks_dict = networks_dict_1

df.network = pd.Categorical(df.network, categories=networks)
df.dataset = pd.Categorical(df.dataset, categories=datasets)
df.saliency_input = pd.Categorical(df.saliency_input, categories=inputs)
df.pointwise_saliency = pd.Categorical(df.pointwise_saliency, categories=pointwise_saliencies)
df.saliency_reduction = pd.Categorical(df.saliency_reduction, categories=reductions)
df.saliency_scaling = pd.Categorical(df.saliency_scaling, categories=scalings)
#df=df.set_index(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling', 'iteration'])
df['sparsity_sqr'] = pd.to_numeric(df['sparsity']**2)
df = df.drop(columns=['Unnamed: 0', 'retraining_steps', 'retraining_steps_2', 'pruning_steps', 'pruning_steps_2'])
mean_df = df.groupby(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']).mean()
df = df.drop(columns=['sparsity_sqr'])
mean_df = mean_df.reset_index()
mean_df['sparsity_err'] = pd.to_numeric(2 * (mean_df['sparsity_sqr'] - (mean_df['sparsity']**2))**0.5)
mean_df = mean_df.drop(columns=['sparsity_sqr']).dropna()
plot_legends = [convert_scaling(i_l) for i_l in scalings]
legend_elements = []
legend_labels = []
for i_l in range(len(plot_legends)): 
  legend_elements.append(Line2D([0], [0], marker='None', color=get_color_normalisation(scalings[i_l]), label=plot_legends[i_l], markerfacecolor=get_color_normalisation(scalings[i_l]), linestyle='-', linewidth=10)) 
  legend_labels.append(plot_legends[i_l])

width = 0.8
x = np.arange(5)
for network in networks_dict.keys():
  for dataset in networks_dict[network]:
    max_network_sparsity = max_sparsity[network + '-' + dataset]
    ylim = (max_network_sparsity - max_network_sparsity % 10) + 10
    for saliency_input in inputs:
      for pointwise_saliency in pointwise_saliencies:
        selected_df = mean_df
        selected_df = selected_df [ np.logical_and.reduce((
                                    selected_df['network'] == network,
                                    selected_df['dataset'] == dataset,
                                    selected_df['saliency_input'] == saliency_input,
                                    selected_df['pointwise_saliency'] == pointwise_saliency
                                  ))]
        if (len(selected_df)==25):
          fig, axs = plt.subplots(1, 1, figsize=(10,6))
          fig.add_subplot(111, frameon=False)
          plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
          for i_s in range(len(scalings)):
            saliency_scaling = scalings[i_s]
            selected_df2 = selected_df[selected_df['saliency_scaling'] == saliency_scaling]
            y_bar = selected_df2['sparsity'].to_list()
            y_bar_err = selected_df2['sparsity_err'].to_list()
            y_reduction = selected_df2['saliency_reduction'].to_list()
            y_scaling = selected_df2['saliency_scaling'].to_list()
            x_bar = ['-'.join([i,j,k]) for i, j, k in zip(selected_df2['saliency_input'].to_list(), y_reduction, y_scaling)]
            y_bar_color = [get_color_normalisation(i) for i in y_scaling]
  #          barlist = axs.bar([convert_label(i_x) for i_x in x_bar], y_bar, color = y_bar_color, alpha=0.99)
            barlist = axs.bar(x - ( 2 - i_s)*width/5 , y_bar, width/5, color = y_bar_color, yerr=y_bar_err, alpha=0.99)
#            for i_bar in range(len(barlist)):
#              bar = barlist[i_bar]
#              bar.set_hatch(get_norm_hatch(x_bar[i_bar].split('-')[1]))
          axs.set_xticks([-0.65,0.35, 1.35, 2.35,3.35])
          axs.set_xticks(x)
          axs.set_xticklabels([convert_label(i_x) for i_x in y_reduction], fontsize=18)
          axs.set_title('{} on {} with f(x) = {} and X = {}'.format(network, dataset, convert_label(pointwise_saliency), saliency_input[0].upper()), fontsize=15)
          axs.set_ylim((0,ylim))
          axs.set_xlabel("Reduction and Scaling used", fontsize=15)
          axs.set_ylabel("Convolution weights removed ($\%$)", fontsize=15)
  #        fig.suptitle(network)
          fig.tight_layout(pad=1.0)
          if args.retrain:
            fig.savefig('graphs/retrain_{}_{}_{}_{}.pdf'.format(network, dataset, saliency_input, pointwise_saliency)) 
          elif args.characterise:
            fig.savefig('graphs/characterise_{}_{}_{}_{}.pdf'.format(network, dataset, saliency_input, pointwise_saliency)) 
          else:
            fig.savefig('graphs/sensitivity_{}_{}_{}_{}.pdf'.format(network, dataset, saliency_input, pointwise_saliency)) 
          plt.close('all')

figsize = (5.5, 1)
fig_leg = plt.figure(figsize=figsize)
ax_leg = fig_leg.add_subplot(111)
ax_leg.legend(labels=legend_labels, handles=legend_elements, loc='center', ncol=5)
ax_leg.axis('off')
fig_leg.savefig('graphs/scaling_legend.pdf')

plt.close('all')
