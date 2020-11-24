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
parser.add_argument('--characterise', action='store_true', default=False)

args = parser.parse_args()  

if args.characterise:
  df = pd.read_csv('summary_characterise.csv')
  ylim = ylim_characterise
  networks_dict = networks_dict_3
else:
  df = pd.read_csv('summary_no_retraining.csv')
  ylim = ylim_sensitivity
  networks_dict = networks_dict_1

networks_dict = networks_dict_3

df_h2 = df[(df['pointwise_saliency'] == 'taylor') & (df['saliency_reduction'] == 'l2_norm')][['network', 'dataset', 'saliency_scaling', 'saliency_input', 'sparsity_mean', 'sparsity_err']]
df_h2['pointwise_saliency'] = 'hessian_diag_approx2'
df_h2['saliency_reduction'] = 'l1_norm'
df=pd.concat([df,df_h2])
df = df[(df['saliency_reduction'] == 'l1_norm') & (df['saliency_scaling'] == 'no_normalisation')]
df['saliency_input-pointwise_saliency'] = df['saliency_input'] + '-' + df['pointwise_saliency']
df['pointwise_saliency'] = pd.Categorical(df.pointwise_saliency, categories= pointwise_saliencies)

width = 0.3

latex_table_a = dict()
latex_table_w = dict()

for network in ['LeNet-5', 'CIFAR10', 'ResNet-20', 'NIN', 'AlexNet']:
  datasets = networks_dict[network]
  plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
  for i_dataset in range(len(datasets)):
    #fig, axs = plt.subplots(5, 2, figsize=(5, 10))
    #fig.add_subplot(111, frameon=False)
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    #fig.add_subplot(111, frameon=False)
    dataset = datasets[i_dataset]
    if dataset not in latex_table_a.keys():
      latex_table_a[dataset] = dict()
      latex_table_w[dataset] = dict()
    selected_df = df[(df['network'] == network) & (df['dataset'] == dataset)]
    df1 = selected_df[selected_df['saliency_input'] == 'activation'].sort_values('pointwise_saliency')
    df2 = selected_df[selected_df['saliency_input'] == 'weight'].sort_values('pointwise_saliency')
    y_bar1 = df1['sparsity_mean'].to_list()
    y_bar_err1 = df1['sparsity_err'].to_list()
    x_bar = df1['pointwise_saliency'].to_list()
    y_bar2 = df2['sparsity_mean'].to_list()
    y_bar_err2 = df2['sparsity_err'].to_list()

    x = np.arange(len(x_bar))
    #axes_to_plot = axs[fig_y, fig_x]
    axes_to_plot = axs
    barlist1 = axes_to_plot.bar(x, y_bar1, width, yerr=y_bar_err1, alpha=0.99, label='$ x = a $')
    barlist2 = axes_to_plot.bar(x + width, y_bar2, width, yerr=y_bar_err2, alpha=0.99, label='$ x = w $')
    axes_to_plot.set_xticks(x + width/2)
    axes_to_plot.set_xticklabels([convert_label(i_x) for i_x in x_bar])
    for tick in axes_to_plot.get_xticklabels():
        tick.set_rotation(30)
    axes_to_plot.set_title(network + ' on ' + dataset)
    axes_to_plot.set_ylim((0, ylim[network][dataset]))
    axes_to_plot.set_ylabel('Weights removed (%)', fontsize=15)
    axes_to_plot.legend()
    #fig_y += 1
    #if fig_y >= 5 :
    #  fig_x += 1
    #  fig_y = 0
    plt.xlabel('Pointwise saliency ', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    #plt.tick_params(axis='both', which='minor', labelsize=10)
    fig.tight_layout(pad=0.2)
    print(network, dataset)
    print(x_bar)
#    print(['%.1f' % y for y in y_bar[0:7]])
#    print(['%.1f' % y for y in y_bar[7:14]])
    latex_table_a[dataset][network] = network + ' & ' + ' & '.join(['%.1f $\pm$ %.1f' % (a,b) for a,b in zip(y_bar1, y_bar_err1)]) + '\\\\ \n'
    latex_table_w[dataset][network] = network + ' & ' + ' & '.join(['%.1f $\pm$ %.1f' % (a,b) for a,b in zip(y_bar2, y_bar_err2)]) + '\\\\ \n'
    if args.characterise:
      fig.savefig('graphs/pointwise_saliency_retraining_' + network + '-' + dataset + '.pdf') 
    else:
      fig.savefig('graphs/pointwise_saliency_' + network + '-' + dataset + '.pdf') 
#fig.savefig('graphs/pointwise_saliency_' + network + '.pdf')

for dataset in latex_table_a.keys():
  print('\hline \n\multicolumn{8}{|l|}{\\textbf{ %s dataset}} \\\\ \n\hline' % dataset)
  print(''.join([latex_table_a[dataset][n] for n in latex_table_a[dataset]]))

for dataset in latex_table_w.keys():
  print('\hline \n\multicolumn{8}{|l|}{\\textbf{ %s dataset}} \\\\ \n\hline' % dataset)
  print(''.join([latex_table_w[dataset][n] for n in latex_table_w[dataset]]))

plt.close('all')
