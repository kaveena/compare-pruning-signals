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
import matplotlib.patches as mpatches
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

df_h2 = df[(df['pointwise_saliency'] == 'taylor') & (df['saliency_reduction'] == 'l2_norm')]
df_h2.loc[:,'pointwise_saliency'] = 'hessian_diag_approx2'
df_h2.loc[:,'saliency_reduction'] = 'l1_norm'
df=pd.concat([df,df_h2])

df_h2 = df[(df['pointwise_saliency'] == 'taylor') & (df['saliency_reduction'] == 'l2_norm')]
df_h2.loc[:,'pointwise_saliency'] = 'hessian_diag_approx2'
df_h2.loc[:,'saliency_reduction'] = 'abs_norm'
df=pd.concat([df,df_h2])

df_h2 = df[(df['pointwise_saliency'] == 'taylor') & (df['saliency_reduction'] == 'l2_norm')]
df_h2.loc[:,'pointwise_saliency'] = 'hessian_diag_approx2'
df_h2.loc[:,'saliency_reduction'] = 'none_norm'
df=pd.concat([df,df_h2])

df_h2 = df[(df['pointwise_saliency'] == 'taylor') & ((df['saliency_reduction'] == 'none_norm') | (df['saliency_reduction'] == 'abs_sum_norm') | (df['saliency_reduction'] == 'sqr_sum_norm') ) & ((df['saliency_scaling'] == 'no_normalisation') | (df['saliency_scaling'] == 'l1_normalisation') | (df['saliency_scaling'] == 'l2_normalisation') | (df['saliency_scaling'] == 'weights_removed') )]
df_h2.loc[:,'saliency_input'] = 'weight'
df=pd.concat([df,df_h2])

df.network = pd.Categorical(df.network, categories=networks)
df.dataset = pd.Categorical(df.dataset, categories=datasets)
df.saliency_input = pd.Categorical(df.saliency_input, categories=inputs)
df.pointwise_saliency = pd.Categorical(df.pointwise_saliency, categories=pointwise_saliencies)
df.saliency_reduction = pd.Categorical(df.saliency_reduction, categories=reductions)
df.saliency_scaling = pd.Categorical(df.saliency_scaling, categories=scalings)
df['sparsity_sqr'] = pd.to_numeric(df['sparsity']**2)
df = df.drop(columns=['Unnamed: 0', 'retraining_steps', 'retraining_steps_2', 'pruning_steps', 'pruning_steps_2'])
# Average scaling
mean_df = df.groupby(['saliency_scaling']).mean()
mean_df = mean_df.reset_index()
mean_df['sparsity_err'] = pd.to_numeric(2 * (mean_df['sparsity_sqr'] - (mean_df['sparsity']**2))**0.5)
mean_df = mean_df.drop(columns=['sparsity_sqr']).dropna()
fig, axs = plt.subplots(1, 1, figsize=(5,5))
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axs.tick_params(axis='x', which='major', labelsize=15)
y_bar = mean_df['sparsity'].to_list()
x_bar_label = mean_df['saliency_scaling'].to_list()
x_bar_color = [get_color_normalisation(i) for i in scalings]
x_bar = [convert_scaling(i_l) for i_l in x_bar_label]
barlist = axs.bar(x_bar, y_bar, color = x_bar_color)
axs.set_xlabel("Scaling factor", fontsize=10)
axs.set_ylabel("Convolution weights removed ($\%$)", fontsize=10)
fig.tight_layout()
fig.savefig('graphs/barchart-scaling.pdf')

# Average reduction
mean_df = df.groupby(['saliency_reduction']).mean()
mean_df = mean_df.reset_index()
mean_df['sparsity_err'] = pd.to_numeric(2 * (mean_df['sparsity_sqr'] - (mean_df['sparsity']**2))**0.5)
mean_df = mean_df.drop(columns=['sparsity_sqr']).dropna()
plot_legends = [convert_label(i_l) for i_l in reductions]
fig, axs = plt.subplots(1, 1, figsize=(5,5))
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axs.tick_params(axis='x', which='major', labelsize=10)
y_bar = mean_df['sparsity'].to_list()
x_bar_label = mean_df['saliency_reduction'].to_list()
x_bar = [convert_label(i_l) for i_l in x_bar_label]
barlist = axs.bar(x_bar, y_bar)
axs.set_xlabel("Reduction method", fontsize=10)
axs.set_ylabel("Convolution weights removed ($\%$)", fontsize=10)
fig.tight_layout()
fig.savefig('graphs/barchart-reduction.pdf')

# Average scaling reduction
mean_df = df.groupby(['saliency_reduction', 'saliency_scaling']).mean()
mean_df = mean_df.reset_index()
mean_df['sparsity_err'] = pd.to_numeric(2 * (mean_df['sparsity_sqr'] - (mean_df['sparsity']**2))**0.5)
mean_df = mean_df.drop(columns=['sparsity_sqr']).dropna()
mean_df.saliency_reduction = pd.Categorical(mean_df.saliency_reduction, categories=reductions)
mean_df.saliency_scaling = pd.Categorical(mean_df.saliency_scaling, categories=scalings)
mean_df = mean_df.sort_values(['saliency_reduction', 'saliency_scaling'])
fig, axs = plt.subplots(1, 1, figsize=(10,6))
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axs.tick_params(axis='x', which='major', labelsize=10)
width = 0.8
x = np.arange(5)
for i_s in range(len(scalings)):
  saliency_scaling = scalings[i_s]
  selected_df = mean_df[mean_df['saliency_scaling'] == saliency_scaling]
  y_bar = selected_df['sparsity'].to_list()
  y_bar_err = selected_df['sparsity_err'].to_list()
  y_reduction = selected_df['saliency_reduction'].to_list()
  y_scaling = selected_df['saliency_scaling'].to_list()
  x_bar = ['-'.join([i,j]) for i, j in zip(y_reduction, y_scaling)]
  y_bar_color = [get_color_normalisation(i) for i in y_scaling]
  barlist = axs.bar(x - ( 2 - i_s)*width/5 , y_bar, width/5, color = y_bar_color)
legend_elements = []
for i_l in range(len(plot_legends)): 
  legend_elements.append(mpatches.Patch(facecolor=get_color_normalisation(scalings[i_l]), label=convert_scaling(scalings[i_l]), linewidth = 1))
legend = plt.legend(handles=legend_elements, title="Scaling factor, L", loc='upper left', bbox_to_anchor=(0,0.85), fontsize=15, fancybox=True)
plt.setp(legend.get_title(),fontsize=15)
#axs.set_xticks([-0.65,0.35, 1.35, 2.35,3.35])
axs.set_xticks(x)
axs.set_xticklabels([convert_label(i_x) for i_x in y_reduction], fontsize=15)
axs.set_xlabel("Reduction method", fontsize=15)
axs.set_ylabel("Convolution weights removed ($\%$)", fontsize=15)
axs.set_ylim((0,53))
maxSparsity = mean_df['sparsity'].max()
axs.plot([-0.4, 4.4], [maxSparsity, maxSparsity], "k--")
fig.tight_layout()
fig.savefig('graphs/barchart-reduction-scaling.pdf')

# Average scaling reduction on a-t22
mean_df = df.reset_index()
mean_df = mean_df[np.logical_and.reduce((mean_df['saliency_input']=='activation', mean_df['pointwise_saliency'] == 'taylor_2nd_approx2'))]
mean_df = mean_df.groupby(['saliency_reduction', 'saliency_scaling']).mean()
mean_df = mean_df.reset_index()
mean_df['sparsity_err'] = pd.to_numeric(2 * (mean_df['sparsity_sqr'] - (mean_df['sparsity']**2))**0.5)
mean_df = mean_df.drop(columns=['sparsity_sqr']).dropna()
mean_df.saliency_reduction = pd.Categorical(mean_df.saliency_reduction, categories=reductions)
mean_df.saliency_scaling = pd.Categorical(mean_df.saliency_scaling, categories=scalings)
mean_df = mean_df.sort_values(['saliency_reduction', 'saliency_scaling'])
fig, axs = plt.subplots(1, 1, figsize=(10,6))
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axs.tick_params(axis='x', which='major', labelsize=10)
width = 0.8
x = np.arange(5)
for i_s in range(len(scalings)):
  saliency_scaling = scalings[i_s]
  selected_df = mean_df[mean_df['saliency_scaling'] == saliency_scaling]
  y_bar = selected_df['sparsity'].to_list()
  y_bar_err = selected_df['sparsity_err'].to_list()
  y_reduction = selected_df['saliency_reduction'].to_list()
  y_scaling = selected_df['saliency_scaling'].to_list()
  x_bar = ['-'.join([i,j]) for i, j in zip(y_reduction, y_scaling)]
  y_bar_color = [get_color_normalisation(i) for i in y_scaling]
  barlist = axs.bar(x - ( 2 - i_s)*width/5 , y_bar, width/5, color = y_bar_color)
legend_elements = []
for i_l in range(len(plot_legends)): 
  legend_elements.append(mpatches.Patch(facecolor=get_color_normalisation(scalings[i_l]), label=convert_scaling(scalings[i_l]), linewidth = 1))
#legend = plt.legend(handles=legend_elements, title="Scaling factor", loc='upper left', fontsize=15, fancybox=True)
axs.set_xticks(x)
axs.set_xticklabels([convert_label(i_x) for i_x in y_reduction], fontsize=15)
axs.set_xlabel("Reduction method", fontsize=15)
axs.set_ylabel("Convolution weights removed ($\%$)", fontsize=15)
axs.set_ylim((0,53))
maxSparsity = mean_df['sparsity'].max()
axs.plot([-0.4, 4.4], [maxSparsity, maxSparsity], "k--")
fig.tight_layout()
fig.savefig('graphs/barchart-reduction-scaling-activation-taylor_2nd_approx2.pdf')

# Average scaling reduction on a-avg
mean_df = df.reset_index()
mean_df = mean_df[np.logical_and.reduce((mean_df['saliency_input']=='activation', mean_df['pointwise_saliency'] == 'average_input'))]
mean_df = mean_df.groupby(['saliency_reduction', 'saliency_scaling']).mean()
mean_df = mean_df.reset_index()
mean_df['sparsity_err'] = pd.to_numeric(2 * (mean_df['sparsity_sqr'] - (mean_df['sparsity']**2))**0.5)
mean_df = mean_df.drop(columns=['sparsity_sqr']).dropna()
mean_df.saliency_reduction = pd.Categorical(mean_df.saliency_reduction, categories=reductions)
mean_df.saliency_scaling = pd.Categorical(mean_df.saliency_scaling, categories=scalings)
mean_df = mean_df.sort_values(['saliency_reduction', 'saliency_scaling'])
fig, axs = plt.subplots(1, 1, figsize=(10,6))
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axs.tick_params(axis='x', which='major', labelsize=10)
width = 0.8
x = np.arange(5)
for i_s in range(len(scalings)):
  saliency_scaling = scalings[i_s]
  selected_df = mean_df[mean_df['saliency_scaling'] == saliency_scaling]
  y_bar = selected_df['sparsity'].to_list()
  y_bar_err = selected_df['sparsity_err'].to_list()
  y_reduction = selected_df['saliency_reduction'].to_list()
  y_scaling = selected_df['saliency_scaling'].to_list()
  x_bar = ['-'.join([i,j]) for i, j in zip(y_reduction, y_scaling)]
  y_bar_color = [get_color_normalisation(i) for i in y_scaling]
  barlist = axs.bar(x - ( 2 - i_s)*width/5 , y_bar, width/5, color = y_bar_color)
legend_elements = []
for i_l in range(len(plot_legends)): 
  legend_elements.append(mpatches.Patch(facecolor=get_color_normalisation(scalings[i_l]), label=convert_scaling(scalings[i_l]), linewidth = 1))
#legend = plt.legend(handles=legend_elements, title="Scaling factor", loc='upper left', fontsize=15, fancybox=True)
axs.set_xticks(x)
axs.set_xticklabels([convert_label(i_x) for i_x in y_reduction], fontsize=15)
axs.set_xlabel("Reduction method", fontsize=15)
axs.set_ylabel("Convolution weights removed ($\%$)", fontsize=15)
axs.set_ylim((0,53))
maxSparsity = mean_df['sparsity'].max()
axs.plot([-0.4, 4.4], [maxSparsity, maxSparsity], "k--")
fig.tight_layout()
fig.savefig('graphs/barchart-reduction-scaling-activation-average_input.pdf')

# Average scaling reduction on w-avg
mean_df = df.reset_index()
mean_df = mean_df[np.logical_and.reduce((mean_df['saliency_input']=='weight', mean_df['pointwise_saliency'] == 'average_input'))]
mean_df = mean_df.groupby(['saliency_reduction', 'saliency_scaling']).mean()
mean_df = mean_df.reset_index()
mean_df['sparsity_err'] = pd.to_numeric(2 * (mean_df['sparsity_sqr'] - (mean_df['sparsity']**2))**0.5)
mean_df = mean_df.drop(columns=['sparsity_sqr']).dropna()
mean_df.saliency_reduction = pd.Categorical(mean_df.saliency_reduction, categories=reductions)
mean_df.saliency_scaling = pd.Categorical(mean_df.saliency_scaling, categories=scalings)
mean_df = mean_df.sort_values(['saliency_reduction', 'saliency_scaling'])
fig, axs = plt.subplots(1, 1, figsize=(10,6))
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axs.tick_params(axis='x', which='major', labelsize=10)
width = 0.8
x = np.arange(5)
for i_s in range(len(scalings)):
  saliency_scaling = scalings[i_s]
  selected_df = mean_df[mean_df['saliency_scaling'] == saliency_scaling]
  y_bar = selected_df['sparsity'].to_list()
  y_bar_err = selected_df['sparsity_err'].to_list()
  y_reduction = selected_df['saliency_reduction'].to_list()
  y_scaling = selected_df['saliency_scaling'].to_list()
  x_bar = ['-'.join([i,j]) for i, j in zip(y_reduction, y_scaling)]
  y_bar_color = [get_color_normalisation(i) for i in y_scaling]
  barlist = axs.bar(x - ( 2 - i_s)*width/5 , y_bar, width/5, color = y_bar_color)
legend_elements = []
for i_l in range(len(plot_legends)): 
  legend_elements.append(mpatches.Patch(facecolor=get_color_normalisation(scalings[i_l]), label=convert_scaling(scalings[i_l]), linewidth = 1))
#legend = plt.legend(handles=legend_elements, title="Scaling factor", loc='upper left', fontsize=15, fancybox=True)
axs.set_xticks(x)
axs.set_xticklabels([convert_label(i_x) for i_x in y_reduction], fontsize=15)
axs.set_xlabel("Reduction method", fontsize=15)
axs.set_ylabel("Convolution weights removed ($\%$)", fontsize=15)
axs.set_ylim((0,53))
maxSparsity = mean_df['sparsity'].max()
axs.plot([-0.4, 4.4], [maxSparsity, maxSparsity], "k--")
fig.tight_layout()
fig.savefig('graphs/barchart-reduction-scaling-weight-average_input.pdf')

# Average reduction and scaling per network
for network in networks_dict_3.keys():
  for dataset in networks_dict_3[network]:
    mean_df = df.reset_index()
    mean_df = mean_df[np.logical_and.reduce((mean_df['network']==network, mean_df.reset_index()['dataset'] == dataset))]
    mean_df = mean_df.groupby(['saliency_reduction', 'saliency_scaling']).mean()
    mean_df = mean_df.reset_index()
    mean_df['sparsity_err'] = pd.to_numeric(2 * (mean_df['sparsity_sqr'] - (mean_df['sparsity']**2))**0.5)
    mean_df = mean_df.drop(columns=['sparsity_sqr']).dropna()
    mean_df.saliency_reduction = pd.Categorical(mean_df.saliency_reduction, categories=reductions)
    mean_df.saliency_scaling = pd.Categorical(mean_df.saliency_scaling, categories=scalings)
    mean_df = mean_df.sort_values(['saliency_reduction', 'saliency_scaling'])
    fig, axs = plt.subplots(1, 1, figsize=(10,6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    axs.tick_params(axis='x', which='major', labelsize=10)
    width = 0.8
    x = np.arange(5)
    for i_s in range(len(scalings)):
      saliency_scaling = scalings[i_s]
      selected_df = mean_df[mean_df['saliency_scaling'] == saliency_scaling]
      y_bar = selected_df['sparsity'].to_list()
      y_bar_err = selected_df['sparsity_err'].to_list()
      y_reduction = selected_df['saliency_reduction'].to_list()
      y_scaling = selected_df['saliency_scaling'].to_list()
      x_bar = ['-'.join([i,j]) for i, j in zip(y_reduction, y_scaling)]
      y_bar_color = [get_color_normalisation(i) for i in y_scaling]
      barlist = axs.bar(x - ( 2 - i_s)*width/5 , y_bar, width/5, color = y_bar_color)
    legend_elements = []
    for i_l in range(len(plot_legends)): 
      legend_elements.append(mpatches.Patch(facecolor=get_color_normalisation(scalings[i_l]), label=convert_scaling(scalings[i_l]), linewidth = 1))
    legend = plt.legend(handles=legend_elements, title="Scaling factor", loc='upper left', fontsize=15, fancybox=True)
    ylim = mean_df['sparsity'].max() + 10
    axs.set_ylim((0, ylim))
    axs.set_xticks(x)
    axs.set_xticklabels([convert_label(i_x) for i_x in y_reduction], fontsize=15)
    axs.set_xlabel("Reduction method", fontsize=15)
    axs.set_ylabel("Convolution weights removed ($\%$)", fontsize=15)
    fig.tight_layout()
    fig.savefig('graphs/barchart-reduction-scaling-{}-{}.pdf'.format(network,dataset))
