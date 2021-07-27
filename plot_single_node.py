import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse

saliencies_to_plot = ['weight-taylor-l1_norm-no_normalisation', 'weight-taylor-l1_norm-l0_normalisation_adjusted', 'weight-average_input-l1_norm-no_normalisation', 'weight-average_input-l1_norm-l0_normalisation_adjusted', 'activation-taylor-l1_norm-no_normalisation', 'activation-taylor-l1_norm-l0_normalisation_adjusted', 'activation-average_input-l1_norm-no_normalisation', 'activation-average_input-l1_norm-l0_normalisation_adjusted']

#rcParams.update({'figure.autolayout': True})

parser = argparse.ArgumentParser()
parser.add_argument('--transitive', action='store_true', default=False)
args = parser.parse_args()

df_io = pd.read_csv('no_retraining_input_output.csv')
df_no = pd.read_csv('no_retraining_single_node.csv')
df_o = pd.read_csv('no_retraining_output.csv')
df_ot = pd.read_csv('no_retraining_transitive.csv')
df_io = df_io.drop(columns=['iteration'])
df_no = df_no.drop(columns=['iteration'])
df_o = df_o.drop(columns=['iteration'])
df_ot = df_ot.drop(columns=['iteration'])

df_merge = pd.merge(df_io, df_no, on=['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling'], suffixes=['_io', '_no'])
df_o.rename(columns={'sparsity': 'sparsity_o'}, inplace=True)
df_merge = pd.merge(df_merge, df_o, on=['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling'], suffixes=['', '_o'])
df_merge = df_merge.drop(columns=['Unnamed: 0'])
df_ot.rename(columns={'sparsity': 'sparsity_ot'}, inplace=True)
df_merge = pd.merge(df_merge, df_ot, on=['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling'], suffixes=['', '_ot'])
df_merge = df_merge.drop(columns=['Unnamed: 0'])
df_merge = df_merge.drop(columns=['Unnamed: 0_io', 'Unnamed: 0_no'])

df_merge['sparsity_sqr_o'] = pd.to_numeric(df_merge['sparsity_o']**2)
df_merge['sparsity_sqr_io'] = pd.to_numeric(df_merge['sparsity_io']**2)
df_merge['sparsity_sqr_no'] = pd.to_numeric(df_merge['sparsity_no']**2)
df_merge['sparsity_sqr_ot'] = pd.to_numeric(df_merge['sparsity_ot']**2)

df_merge.loc[df_merge['dataset'] == 'IMAGENET2012', 'dataset'] = 'ImageNet'
df_merge.loc[df_merge['dataset'] == 'IMAGENET32x32', 'dataset'] = 'ImageNet-32'
df_merge.loc[df_merge['dataset'] == 'CIFAR10', 'dataset'] = 'CIFAR-10'
df_merge.loc[df_merge['dataset'] == 'CIFAR100', 'dataset'] = 'CIFAR-100'


mean_df = df_merge.groupby(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']).mean()
mean_df['sparsity_err_o'] = pd.to_numeric( 2*(((mean_df['sparsity_sqr_o']) - (mean_df['sparsity_o']**2))**0.5))
mean_df['sparsity_err_io'] = pd.to_numeric( 2*(((mean_df['sparsity_sqr_io']) - (mean_df['sparsity_io']**2))**0.5))
mean_df['sparsity_err_no'] = pd.to_numeric( 2*(((mean_df['sparsity_sqr_no']) - (mean_df['sparsity_no']**2))**0.5))
mean_df['sparsity_err_ot'] = pd.to_numeric( 2*(((mean_df['sparsity_sqr_ot']) - (mean_df['sparsity_ot']**2))**0.5))
mean_df['imp_ot'] = pd.to_numeric(mean_df['sparsity_ot'] - mean_df['sparsity_o'])
mean_df['imp_io'] = pd.to_numeric(mean_df['sparsity_io'] - mean_df['sparsity_o'])
#rename 


mean_df = mean_df.reset_index()
mean_df['Metric'] = mean_df[['saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']].agg('-'.join, axis=1)
mean_df = mean_df[mean_df.Metric.isin(saliencies_to_plot)]
#saliencies_to_plot = ['weight-taylor-l1_norm-no_normalisation', 'weight-taylor-l1_norm-l0_normalisation_adjusted', 'weight-average_input-l1_norm-no_normalisation', 'weight-average_input-l1_norm-l0_normalisation_adjusted', 'activation-taylor-l1_norm-no_normalisation', 'activation-taylor-l1_norm-l0_normalisation_adjusted', 'activation-average_input-l1_norm-no_normalisation', 'activation-average_input-l1_norm-l0_normalisation_adjusted']
mean_df.loc[mean_df['Metric'] == 'weight-taylor-l1_norm-no_normalisation', 'Metric'] = 'Taylor-w'
mean_df.loc[mean_df['Metric'] == 'activation-taylor-l1_norm-no_normalisation', 'Metric'] = 'Taylor-a'
mean_df.loc[mean_df['Metric'] == 'weight-taylor-l1_norm-l0_normalisation_adjusted', 'Metric'] = 'Taylor-w-avg'
mean_df.loc[mean_df['Metric'] == 'activation-taylor-l1_norm-l0_normalisation_adjusted', 'Metric'] = 'Taylor-a-avg'
mean_df.loc[mean_df['Metric'] == 'weight-average_input-l1_norm-no_normalisation', 'Metric'] = 'L1-w'
mean_df.loc[mean_df['Metric'] == 'weight-average_input-l1_norm-l0_normalisation_adjusted', 'Metric'] = 'L1-w-avg'

networks = mean_df['network'].unique()
num_single = 0
for network in networks:
  network_df = mean_df[mean_df['network'] == network]
  datasets = network_df['dataset'].unique()
  fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 5), sharey=True)
  if len(datasets) == 1:
    num_single = num_single + 1
    continue
  for i_dataset in range(len(datasets)):
    dataset = datasets[i_dataset]
    print(network, dataset)
    selected_df = network_df[network_df['dataset'] == dataset]
    selected_df = selected_df.drop(columns=['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']).set_index(['Metric'])
    #data = selected_df[['sparsity_io', 'sparsity_no', 'sparsity_ot', 'sparsity_o']]
    #data_err = selected_df[['sparsity_err_io', 'sparsity_err_no', 'sparsity_err_ot', 'sparsity_err_o']].rename(columns = {'sparsity_err_io': 'sparsity_io', 'sparsity_err_no': 'sparsity_no', 'sparsity_err_ot': 'sparsity_ot', 'sparsity_err_o': 'sparsity_o'})
    data = selected_df[['sparsity_ot', 'sparsity_io', 'sparsity_o']].rename(columns = { 'sparsity_io': 'Domino-io', 'sparsity_ot': 'Domino-o', 'sparsity_o' : 'Channel saliency'})
    data_err = selected_df[['sparsity_err_ot', 'sparsity_err_io', 'sparsity_err_o']].rename(columns = {'sparsity_err_io': 'Domino-io', 'sparsity_err_ot': 'Domino-o', 'sparsity_err_o': 'Channel saliency'})
    if len(datasets) > 1 :
      axe=axes[i_dataset]
    else:
      axe = axes
    data.plot(ax=axe, kind='barh', xerr=data_err, legend=False)
    axe.set_title('{}'.format(dataset), pad=40.0, fontsize=16)
    axe.set_xlabel('Convolution Weights Removed (%)', fontsize=16)
    axe.set_ylabel('Metric', fontsize=16)
    if i_dataset == 0 :
      axe.legend(bbox_to_anchor=(0,1), loc="lower left", ncol=3, prop={'size': 16})
  plt.subplots_adjust(wspace=0.01, hspace=0)
  #plt.tight_layout()
  plt.savefig('graphs/domino-pruning/{}_transitive_comp.pdf'.format(network), bbox_inches='tight')

fig, axes = plt.subplots(1, num_single, figsize=(5*num_single, 5), sharey=True)
i_network = 0
for network in networks:
  network_df = mean_df[mean_df['network'] == network]
  datasets = network_df['dataset'].unique()
  if len(datasets) > 1:
    continue
  for i_dataset in range(len(datasets)):
    axe=axes[i_network]
    dataset = datasets[i_dataset]
    print(network, dataset)
    selected_df = network_df[network_df['dataset'] == dataset]
    selected_df = selected_df.drop(columns=['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']).set_index(['Metric'])
    data = selected_df[['sparsity_ot', 'sparsity_io', 'sparsity_o']].rename(columns = { 'sparsity_io': 'Domino-io', 'sparsity_ot': 'Domino-o', 'sparsity_o' : 'Channel saliency'})
    data_err = selected_df[['sparsity_err_ot', 'sparsity_err_io', 'sparsity_err_o']].rename(columns = {'sparsity_err_io': 'Domino-io', 'sparsity_err_ot': 'Domino-o', 'sparsity_err_o': 'Channel saliency'})
    data.plot(ax=axe, kind='barh', xerr=data_err, legend=False)
    axe.set_title('{} on {}'.format(network, dataset), pad=40.0, fontsize=16)
    axe.set_xlabel('Convolution Weights Removed (%)', fontsize=16)
    axe.set_ylabel('Metric', fontsize=16)
    if i_network == 0 :
      axe.legend(bbox_to_anchor=(0,1), loc="lower left", ncol=3, prop={'size': 16})
  plt.subplots_adjust(wspace=0.01, hspace=0)
  plt.savefig('graphs/domino-pruning/Others_transitive_comp.pdf'.format(network), bbox_inches = 'tight')
  i_network += 1

selected_df =mean_df.groupby(['network', 'dataset']).mean().reset_index()
selected_df['network-dataset'] = selected_df[['network', 'dataset']].agg('\n'.join, axis=1)
selected_df = selected_df.set_index(['network-dataset'])
data = selected_df[['imp_ot', 'imp_io']].rename(columns = { 'imp_ot' : 'Domino-o', 'imp_io' : 'Domino-io'})
data.plot(kind='bar', figsize=(5, 3), rot=90)
plt.title('Average Improvement')
plt.ylabel('Convolution Weights Removed (%)')
plt.xlabel('')
plt.savefig('graphs/domino-pruning/improvement.pdf', bbox_inches='tight')
plt.close('all')

selected_df = mean_df.groupby(['network', 'dataset']).max().reset_index()
selected_df['network-dataset'] = selected_df[['network', 'dataset']].agg('\n'.join, axis=1)
selected_df = selected_df.set_index(['network-dataset'])
data = selected_df
data['imp_ot'] = pd.to_numeric(data['sparsity_ot'] - data['sparsity_o'])
data['imp_io'] = pd.to_numeric(data['sparsity_io'] - data['sparsity_o'])
data = data[['imp_ot', 'imp_io']].rename(columns = { 'imp_ot' : 'Domino-o', 'imp_io' : 'Domino-io'})
data.plot(kind='bar', figsize=(5, 3), rot=90)
plt.title('Improvement in Max Weights Removed')
plt.ylabel('Convolution Weights Removed (%)')
plt.xlabel('')
plt.savefig('graphs/domino-pruning/improvement_max.pdf', bbox_inches='tight')
plt.close('all')
