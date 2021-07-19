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


mean_df = df_merge.groupby(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']).mean()
mean_df['sparsity_err_o'] = pd.to_numeric( 2*(((mean_df['sparsity_sqr_o']) - (mean_df['sparsity_o']**2))**0.5))
mean_df['sparsity_err_io'] = pd.to_numeric( 2*(((mean_df['sparsity_sqr_io']) - (mean_df['sparsity_io']**2))**0.5))
mean_df['sparsity_err_no'] = pd.to_numeric( 2*(((mean_df['sparsity_sqr_no']) - (mean_df['sparsity_no']**2))**0.5))
mean_df['sparsity_err_ot'] = pd.to_numeric( 2*(((mean_df['sparsity_sqr_ot']) - (mean_df['sparsity_ot']**2))**0.5))

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
for network in networks:
  network_df = mean_df[mean_df['network'] == network]
  datasets = network_df['dataset'].unique()
  for dataset in datasets:
    print(network, dataset)
    selected_df = network_df[network_df['dataset'] == dataset]
    selected_df = selected_df.drop(columns=['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']).set_index(['Metric'])
    #data = selected_df[['sparsity_io', 'sparsity_no', 'sparsity_ot', 'sparsity_o']]
    #data_err = selected_df[['sparsity_err_io', 'sparsity_err_no', 'sparsity_err_ot', 'sparsity_err_o']].rename(columns = {'sparsity_err_io': 'sparsity_io', 'sparsity_err_no': 'sparsity_no', 'sparsity_err_ot': 'sparsity_ot', 'sparsity_err_o': 'sparsity_o'})
    data = selected_df[['sparsity_io', 'sparsity_ot', 'sparsity_o']].rename(columns = { 'sparsity_io': 'Domino-io', 'sparsity_ot': 'Domino-o', 'sparsity_o' : 'Channel saliency'})
    data_err = selected_df[['sparsity_err_io', 'sparsity_err_ot', 'sparsity_err_o']].rename(columns = {'sparsity_err_io': 'sparsity_io', 'sparsity_err_ot': 'sparsity_ot', 'sparsity_err_o': 'sparsity_o'})
    data_err = data_err.rename(columns = { 'sparsity_io': 'Domino-io', 'sparsity_ot': 'Domino-o', 'sparsity_o' : 'channel saliency'})
    data.plot(kind='barh', xerr=data_err, figsize=(10,5), legend=False)
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", ncol=3)
    plt.xlabel('Convolution Weights Removed (%)', fontsize=16)
    plt.title('{} on {}'.format(network, dataset), pad=30.0, fontsize=16)
    plt.tight_layout()
    plt.savefig('graphs/domino-pruning/{}-{}_transitive_comp.pdf'.format(network, dataset))
plt.close('all')
