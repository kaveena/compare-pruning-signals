import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})

df_caffe = pd.read_csv('no_retraining_input_output.csv')
df_var = pd.read_csv('no_retraining_single_node.csv')
df_caffe = df_caffe.drop(columns=['iteration'])
df_var = df_var.drop(columns=['iteration'])
df_merge = pd.merge(df_caffe, df_var, on=['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling'], suffixes=['_all_nodes', '_single_node'])
df_merge = df_merge.drop(columns=['Unnamed: 0_all_nodes', 'Unnamed: 0_single_node'])
mean_df = df_merge.groupby(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']).mean()
#mean_df['sparsity_err_x'] = pd.to_numeric( 2*(((mean_df['sparsity_sqr_x']) - (mean_df['sparsity_x']**2))**0.5))
#mean_df['sparsity_err_y'] = pd.to_numeric( 2*(((mean_df['sparsity_sqr_y']) - (mean_df['sparsity_y']**2))**0.5))
mean_df = mean_df.reset_index()
networks = mean_df['network'].unique()
for network in networks:
  network_df = mean_df[mean_df['network'] == network]
  datasets = network_df['dataset'].unique()
  for dataset in datasets:
    print(network, dataset)
    selected_df = network_df[network_df['dataset'] == dataset]
    selected_df = selected_df.drop(columns=['network', 'dataset']).set_index(['saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling'])
    data = selected_df[['sparsity_all_nodes', 'sparsity_single_node']]
    data.plot(kind='barh', figsize=(10,5))
    plt.tight_layout(pad=2.0)
    plt.savefig('{}-{}_all_nodes_single_node_comp.pdf'.format(network, dataset))
plt.close('all')
