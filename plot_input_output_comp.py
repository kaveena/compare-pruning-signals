import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})

df_caffe = pd.read_csv('no_retraining.csv')
df_input = pd.read_csv('no_retraining_input.csv')
df_input_output = pd.read_csv('no_retraining_input_output.csv')
df_merge1 = pd.merge(df_caffe, df_input, on=['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling'], suffixes=('_output', '_input'))
df_merge2 = pd.merge(df_caffe, df_input_output, on=['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling'], suffixes=('_o', '_input_output'))
df_merge = pd.merge(df_merge1, df_merge2, on=['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling'])
df_merge = df_merge.drop(columns=['Unnamed: 0_output', 'iteration_output', 'Unnamed: 0_input', 'iteration_input', 'Unnamed: 0_o', 'iteration_o', 'sparsity_o', 'Unnamed: 0_input_output', 'iteration_input_output'])
#df_merge['sparsity_sqr_x'] = pd.to_numeric(df_merge['sparsity_x']**2)
#df_merge['sparsity_sqr_y'] = pd.to_numeric(df_merge['sparsity_y']**2)
mean_df = df_merge.groupby(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']).mean()
#mean_df['sparsity_err_x'] = pd.to_numeric( 2*(((mean_df['sparsity_sqr_x']) - (mean_df['sparsity_x']**2))**0.5))
#mean_df['sparsity_err_y'] = pd.to_numeric( 2*(((mean_df['sparsity_sqr_y']) - (mean_df['sparsity_y']**2))**0.5))
mean_df = mean_df.reset_index()
networks = mean_df['network'].unique()
for network in networks:
  network_df = mean_df[mean_df['network'] == network]
  datasets = network_df['dataset'].unique()
  for dataset in datasets:
    selected_df = network_df[network_df['dataset'] == dataset]
    selected_df = selected_df.drop(columns=['network', 'dataset'])
    data = selected_df.set_index(['saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling'])[['sparsity_output', 'sparsity_input', 'sparsity_input_output']]
    data.plot(kind='barh', figsize=(10,5))
    plt.tight_layout(pad=2.0)
    plt.savefig('{}-{}_input_output_comp.pdf'.format(network, dataset))
plt.close('all')
