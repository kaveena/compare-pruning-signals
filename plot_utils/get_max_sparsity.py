import numpy as np
import pandas as pd
from plot_utils.plot_data_util import *
from scipy.stats.mstats import gmean

df = pd.read_csv('characterise.csv')
best_df = pd.read_csv('best_characterise.csv')
df_noretraining = pd.read_csv('no_retraining.csv')

ylim = ylim_characterise
networks_dict = networks_dict_3

pd.set_option('display.max_rows', 500)

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

def df_reset_index2(df):
  df = df.reset_index()
  df.network = pd.Categorical(df.network, categories=networks)
  df.dataset = pd.Categorical(df.dataset, categories=datasets)
  df.saliency_input = pd.Categorical(df.saliency_input, categories=inputs)
  df.pointwise_saliency = pd.Categorical(df.pointwise_saliency, categories=pointwise_saliencies)
  df.saliency_reduction = pd.Categorical(df.saliency_reduction, categories=reductions)
  df.saliency_scaling = pd.Categorical(df.saliency_scaling, categories=scalings)
  return df

def df_reset_index3(df):
  df = df.reset_index()
  df.network = pd.Categorical(df.network, categories=networks)
  df.dataset = pd.Categorical(df.dataset, categories=datasets)
  df.saliency_input = pd.Categorical(df.saliency_input, categories=inputs)
  df.pointwise_saliency = pd.Categorical(df.pointwise_saliency, categories=pointwise_saliencies)
  df.saliency_reduction = pd.Categorical(df.saliency_reduction, categories=reductions)
  df.saliency_scaling = pd.Categorical(df.saliency_scaling, categories=scalings)
  df=df.set_index(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling'])
  return df


df.network = pd.Categorical(df.network, categories=networks)
df.dataset = pd.Categorical(df.dataset, categories=datasets)
df.saliency_input = pd.Categorical(df.saliency_input, categories=inputs)
df.pointwise_saliency = pd.Categorical(df.pointwise_saliency, categories=pointwise_saliencies)
df.saliency_reduction = pd.Categorical(df.saliency_reduction, categories=reductions)
df.saliency_scaling = pd.Categorical(df.saliency_scaling, categories=scalings)
df=df.set_index(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling', 'iteration'])
df['sparsity_sqr'] = pd.to_numeric(df['sparsity']**2)
df = df.drop(columns=['Unnamed: 0', 'retraining_steps', 'retraining_steps_2', 'pruning_steps', 'pruning_steps_2'])
best_df = df_reset_index(best_df)
best_df['sparsity_sqr'] = pd.to_numeric(best_df['sparsity']**2)
best_df = best_df.drop(columns=['Unnamed: 0', 'retraining_steps', 'retraining_steps_2', 'pruning_steps', 'pruning_steps_2'])

df_noretraining.network = pd.Categorical(df_noretraining.network, categories=networks)
df_noretraining.dataset = pd.Categorical(df_noretraining.dataset, categories=datasets)
df_noretraining.saliency_input = pd.Categorical(df_noretraining.saliency_input, categories=inputs)
df_noretraining.pointwise_saliency = pd.Categorical(df_noretraining.pointwise_saliency, categories=pointwise_saliencies)
df_noretraining.saliency_reduction = pd.Categorical(df_noretraining.saliency_reduction, categories=reductions)
df_noretraining.saliency_scaling = pd.Categorical(df_noretraining.saliency_scaling, categories=scalings)
df_noretraining=df_noretraining.set_index(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling', 'iteration'])
df_noretraining.rename(columns={'sparsity':'sparsity_noretraining'})

mean_df = df.groupby(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']).mean()
df = df.drop(columns=['sparsity_sqr'])
mean_df['sparsity_err'] = pd.to_numeric(2 * (mean_df['sparsity_sqr'] - (mean_df['sparsity']**2))**0.5)
mean_df = mean_df.drop(columns=['sparsity_sqr']).dropna()

best_mean_df = best_df.groupby(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']).mean()
best_df = best_df.drop(columns=['sparsity_sqr'])
best_mean_df['sparsity_err'] = pd.to_numeric(2 * (best_mean_df['sparsity_sqr'] - (best_mean_df['sparsity']**2))**0.5)
best_mean_df = best_mean_df.drop(columns=['sparsity_sqr']).dropna()

max_mean_sparsity_df = mean_df[['sparsity']].groupby(['network', 'dataset'])['sparsity'].max().dropna().to_frame()

max_df = df.groupby(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']).max().dropna()

min_df = df.groupby(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']).min().dropna()
min_df = min_df.rename(columns={'sparsity':'sparsity_itrmin'})

max_sparsity_df = df.groupby(['network', 'dataset']).max().dropna()

df = df_reset_index(df.join(mean_df, how='inner', rsuffix='_mean'))
df = df_reset_index(df.join(max_df, how='inner', rsuffix='_itrmax'))
df = df_reset_index(df.join(min_df, how='inner'))
df = df_reset_index(df.join(max_mean_sparsity_df, how='inner', rsuffix='_mean_max'))
df = df_reset_index(df.join(max_sparsity_df, how='inner', rsuffix='_max'))

df['dist_mean'] = pd.to_numeric(df['sparsity_mean_max'] - df['sparsity_mean'])

#geo_df = df.groupby(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']).sparsity.apply(lambda group: group.prod() ** (1 / float(len(group)))).to_frame()
geo_df = df.groupby(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']).sparsity.apply(gmean).to_frame()
max_geo_sparsity_df = geo_df[['sparsity']].groupby(['network', 'dataset'])['sparsity'].max().dropna()
geo_df = geo_df.join(max_geo_sparsity_df, on=['network', 'dataset'], rsuffix='_max')
geo_df = geo_df.reset_index()
geo_df.network = pd.Categorical(geo_df.network, categories=networks)
geo_df.dataset = pd.Categorical(geo_df.dataset, categories=datasets)
geo_df.saliency_input = pd.Categorical(geo_df.saliency_input, categories=inputs)
geo_df.pointwise_saliency = pd.Categorical(geo_df.pointwise_saliency, categories=pointwise_saliencies)
geo_df.saliency_reduction = pd.Categorical(geo_df.saliency_reduction, categories=reductions)
geo_df.saliency_scaling = pd.Categorical(geo_df.saliency_scaling, categories=scalings)
geo_df=geo_df.set_index(['network', 'dataset', 'saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling'])
geo_df['dist'] = pd.to_numeric(geo_df['sparsity_max'] - geo_df['sparsity'])

best_weights_only = 'weight-average_input-l1_norm-l1_normalisation'
best_weights_only = 'weight-average_input-l2_norm-weights_removed'
best_activations_only = 'activation-average_input-l2_norm-weights_removed'
best_gradients = 'taylor_2nd_approx2-abs_sum_norm-weights_removed'

weights_geo_only =  geo_df[ geo_df.index.get_level_values('saliency_input').isin(['weight'])
                &   geo_df.index.get_level_values('pointwise_saliency').isin(['average_input'])]
activations_geo_only =  geo_df[ geo_df.index.get_level_values('saliency_input').isin(['activation'])
                &   geo_df.index.get_level_values('pointwise_saliency').isin(['average_input'])]
gradients_geo_only = geo_df[np.logical_not(geo_df.index.get_level_values('pointwise_saliency').isin(['average_input']))]
weights_mean_only =  mean_df[ mean_df.index.get_level_values('saliency_input').isin(['weight'])
                &   mean_df.index.get_level_values('pointwise_saliency').isin(['average_input'])]
activations_mean_only =  mean_df[ mean_df.index.get_level_values('saliency_input').isin(['activation'])
                &   mean_df.index.get_level_values('pointwise_saliency').isin(['average_input'])]
gradients_mean_only = mean_df[np.logical_not(mean_df.index.get_level_values('pointwise_saliency').isin(['average_input']))]

max_weights_df = weights_geo_only.groupby(['network', 'dataset']).max().dropna()
max_activations_df = activations_geo_only.groupby(['network', 'dataset']).max().dropna()
max_gradients_df = gradients_geo_only.groupby(['network', 'dataset']).max().dropna()

max_geo_df = geo_df[geo_df.groupby(['network', 'dataset'])['sparsity'].transform(max) == geo_df['sparsity']]
max_weights_geo_df = weights_geo_only[weights_geo_only.groupby(['network', 'dataset'])['sparsity'].transform(max) == weights_geo_only['sparsity']]
max_activations_geo_df = activations_geo_only[activations_geo_only.groupby(['network', 'dataset'])['sparsity'].transform(max) == activations_geo_only['sparsity']]
max_gradients_geo_df = gradients_geo_only[gradients_geo_only.groupby(['network', 'dataset'])['sparsity'].transform(max) == gradients_geo_only['sparsity']]
max_weights_mean_df = weights_mean_only[weights_mean_only.groupby(['network', 'dataset'])['sparsity'].transform(max) == weights_mean_only['sparsity']]
max_activations_mean_df = activations_mean_only[activations_mean_only.groupby(['network', 'dataset'])['sparsity'].transform(max) == activations_mean_only['sparsity']]
max_gradients_mean_df = gradients_mean_only[gradients_mean_only.groupby(['network', 'dataset'])['sparsity'].transform(max) == gradients_mean_only['sparsity']]

best_weights_df = geo_df[ geo_df.index.get_level_values('saliency_input').isin(['weight'])
                  &   geo_df.index.get_level_values('pointwise_saliency').isin(['average_input'])
                  &   geo_df.index.get_level_values('saliency_reduction').isin(['l2_norm'])
                  &   geo_df.index.get_level_values('saliency_scaling').isin(['weights_removed'])
                    ]
best_activations_df = geo_df[ geo_df.index.get_level_values('saliency_input').isin(['activation'])
                  &   geo_df.index.get_level_values('pointwise_saliency').isin(['average_input'])
                  &   geo_df.index.get_level_values('saliency_reduction').isin(['l2_norm'])
                  &   geo_df.index.get_level_values('saliency_scaling').isin(['weights_removed'])
                    ]
best_gradients_df = geo_df[ geo_df.index.get_level_values('saliency_input').isin(['activation'])
                  &   geo_df.index.get_level_values('pointwise_saliency').isin(['taylor_2nd_approx2'])
                  &   geo_df.index.get_level_values('saliency_reduction').isin(['abs_sum_norm'])
                  &   geo_df.index.get_level_values('saliency_scaling').isin(['weights_removed'])
                    ]


#components =['saliency_input', 'pointwise_saliency', 'saliency_reduction', 'saliency_scaling']
#
#df['saliency_name']=df[components].agg('-'.join, axis=1)
#max_sparsity_df = df[['network', 'dataset', 'sparsity_mean']].groupby(['network', 'dataset']).max().rename(columns={'sparsity_mean' : 'max_sparsity'})
#max_weights_df = df[(df['saliency_input'] == 'weight') & (df['pointwise_saliency'] == 'average_input')][['network', 'dataset', 'sparsity_mean']].groupby(['network', 'dataset']).max().rename(columns={'sparsity_mean' : 'max_weights'})
#max_activations_df = df[(df['saliency_input'] == 'activation') & (df['pointwise_saliency'] == 'average_input')][['network', 'dataset', 'sparsity_mean']].groupby(['network', 'dataset']).max().rename(columns={'sparsity_mean' : 'max_activations'})
#max_gradients_df = df[(df['pointwise_saliency'] != 'average_input')][['network', 'dataset', 'sparsity_mean']].groupby(['network', 'dataset']).max().rename(columns={'sparsity_mean' : 'max_activations'})
#df = pd.merge(df, max_sparsity_df, on = ['network', 'dataset'])
#df['dist_sparsity'] = pd.to_numeric(df['max_sparsity'] - df['sparsity_mean'])
#df['dist_sparsity_norm'] = pd.to_numeric(( df['max_sparsity'] - df['sparsity_mean'] ) / df['sparsity_mean'])
#
#df_noretraining['saliency_name']=df_noretraining[components].agg('-'.join, axis=1)
#max_sparsity_df_noretraining = df_noretraining[['network', 'dataset', 'sparsity_mean']].groupby(['network', 'dataset']).max().rename(columns={'sparsity_mean' : 'max_sparsity'})
#max_weights_df_noretraining = df_noretraining[(df_noretraining['saliency_input'] == 'weight') & (df_noretraining['pointwise_saliency'] == 'average_input')][['network', 'dataset', 'sparsity_mean']].groupby(['network', 'dataset']).max().rename(columns={'sparsity_mean' : 'max_weights'})
#max_activations_df_noretraining = df_noretraining[(df_noretraining['saliency_input'] == 'activation') & (df_noretraining['pointwise_saliency'] == 'average_input')][['network', 'dataset', 'sparsity_mean']].groupby(['network', 'dataset']).max().rename(columns={'sparsity_mean' : 'max_activations'})
#df_noretraining = pd.merge(df_noretraining, max_sparsity_df_noretraining, on = ['network', 'dataset'])
#df_noretraining['dist_sparsity'] = pd.to_numeric(df_noretraining['max_sparsity'] - df_noretraining['sparsity_mean'])
#df_noretraining['dist_sparsity_norm'] = pd.to_numeric(( df_noretraining['max_sparsity'] - df_noretraining['sparsity_mean'] ) / df_noretraining['sparsity_mean'])
#
#weights_summary_df=df[(df['saliency_input'] == 'weight') & (df['pointwise_saliency'] == 'average_input')].groupby(components).mean().sort_values('dist_sparsity_norm')
#weights_summary_df_noretraining=df_noretraining[(df_noretraining['saliency_input'] == 'weight') & (df_noretraining['pointwise_saliency'] == 'average_input')].groupby(components).mean().sort_values('dist_sparsity_norm')
#
#avg_best_weights = 'weight-average_input-l2_norm-weights_removed'
#
#weights_avg_summary_df = df[df['saliency_name'] == avg_best_weights].groupby(['network', 'dataset'])[['network', 'dataset', 'sparsity_mean']].max()
#weights_avg_summary_df_noretraining = df_noretraining[df_noretraining['saliency_name'] == avg_best_weights].groupby(['network', 'dataset'])[['network', 'dataset', 'sparsity_mean']].max()
#
#activations_summary_df=df[(df['saliency_input'] == 'activation') & (df['pointwise_saliency'] == 'average_input')].groupby(components).mean().sort_values('dist_sparsity_norm')
#activations_summary_df_noretraining=df_noretraining[(df_noretraining['saliency_input'] == 'activation') & (df_noretraining['pointwise_saliency'] == 'average_input')].groupby(components).mean().sort_values('dist_sparsity_norm')
#
#avg_best_activations = 'activation-average_input-l2_norm-weights_removed'
#
#activations_avg_summary_df = df[df['saliency_name'] == avg_best_activations].groupby(['network', 'dataset'])[['network', 'dataset', 'sparsity_mean']].max()
#activations_avg_summary_df_noretraining = df_noretraining[df_noretraining['saliency_name'] == avg_best_activations].groupby(['network', 'dataset'])[['network', 'dataset', 'sparsity_mean']].max()
#
#avg_best_taylor = 'activation-taylor_2nd_approx2-l1_norm-no_normalisation'
#avg_best_taylor2 = 'activation-taylor_2nd_approx2-l1_norm-weights_removed'
#avg_best_taylor3 = 'activation-taylor_2nd_approx1-sqr_sum_norm-weights_removed'
#avg_best_gradient = 'activation-average_gradient-sqr_sum_norm-no_normalisation'
#avg_best_gradient2 = 'activation-average_gradient-sqr_sum_norm-weights_removed'
#
#taylor_avg_summary_df = df[df['saliency_name'] == avg_best_taylor].groupby(['network', 'dataset'])[['network', 'dataset', 'sparsity_mean']].max()
#taylor_avg_summary_df_noretraining = df_noretraining[df_noretraining['saliency_name'] == avg_best_taylor].groupby(['network', 'dataset'])[['network', 'dataset', 'sparsity_mean']].max()
#taylor2_avg_summary_df = df[df['saliency_name'] == avg_best_taylor2].groupby(['network', 'dataset'])[['network', 'dataset', 'sparsity_mean']].max()
#taylor3_avg_summary_df = df[df['saliency_name'] == avg_best_taylor3].groupby(['network', 'dataset'])[['network', 'dataset', 'sparsity_mean']].max()
#gradient_avg_summary_df = df[df['saliency_name'] == avg_best_gradient].groupby(['network', 'dataset'])[['network', 'dataset', 'sparsity_mean']].max()
#gradient2_avg_summary_df = df[df['saliency_name'] == avg_best_gradient2].groupby(['network', 'dataset'])[['network', 'dataset', 'sparsity_mean']].max()
#
#summary_df=df.groupby(components).mean()
#summary_df_noretraining=df_noretraining.groupby(components).mean()
