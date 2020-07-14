import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

networks = ['LeNet-5-CIFAR10', 'CIFAR10-CIFAR10', 'NIN-CIFAR10']

all_networks = pd.DataFrame()

all_networks_dict = dict()
for network in networks:
  csv_file = 'results_csv/' + network + '/summary_database.csv'
  if os.path.isfile(csv_file):
    all_networks_dict[network] = (pd.read_csv(csv_file))

for network in networks:
  all_networks = all_networks.append(all_networks_dict[network], ignore_index=True)

all_networks['layerwise_normalisation_improvement_abs_test_acc_1'] = 0
all_networks['layerwise_normalisation_improvement_per_test_acc_1'] = 0
all_networks['layerwise_normalisation_improvement_abs_test_acc_01'] = 0
all_networks['layerwise_normalisation_improvement_per_test_acc_01'] = 0
all_networks['layerwise_normalisation_improvement_abs_test_acc_2'] = 0
all_networks['layerwise_normalisation_improvement_per_test_acc_2'] = 0
all_networks['layerwise_normalisation_improvement_abs_test_acc_5'] = 0
all_networks['layerwise_normalisation_improvement_per_test_acc_5'] = 0

for index, row in all_networks.iterrows():
  no_normalisation_df = all_networks.loc[(all_networks['network'] == row['network']) & (all_networks['saliency_input'] == row['saliency_input']) & (all_networks['method'] == row['method']) & (all_networks['saliency_norm'] == row['saliency_norm']) & (all_networks['layerwise_normalisation'] == 'no_normalisation')]
  if len(no_normalisation_df) != 0:
    no_normalisation = no_normalisation_df.iloc[0]
    all_networks.loc[index, 'layerwise_normalisation_improvement_abs_test_acc_1'] = (row['sparsity_test_acc_1'] - no_normalisation['sparsity_test_acc_1'])
    if no_normalisation['sparsity_test_acc_1'] > 0:
      all_networks.loc[index, 'layerwise_normalisation_improvement_per_test_acc_1'] = (row['sparsity_test_acc_1'] - no_normalisation['sparsity_test_acc_1']) / no_normalisation['sparsity_test_acc_1'] 
    all_networks.loc[index, 'layerwise_normalisation_improvement_abs_test_acc_01'] = (row['sparsity_test_acc_01'] - no_normalisation['sparsity_test_acc_01'])
    if no_normalisation['sparsity_test_acc_01'] > 0:
      all_networks.loc[index, 'layerwise_normalisation_improvement_per_test_acc_01'] = (row['sparsity_test_acc_01'] - no_normalisation['sparsity_test_acc_01']) / no_normalisation['sparsity_test_acc_01'] 
    all_networks.loc[index, 'layerwise_normalisation_improvement_abs_test_acc_2'] = (row['sparsity_test_acc_2'] - no_normalisation['sparsity_test_acc_2'])
    if no_normalisation['sparsity_test_acc_2'] > 0:
      all_networks.loc[index, 'layerwise_normalisation_improvement_per_test_acc_2'] = (row['sparsity_test_acc_2'] - no_normalisation['sparsity_test_acc_2']) / no_normalisation['sparsity_test_acc_2'] 
    all_networks.loc[index, 'layerwise_normalisation_improvement_abs_test_acc_5'] = (row['sparsity_test_acc_5'] - no_normalisation['sparsity_test_acc_5'])
    if no_normalisation['sparsity_test_acc_5'] > 0:
      all_networks.loc[index, 'layerwise_normalisation_improvement_per_test_acc_5'] = (row['sparsity_test_acc_5'] - no_normalisation['sparsity_test_acc_5']) / no_normalisation['sparsity_test_acc_5'] 
    all_networks.loc[index, 'layerwise_normalisation_improvement_abs_test_acc_abs_1'] = (row['sparsity_test_acc_abs_1'] - no_normalisation['sparsity_test_acc_abs_1'])
    if no_normalisation['sparsity_test_acc_abs_1'] > 0:
      all_networks.loc[index, 'layerwise_normalisation_improvement_per_test_acc_abs_1'] = (row['sparsity_test_acc_abs_1'] - no_normalisation['sparsity_test_acc_abs_1']) / no_normalisation['sparsity_test_acc_abs_1'] 
    all_networks.loc[index, 'layerwise_normalisation_improvement_abs_test_acc_abs_01'] = (row['sparsity_test_acc_abs_01'] - no_normalisation['sparsity_test_acc_abs_01'])
    if no_normalisation['sparsity_test_acc_abs_01'] > 0:
      all_networks.loc[index, 'layerwise_normalisation_improvement_per_test_acc_abs_01'] = (row['sparsity_test_acc_abs_01'] - no_normalisation['sparsity_test_acc_abs_01']) / no_normalisation['sparsity_test_acc_abs_01'] 
    all_networks.loc[index, 'layerwise_normalisation_improvement_abs_test_acc_abs_2'] = (row['sparsity_test_acc_abs_2'] - no_normalisation['sparsity_test_acc_abs_2'])
    if no_normalisation['sparsity_test_acc_abs_2'] > 0:
      all_networks.loc[index, 'layerwise_normalisation_improvement_per_test_acc_abs_2'] = (row['sparsity_test_acc_abs_2'] - no_normalisation['sparsity_test_acc_abs_2']) / no_normalisation['sparsity_test_acc_abs_2'] 
    all_networks.loc[index, 'layerwise_normalisation_improvement_abs_test_acc_abs_5'] = (row['sparsity_test_acc_abs_5'] - no_normalisation['sparsity_test_acc_abs_5'])
    if no_normalisation['sparsity_test_acc_abs_5'] > 0:
      all_networks.loc[index, 'layerwise_normalisation_improvement_per_test_acc_abs_5'] = (row['sparsity_test_acc_abs_5'] - no_normalisation['sparsity_test_acc_abs_5']) / no_normalisation['sparsity_test_acc_abs_5']

networks = list(set(all_networks['network'].values))
layerwise_normalisation = list(set(all_networks['layerwise_normalisation'].values))
if 'N' in layerwise_normalisation:
  layerwise_normalisation.remove('N')

for n in networks:
  plt.figure()
  for l in layerwise_normalisation:
    if (l != 'no_normalisation'):
      sns.kdeplot(all_networks.loc[(all_networks['network']==n) & (all_networks['layerwise_normalisation']==l)]['layerwise_normalisation_improvement_abs_test_acc_5'], label=l)
  plt.title(n + ' : Distribution of Improvement on Sparsity for a 5% Drop in Top-1 Test Accuracy ')
  plt.ylabel('Density')
  plt.xlabel('Improvement in Sparsity for a 5% drop in Top-1 Test Accuracy')
  plt.savefig('graphs/overall_results/layerwise_normalisation_abs_test_acc_5_' + n + '.pdf')

#for l in layerwise_normalisation:
#  plt.figure()
#  for n in networks:
#    if (l != 'no_normalisation'):
#      sns.kdeplot(all_networks.loc[(all_networks['network']==n) & (all_networks['layerwise_normalisation']==l)]['layerwise_normalisation_improvement_abs_test_acc_5'], label=n)
#  plt.title(l + ' : Distribution of Improvement on Sparsity for a 5% Drop in Top-1 Test Accuracy ')
#  plt.ylabel('Density')
#  plt.xlabel('Improvement in Sparsity for a 5% drop in Top-1 Test Accuracy')
#  plt.savefig('graphs/overall_results/layerwise_normalisation_abs_test_acc_5_' + l + '.pdf')

for n in networks:
  plt.figure()
  for l in layerwise_normalisation:
    if (l != 'no_normalisation'):
      sns.kdeplot(all_networks.loc[(all_networks['network']==n) & (all_networks['layerwise_normalisation']==l)]['layerwise_normalisation_improvement_abs_test_acc_1'], label=l)
  plt.title(n + ' : Distribution of Improvement on Sparsity for a 1% Drop in Top-1 Test Accuracy ')
  plt.ylabel('Density')
  plt.xlabel('Improvement in Sparsity for a 1% drop in Top-1 Test Accuracy')
  plt.savefig('graphs/overall_results/layerwise_normalisation_abs_test_acc_1_' + n + '.pdf')

#for l in layerwise_normalisation:
#  plt.figure()
#  for n in networks:
#    if (l != 'no_normalisation'):
#      sns.kdeplot(all_networks.loc[(all_networks['network']==n) & (all_networks['layerwise_normalisation']==l)]['layerwise_normalisation_improvement_abs_test_acc_1'], label=n)
#  plt.title(l + ' : Distribution of Improvement on Sparsity for a 1% Drop in Top-1 Test Accuracy ')
#  plt.ylabel('Density')
#  plt.xlabel('Improvement in Sparsity for a 1% drop in Top-1 Test Accuracy')
#  plt.savefig('graphs/overall_results/layerwise_normalisation_abs_test_acc_1_' + l + '.pdf')
