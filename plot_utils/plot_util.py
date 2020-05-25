import scipy
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from functools import reduce
import matplotlib.colors as mcolors
from utils import *

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def convert_label(label):
  labels = label.split('-')
  l = ' f(x) '
  l_prefix = ''
  l_suffix = ''
  if ('none_norm' in labels) or ('N' in labels):
    l_prefix = ' \sum'
    l_suffix = ' '
  elif 'l1_norm' in labels:
    l_prefix = ' \sum | '
    l_suffix = ' | '
  elif 'l2_norm' in labels:
    l_prefix = ' \sum ( '
    l_suffix = ' )^2 '
  elif 'abs_sum_norm' in labels:
    l_prefix = ' | \sum '
    l_suffix = ' | '
  elif 'sqr_sum_norm' in labels:
    l_prefix = ' ( \sum  '
    l_suffix = ' )^2'
  
  if 'input_average' in labels:
    l = ' x '
  elif 'gradient_average' in labels:
    l = ' \\frac{d\mathcal{L}}{dx}'
  elif 'taylor_2nd_approx2' in labels:
    l = ' -x \\frac{d\mathcal{L}}{dx} + \\frac{x^2}{2}\\frac{d^2\mathcal{L}}{dx^2}_{app. 2} (x)'
  elif 'taylor_2nd_approx1' in labels:
    l = ' -x \\frac{d\mathcal{L}}{dx} + \\frac{x^2}{2}\\frac{d^2\mathcal{L}}{dx^2}_{app. 1} (x)'
  elif 'taylor' in labels:
    l = ' -x \\frac{d\mathcal{L}}{dx} (x)'
  elif 'hessian_diag_approx2' in labels:
    l = ' \\frac{x^2}{2} \\frac{d^2\mathcal{L}}{dx^2}_{app. 2}(x)'
  elif 'hessian_diag_approx1' in labels:
    l = ' \\frac{x^2}{2} \\frac{d^2\mathcal{L}}{dx^2}_{app. 1} (x)'
  elif 'apoz' in labels:
    l = 'APoZ'
  
  if 'l0_normalisation' in labels:
    l_prefix = ' \\frac{1}{card(x)}' + l_prefix
  elif 'l1_normalisation' in labels:
    l_prefix = ' \\frac{1}{ \| f(x) \| _1}' + l_prefix
  elif 'l2_normalisation' in labels:
    l_prefix = ' \\frac{1}{ \| f(x) \| _2}' + l_prefix
  elif 'weights_removed' in labels:
    l_prefix = ' \\frac{1}{n.weights}' + l_prefix
  elif 'l0_normalisation_adjusted' in labels:
    l_prefix = ' \\frac{1}{card(x_{pruned})}' + l_prefix
  
  l_prefix = '$' + l_prefix
  l_suffix = l_suffix + '$' 
  
  if 'activation' in labels:
    l_prefix = l_prefix.replace('x', 'a')
    l_suffix = l_suffix.replace('x', 'a')
    l = l.replace('x', 'a')
  elif 'weight' in labels:
    l_prefix = l_prefix.replace('x', 'w')
    l_suffix = l_suffix.replace('x', 'w')
    l = l.replace('x', 'w')
  if 'random' in labels:
    l = 'Random'
    l_prefix = ''
    l_suffix = ''
  return l_prefix + l + l_suffix
0

def choose_linewidth(method):
    if 'oracle' in method:
        return 2.0
    else:
        return 1.0

def get_color_normalisation(normalisation):
  normalisation_to_color = {'no_normalisation':'r', 'l0_normalisation':'g', 'l1_normalisation':'b', 'l2_normalisation': 'y', 'l0_normalisation_adjusted': 'c', 'weights_removed': 'm'}
  return normalisation_to_color[normalisation]

def get_norm_hatch(norm):
  norm_to_hatch = {'none_norm': '//', 'l1_norm': '\\', 'l2_norm': '+', 'abs_sum_norm': 'o', 'sqr_sum_norm': '*'}
  return norm_to_hatch[norm]

def get_color(norm, normalisation):
  plot_colors = dict()
  norm_to_int = {'no_normalisation':0, 'l0_normalisation':1, 'l1_normalisation':2, 'l2_normalisation': 3, 'l0_normalisation_adjusted': 4, 'weights_removed': 5}
  plot_colors['none_norm'] = ['rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown']
  plot_colors['l1_norm'] = ['olive', 'yellowgreen', 'lawngreen', 'lightgreen', 'darkgreen', 'mediumseagreen']
  plot_colors['l2_norm'] = ['teal', 'cyan', 'lightblue', 'steelblue', 'cornflowerblue', 'blue']
  plot_colors['abs_sum_norm'] = ['pink', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'hotpink']
  plot_colors['sqr_sum_norm'] = ['indigo', 'darkorchid', 'darkviolet', 'rebeccapurple', 'purple', 'darkmagenta']
  return mcolors.CSS4_COLORS[plot_colors[norm][norm_to_int[normalisation]]]

def choose_label(method):
    if '_retrained' in method:
        method = method.split('_retrained')[0]
    if 'augmented' in method:
        label = method.split('augmented_')[1].split('_top')[0]
        if 'hybrid' not in label:
            label = label + ' + oracle'
    else:
        label = method
    return label

def choose_network(arch):
    if arch=='vgg19':
        return 'VGG-19'
    if 'resnet' in arch:
        return 'ResNet-' + arch.split('resnet')[1]
    if arch=='LeNet_5':
        return 'LeNet-5'
    return arch

def choose_linestyle(method):
    if 'augmented' in method:
        return '-'
    else:
        return '--'


