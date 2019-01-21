import scipy
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def choose_linewidth(method):
    if 'oracle' in method:
        return 2.0
    else:
        return 1.0

def choose_color(method):
    if 'fisher' in method:
        return colors[1]    #orange
    elif 'taylor-abbs-norm' in method:
        return colors[0]    #blue
    elif 'min-weight' in method:
        return colors[2]    #green
    elif 'mean-act' in method:
        return colors[3]    #red
    elif 'apoz' in method:
        return colors[4]    #purple
    elif 'l1-norm-weight' in method:
        return colors[5]    #brown
    elif 'random' in method:
        return colors[6]    #pink
    elif 'rms-grad-weight' in method:
        return colors[8]    #
    elif 'hybrid' in method:
        return colors[9]    #cyan
    elif 'oracle' in method:
        return '#ffffff'    #black
    else:
        return colors[7]    #grey

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

def get_corresponding_channel_resnet(channels_so_far_pruned, layer_1, local_channel_1, layer_2, convolution_original, convolution_list, channels):
    m1, c1, k1, _ = convolution_original[layer_1] 
    m2, c2, k2, _ = convolution_original[layer_2] # in resnets channels only increase
    if m1 > m2 :
        if (local_channel_1 < m2 //2)  or (local_channel_1 > (m2 - 1 + (m2//2))):
            return True
        else:
            local_channel_2 = local_channel_1 - (m2 //2)
    elif m2 < m1 : 
            local_channel_2 = local_channel_1 + (m2 //2)
    else:
        local_channel_2 = local_channel_1
    global_channel_2 = local_channel_2 if convolution_list.index(layer_2) == 0 else channels[convolution_list.index(layer_2)-1] + local_channel_2
    if (global_channel_2 in channels_so_far_pruned):
        return True
    else:
        return False

def correct_sparsity(summary, convolution_list, convolution_original, convolution_previous, convolution_next, convolution_summary, channels, arch, stop_itr=0):
    total_channels = len(summary['test_acc'])
    old_sparsity = summary['num_param']
    correct_param = np.zeros(total_channels)
    initial_param = summary['initial_param']
    channels_so_far_pruned = []
    for i in range(total_channels):
        pruned_channel = summary['pruned_channel'][i]
        channels_so_far_pruned.append(pruned_channel)
        if old_sparsity[i] == 0:
            break
        idx = np.where(channels>pruned_channel)[0][0]
        idx_convolution = convolution_list[idx]
        idx_channel = (pruned_channel - channels[idx-1]) if idx > 0 else pruned_channel
        m_0, c_0, k_0, _ = convolution_summary[idx_convolution]
        initial_param -= c_0 * k_0 * k_0
        convolution_summary[idx_convolution][0] -= 1 # Decrease the number of output features of the current layer by one
        if idx_convolution in convolution_next.keys(): # If the next layer is a conv layer
            l = convolution_next[idx_convolution]
            if arch.startswith('resnet'):
                for output_layer in l: #consumer layers of currently pruned channel
                    safe_to_prune = True
                    for input_layer in convolution_previous[output_layer]:# producer layers of consumer layer, we need to check that all producers of the current consumer are pruned before pruning the consumer
                        safe_to_prune &= get_corresponding_channel_resnet(channels_so_far_pruned, idx_convolution, idx_channel, input_layer, convolution_original, convolution_list, channels)
                    if safe_to_prune:
                        m_1, c_1, k_1, _ = convolution_summary[output_layer]
                        initial_param -= m_1 * k_1 * k_1
                        convolution_summary[output_layer][1] -= 1
            else:
                m_1, c_1, k_1, _ = convolution_summary[l]
                initial_param -= m_1 * k_1 * k_1
                convolution_summary[l][1] -= 1
        correct_param[i] = initial_param
        if (stop_itr>0) and (i == stop_itr):
            break
    return correct_param
