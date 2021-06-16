import caffe
import os
import struct
import sys
import random
import numpy as np
import caffe
import argparse
import random
import time
from utils.utils import *
from utils.pruning_graph import *
import tempfile

sys.dont_write_bytecode = True

def parser():
    parser = argparse.ArgumentParser(description='Caffe Channel Pruning Example')
    parser.add_argument('--arch', action='store', default='AlexNet',
            help='the network architecture to use')
    parser.add_argument('--dataset', action='store', default='CIFAR10',
            help='the dataset to use')
    parser.add_argument('--retrain', type=str2bool, nargs='?', default=False,
            help='retrain the pruned network')
    parser.add_argument('--characterise', type=str2bool, nargs='?', default=False,
            help='characterise the pruned network')
    parser.add_argument('--tolerance', type=float, default='1.0',
            help='Drop in train loss before retraining starts')
    parser.add_argument('--filename', action='store', default='summary_',
            help='prefix for storing pruning data')
    parser.add_argument('--stop-acc', type=float, default='10.0',
            help='Stop pruning when test accuracy drops below this value')
    parser.add_argument('--saliency-pointwise', action='store', default='TAYLOR_2ND_APPROX2',
            help='pointwise saliency')
    parser.add_argument('--saliency-norm', action='store', default='NONE',
            help='Caffe saliency_norm')
    parser.add_argument('--saliency-input', action='store', default='WEIGHT',
            help='Caffe saliency_input')
    parser.add_argument('--scaling', action='store', default='l1_normalisation',
            help='Layer-wise scaling to use for saliency')
    parser.add_argument('--test-size', type=int, default=80, 
            help='Number of batches to use for testing')
    parser.add_argument('--train-size', type=int, default=20, 
            help='Number of batches to use for training')
    parser.add_argument('--eval-size', type=int, default=2, 
            help='Number of batches to use for evaluating the saliency')
    parser.add_argument('--test-interval', type=int, default=1, 
            help='After how many pruning steps to test')
    parser.add_argument('--input-channels', type=str2bool, nargs='?', default=True,
            help='consider saliency as input channels')
    parser.add_argument('--output-channels', type=str2bool, nargs='?', default=True,
            help='consider saliency as output channels')
    parser.add_argument('--skip-input-channels-only', type=str2bool, nargs='?', default=True,
            help='skip the channels that are only input channels')
    parser.add_argument('--skip-output-channels-only', type=str2bool, nargs='?', default=False,
            help='skip the channels that are only output channels')
    parser.add_argument('--saliency-scale', action='store', default='none_scale',
            help='Layer-wise scaling to use for saliency')
    parser.add_argument('--remove-all-nodes', type=str2bool, nargs='?', default=True,
            help='remove all connected channels when removing a channel')
    parser.add_argument('--transitive-saliency', type=str2bool, nargs='?', default=False,
            help='remove all connected channels when removing a channel')
    return parser

start = time.time()
args = parser().parse_args()

if args.input_channels and args.output_channels:
  args.transitive_saliency = True

if args.arch is None:
  print("Caffe network needed")
  exit(1)

if args.dataset is None:
  print("Dataset required")
  exit(1)

forward_passes = args.eval_size
backward_passes = args.eval_size
if args.saliency_pointwise == 'random':
  forward_passes = 0
  backward_passes = 0
elif args.saliency_pointwise == 'AVG' and args.saliency_input == 'WEIGHT':
  forward_passes = 1
  backward_passes = 0
elif args.saliency_pointwise == 'AVG' or args.saliency_pointwise == 'APOZ':
  backward_passes = 0

eval_index_filename = './caffe-training-data/' + args.dataset + '/eval-index.txt'

saliency_prototxt_file, saliency_prototxt_filename = tempfile.mkstemp()
masked_prototxt_file, masked_prototxt_filename = tempfile.mkstemp()
solver_prototxt_file, solver_prototxt_filename = tempfile.mkstemp()

arch_solver = 'trained-models/' + args.arch + '-' + args.dataset + '/solver-gpu.prototxt'
pretrained = 'trained-models/' + args.arch + '-' + args.dataset + '/original.caffemodel'
solver_prototxt = get_solver_proto_from_file(arch_solver)

# Create a new solver prototxt from original solver
original_prototxt_filename = solver_prototxt.net
solver_prototxt.net = masked_prototxt_filename
# Create a new network that uses mask blobs to be used with the solver
masked_prototxt = get_prototxt_from_file(original_prototxt_filename)
add_mask_to_prototxt(masked_prototxt)

# Create a prototxt that uses the validation set and has saliency computation
saliency_prototxt = get_prototxt_from_file(original_prototxt_filename)
add_mask_to_prototxt(saliency_prototxt)
for l in saliency_prototxt.layer:
  if l.type == 'ImageData':
    l.image_data_param.batch_size = 128
    l.image_data_param.source = eval_index_filename
    l.image_data_param.shuffle = True
    if (args.arch == 'ResNet-20') and (args.dataset == 'IMAGENET2012'):
      l.image_data_param.batch_size = 32
    if (args.arch == 'ResNet-50') and (args.dataset == 'IMAGENET2012'):
      l.image_data_param.batch_size = 8
    if (args.arch == 'NFNET-F0') and (args.dataset == 'IMAGENET2012'):
      l.image_data_param.batch_size = 8
#    l.data_param.source = eval_index_filename
add_saliency_to_prototxt(saliency_prototxt, [args.saliency_pointwise], [args.saliency_input], [args.saliency_norm], args.output_channels, args.input_channels)
# if second derivative computation is required...
if args.saliency_pointwise == 'TAYLOR_2ND_APPROX1' or args.saliency_pointwise == 'HESSIAN_DIAG_APPROX1':
  saliency_prototxt.compute_2nd_derivative = True
  solver_prototxt.allocate_2nd_derivative_memory = True
else:
  saliency_prototxt.compute_2nd_derivative = False
  masked_prototxt.compute_2nd_derivative = False

save_prototxt_to_file(solver_prototxt, solver_prototxt_filename)
save_prototxt_to_file(masked_prototxt, masked_prototxt_filename)
save_prototxt_to_file(saliency_prototxt, saliency_prototxt_filename)
save_prototxt_to_file(solver_prototxt, "new_solver.prototxt")
save_prototxt_to_file(masked_prototxt, "masked.prototxt")
save_prototxt_to_file(saliency_prototxt, "saliency.prototxt")

caffe.set_mode_gpu()

#  Allocate two models : one for masking weights and retraining (weights + mask)
#                          one for computing saliency (weights + masks + saliency)
# net with only mask blobs
saliency_solver = caffe.SGDSolver(solver_prototxt_filename)
saliency_solver.net.copy_from(pretrained)
# net with saliency and mask blobs
net = caffe.Net(saliency_prototxt_filename, caffe.TEST)
# share the blobs between all networks
saliency_solver.test_nets[0].share_with(saliency_solver.net)
net.share_with(saliency_solver.net) 

# global pruning helpers
pruning_net = PruningGraph(net, saliency_prototxt)

# global pruning helpers
method = args.saliency_pointwise
initial_density = 0
total_param = 0 
total_output_channels = 0

net.reshape()

print('Total number of channels to be considered for pruning: ', pruning_net.total_output_channels)

summary = dict()
summary['test_acc'] = np.zeros(pruning_net.total_output_channels)
summary['test_loss'] = np.zeros(pruning_net.total_output_channels)
summary['pruned_channel'] = np.zeros(pruning_net.total_output_channels)
summary['method'] = method + '-' + args.scaling
active_channel = list(range(pruning_net.total_output_channels))
summary['initial_conv_param'], summary['initial_fc_param'] = pruning_net.GetNumParam()
summary['initial_test_acc'], summary['initial_test_loss'] = test_network(saliency_solver.test_nets[0], args.test_size)
summary['eval_loss'] = np.zeros(pruning_net.total_output_channels)
summary['eval_acc'] = np.zeros(pruning_net.total_output_channels)
summary['predicted_eval_loss'] = np.zeros(pruning_net.total_output_channels)
summary['conv_param'] = np.zeros(pruning_net.total_output_channels)
summary['fc_param'] = np.zeros(pruning_net.total_output_channels)

summary['initial_eval_acc'], summary['initial_eval_loss'] = test_network(net, 100)
print('Initial eval loss', summary['initial_eval_loss'])
print('Initial eval acc', summary['initial_eval_acc'])

# Train accuracy and loss
summary['initial_train_acc'], summary['initial_train_loss'] = test_network(saliency_solver.net, 100)
train_loss_upper_bound = (100 + args.tolerance) * summary['initial_train_loss'] / 100.0
train_acc_lower_bound = (100 - args.tolerance) * summary['initial_train_acc'] / 100.0
if args.characterise:
  summary['train_loss'] = np.zeros(pruning_net.total_output_channels)
  summary['train_acc'] = np.zeros(pruning_net.total_output_channels)
  summary['retraining_loss'] = np.empty((pruning_net.total_output_channels,), dtype=object)
  summary['retraining_acc'] = np.empty((pruning_net.total_output_channels,), dtype=object)


for j in range(pruning_net.total_output_channels):
  pruning_net.ClearSaliencyBlobs()

  # compute saliency    
  current_eval_loss = 0.0
  current_eval_acc = 0.0
  for iter in range(forward_passes):
    net.clear_param_diffs()
    output = net.forward()
    if iter < backward_passes:
      net.backward()
    current_eval_loss += output['loss']
    current_eval_acc += output['top-1']
  summary['eval_loss'][j] = current_eval_loss / float(iter+1)
  summary['eval_acc'][j] = current_eval_acc / float(iter+1)

  if method == 'random':
    pruning_signal = np.zeros(net.total_output_channels)
    pruning_signal[random.sample(active_channel, 1)] = -1
  else:
    pruning_signal = saliency_scaling_regular(pruning_net, output_saliency=args.output_channels, input_saliency=args.input_channels, transitive=args.transitive_saliency, input_saliency_type=args.saliency_input, scaling=args.scaling)
    pruning_signal /= float(iter+1)

  prune_channel_idx = np.argmin(pruning_signal[active_channel])
  prune_channel = active_channel[prune_channel_idx]
  pruning_net.PruneChannel(prune_channel, final=True, remove_all_nodes=args.remove_all_nodes)

  if args.characterise:
    output_train = saliency_solver.net.forward()
    current_loss = output_train['loss']
    current_acc = 100.0 * output_train['top-1']
    summary['train_loss'][j] = current_loss
    summary['train_acc'][j] = current_acc
    retraining_loss = np.array([])
    retraining_acc = np.array([])
    for i in range(args.train_size):
      if ((i==0) and (current_acc >= train_acc_lower_bound)):
        break
      if (current_acc >= summary['initial_train_acc']):
        break
      saliency_solver.net.clear_param_diffs()
      output_train = saliency_solver.net.forward()
      current_loss = output_train['loss']
      current_acc = 100.0 * output_train['top-1']
      saliency_solver.net.backward()
      saliency_solver.apply_update()
      retraining_loss = np.hstack([retraining_loss, current_loss])
      retraining_acc = np.hstack([retraining_acc, current_acc])
    summary['retraining_loss'][j] = retraining_loss.copy()
    summary['retraining_acc'][j] = retraining_acc.copy()

  if args.retrain:
    saliency_solver.step(args.train_size)

  if (j % args.test_interval == 0):
    test_acc, ce_loss = test_network(saliency_solver.test_nets[0], args.test_size)
  summary['test_acc'][j] = test_acc
  summary['test_loss'][j] = ce_loss
  summary['pruned_channel'][j] = prune_channel
  summary['predicted_eval_loss'][j] = (pruning_signal[active_channel])[prune_channel_idx]
  summary['conv_param'][j], summary['fc_param'][j] = pruning_net.GetNumParam()
  print(args.scaling, method, ' Step: ', j +1,'  ||   Remove Channel: ', prune_channel, '  ||  Test Acc: ', test_acc)
  active_channel.remove(prune_channel)
  if args.remove_all_nodes:
    for transitive_channel in pruning_net.output_channel_sources[prune_channel]:
      if transitive_channel in active_channel:
        active_channel.remove(transitive_channel)
  
  if test_acc < args.stop_acc or len(active_channel) < 1:
      break

end = time.time()
summary['exec_time'] = end - start
np.save(args.filename, summary)

caffe.set_mode_cpu()
