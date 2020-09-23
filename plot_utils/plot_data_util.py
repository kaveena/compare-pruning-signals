
inputs = ['weight', 'activation']
pointwise_saliencies = ['average_input', 'average_gradient', 'taylor', 'taylor_2nd_approx2', 'hessian_diag_approx2', 'taylor_2nd_approx1', 'hessian_diag_approx1']
reductions = ['none_norm', 'l1_norm', 'l2_norm', 'abs_sum_norm', 'sqr_sum_norm']
scalings = ['no_normalisation', 'l1_normalisation', 'l2_normalisation', 'l0_normalisation_adjusted', 'weights_removed']

ylim_sensitivity = dict()
ylim_sensitivity['LeNet-5'] = {'CIFAR10': 25}
ylim_sensitivity['CIFAR10'] = {'CIFAR10': 45}
ylim_sensitivity['ResNet-20'] = {'CIFAR10': 15, 'CIFAR100': 5}
ylim_sensitivity['NIN'] = {'CIFAR10': 35, 'CIFAR100': 45}
ylim_sensitivity['AlexNet'] = {'CIFAR10': 65, 'CIFAR100': 60, 'IMAGENET32x32': 55}

ylim_retrain = dict()
ylim_retrain['LeNet-5'] = {'CIFAR10': 85}
ylim_retrain['CIFAR10'] = {'CIFAR10': 90}
ylim_retrain['ResNet-20'] = {'CIFAR10': 25, 'CIFAR100': 10}
ylim_retrain['NIN'] = {'CIFAR10': 75, 'CIFAR100': 60}
ylim_retrain['AlexNet'] = {'CIFAR10': 75, 'CIFAR100': 65, 'IMAGENET32x32': 55}

ylim_characterise = dict()
ylim_characterise['LeNet-5'] = {'CIFAR10': 85}
ylim_characterise['CIFAR10'] = {'CIFAR10': 75}
ylim_characterise['ResNet-20'] = {'CIFAR10': 30, 'CIFAR100': 15}
ylim_characterise['NIN'] = {'CIFAR10': 75, 'CIFAR100': 65}
ylim_characterise['AlexNet'] = {'CIFAR10': 75, 'CIFAR100': 65, 'IMAGENET32x32': 55}

networks = ['LeNet-5', 'CIFAR10', 'ResNet-20', 'NIN', 'AlexNet']
datasets = ['CIFAR10', 'CIFAR100', 'IMAGENET32x32']

networks_dict_1 = { 'LeNet-5': ['CIFAR10'],
                  'CIFAR10': ['CIFAR10'],
                  'ResNet-20': ['CIFAR10', 'CIFAR100'],
                  'NIN': ['CIFAR10', 'CIFAR100'],
                  'AlexNet': ['CIFAR10', 'CIFAR100', 'IMAGENET32x32']}

networks_dict_2 = { 'LeNet-5': ['CIFAR10'],
                  'CIFAR10': ['CIFAR10'],
                  'ResNet-20': ['CIFAR10', 'CIFAR100'],
                  'NIN': ['CIFAR10', 'CIFAR100'],
                  'AlexNet': ['CIFAR10', 'CIFAR100']}

networks_dict_3 = { 'LeNet-5': ['CIFAR10'],
                  'CIFAR10': ['CIFAR10'],
                  'ResNet-20': ['CIFAR10', 'CIFAR100'],
                  'NIN': ['CIFAR10', 'CIFAR100'],
                  'AlexNet': ['CIFAR10', 'CIFAR100', 'IMAGENET32x32']}

max_sparsity = {'LeNet-5-CIFAR10':        84.9,
                'CIFAR10-CIFAR10':        67.5,
                'ResNet-20-CIFAR10':      25.4,
                'NIN-CIFAR10':            73.8,
                'AlexNet-CIFAR10':        70.1,
                'ResNet-20-CIFAR100':     12.5,
                'NIN-CIFAR100':           59.3,
                'AlexNet-CIFAR100':       64.0,
                'AlexNet-IMAGENET32x32':  55.4}

total_channels = dict()
total_channels['LeNet-5'] = {'CIFAR10': 70}
total_channels['CIFAR10'] = {'CIFAR10': 128}
total_channels['ResNet-20'] = {'CIFAR10': 784, 'CIFAR100': 784}
total_channels['NIN'] = {'CIFAR10': 1418, 'CIFAR100': 1508}
total_channels['AlexNet'] = {'CIFAR10': 1376, 'CIFAR100': 1376, 'IMAGENET32x32': 1376}

accuracies = {  'LeNet-5-CIFAR10':        69.35,
                'CIFAR10-CIFAR10':        72.76,
                'ResNet-20-CIFAR10':      88.42,
                'NIN-CIFAR10':            88.26,
                'AlexNet-CIFAR10':        84.22,
                'ResNet-20-CIFAR100':     59.22,
                'NIN-CIFAR100':           65.7,
                'AlexNet-CIFAR100':       54.15,
                'AlexNet-IMAGENET32x32':  39.69}
