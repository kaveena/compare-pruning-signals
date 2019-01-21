#! /bin/bash
arch=CIFAR10-CIFAR10
caffe_models=caffe-pruned-models
cifar10_data=../caffe-cifar-10-training
saliency_input="WEIGHT ACTIVATION"
saliency_norm="NONE L1 L2"
saliency_caffe="FISHER TAYLOR HESSIAN_DIAG HESSIAN_DIAG_APPROX2 TAYLOR_2ND TAYLOR_2ND_APPROX2 WEIGHT_AVG"
saliency_python="random apoz"
saliency_available=$saliency_caffe$saliency_python
normalisation_python="l0_normalisation l1_normalisation l2_normalisation no_normalisation"
masked_prototoxt=$caffe_models\/$arch\/solver\-gpu.prototxt
saliency_prototxt=$caffe_models/$arch/masked-one-saliency.prototxt
default_save_path=$arch/results/prune
mkdir -p $default_save_path
for input in $saliency_input
do
  for saliency_method in $saliency_caffe
  do
    for norm in $saliency_norm
    do
      for normalisation in $normalisation_python
      do
        if [[ "$saliency_method" == "FISHER" ]] && [[ "$norm" != "NONE" ]]
        then
          continue
        fi
        input_lower=${input,,}
        saliency_method_lower=${saliency_method,,}
        norm_lower=${norm,,}
        filename=$default_save_path/summary_$input_lower-$saliency_method_lower-$norm_lower\_norm-$normalisation\_caffe.npy
        if [[ ! -e  $filename ]]
        then
          echo $filename
          GLOG_minloglevel=1 python compare_pruning_techniques_caffe.py \
          \--arch $masked_prototoxt \
          \--arch-saliency $saliency_prototxt \
          \--pretrained $caffe_models/$arch/original.caffemodel \
          \--prune \
          \--filename $filename \
          \--stop-acc 10.0 \
          \--method $saliency_method \
          \--saliency-norm $norm \
          \--normalisation $normalisation \
          \--retrain \
          \--train-size 20 \
          \--saliency-input $input
        fi
      done
    done
  done
done

# random
filename=$default_save_path/summary_random_caffe.npy
if [[ ! -e  $filename ]]
then
  echo $filename
  GLOG_minloglevel=1 python compare_pruning_techniques_caffe.py \
  \--arch $masked_prototoxt \
  \--arch-saliency $saliency_prototxt \
  \--pretrained $caffe_models/$arch/original.caffemodel \
  \--prune \
  \--filename $filename \
  \--stop-acc 10.0 \
  \--retrain \
  \--train-size 20 \
  \--method random
fi

saliency_method="apoz"
for normalisation in $normalisation_python
do
  filename=$default_save_path/summary_$saliency_method-$normalisation\_caffe.npy
  if [[ ! -e  $filename ]]
  then
    echo $filename
    GLOG_minloglevel=1 python compare_pruning_techniques_caffe.py \
    \--arch $masked_prototoxt \
    \--arch-saliency $saliency_prototxt \
    \--pretrained $caffe_models/$arch/original.caffemodel \
    \--prune \
    \--filename $filename \
    \--stop-acc 10.0 \
    \--method $saliency_method \
    \--normalisation $normalisation \
    \--retrain \
    \--train-size 20 \
    \--saliency-input ACTIVATION
  fi
done
