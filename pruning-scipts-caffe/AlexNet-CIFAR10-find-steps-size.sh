#! /bin/bash
arch=AlexNet-CIFAR10
caffe_models=caffe-pruned-models
cifar10_data=../caffe-cifar-10-training
saliency_input="WEIGHT ACTIVATION"
saliency_norm="NONE L1 L2"
saliency_caffe="FISHER TAYLOR HESSIAN_DIAG HESSIAN_DIAG_APPROX2 TAYLOR_2ND TAYLOR_2ND_APPROX2 WEIGHT_AVG DIFF_AVG"
saliency_python="random apoz"
saliency_available=$saliency_caffe$saliency_python
normalisation_python="l0_normalisation l1_normalisation l2_normalisation no_normalisation"
masked_prototoxt=$caffe_models\/$arch\/solver\-gpu.prototxt
saliency_prototxt=$caffe_models/$arch/masked-one-saliency.prototxt
default_save_path=$arch/results/prune

all_eval_size="2 4 8 16 32 64"
all_test_size="2 4 8 16 32 64 128"
all_train_size="2 4 8 16 32 64 128 256 512"


mkdir -p $default_save_path
for train_size in $all_train_size
do
  for test_size in $all_test_size
  do
    for eval_size in $all_eval_size
    do
      filename=$default_save_path/summary_activation-fisher-none_norm-no_normalisation-$eval_size-$test_size-$train_size\_caffe.npy
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
        \--method FISHER \
        \--saliency-norm NONE \
        \--normalisation no_normalisation \
        \--eval-size $eval_size \
        \--test-size $test_size \
        \--train-size $train_size \
        \--test-interval 50 \
        \--retrain \
        \--saliency-input ACTIVATION
      fi
    done
  done
done

eval_size=1
for train_size in $all_train_size
do
  for test_size in $all_test_size
  do
    filename=$default_save_path/summary_weight-weight_avg-l1_norm-no_normalisation-$eval_size-$test_size-$train_size\_caffe.npy
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
    \--method WEIGHT_AVG \
    \--saliency-norm L1 \
    \--normalisation no_normalisation \
    \--eval-size $eval_size \
    \--test-size $test_size \
    \--train-size $train_size \
    \--test-interval 50 \
    \--retrain \
    \--saliency-input WEIGHT
    fi
  done
done
