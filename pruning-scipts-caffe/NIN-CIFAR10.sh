#! /bin/bash
arch=NIN-CIFAR10
train_size=20
eval_size=10
test_size=40
test_interval=50
tolerance=10.0
caffe_models=caffe-pruned-models
cifar10_data=../caffe-cifar-10-training
saliency_input="ACTIVATION WEIGHT"
saliency_norm="NONE L1 L2"
saliency_caffe="FISHER TAYLOR HESSIAN_DIAG HESSIAN_DIAG_APPROX2 TAYLOR_2ND TAYLOR_2ND_APPROX2 WEIGHT_AVG DIFF_AVG"
saliency_python="random apoz"
saliency_available=$saliency_caffe$saliency_python
normalisation_python="l0_normalisation l1_normalisation l2_normalisation no_normalisation"
masked_prototoxt=$caffe_models\/$arch\/solver\-gpu.prototxt
saliency_prototxt=$caffe_models/$arch/masked-one-saliency.prototxt
default_save_path=$arch/results/prune
force=false
input_channels=false
retrain=false
characterise=false
while getopts ":firc" arg; do
  case $arg in
    c ) # Display help.
      characterise=true
      retrain=false
      ;;
    f ) # Display help.
      force=true
      ;;
    i ) # Display help.
      input_channels=true
      ;;
    r ) # Display help.
      retrain=true
      characterise=false
      ;;
  esac
done
filename_prefix=$default_save_path/summary_
if [[ $characterise == true ]]
then
  filename_prefix=$filename_prefix\characterise_
elif [[ $retrain == true ]]
then
  filename_prefix=$filename_prefix\retrain_
else
  filename_prefix=$filename_prefix\sensitivity_
fi
if [[ $input_channels == true ]]
then
  filename_prefix=$filename_prefix\input_channels_
fi
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
        filename=$filename_prefix$input_lower-$saliency_method_lower-$norm_lower\_norm-$normalisation\_caffe.npy
        if [[ ! -e $filename ]] || [[ $force == true ]]
        then
          echo $filename
          GLOG_minloglevel=1 python compare_pruning_techniques_caffe.py \
          \--filename $filename \
          \--arch $masked_prototoxt \
          \--arch-saliency $saliency_prototxt \
          \--pretrained $caffe_models/$arch/original.caffemodel \
          \--stop-acc 10.0 \
          \--characterise $characterise \
          \--retrain $retrain \
          \--input-channels-only $input_channels \
          \--test-interval $test_interval \
          \--test-size $test_size \
          \--eval-size $eval_size \
          \--train-size $train_size \
          \--method $saliency_method \
          \--saliency-norm $norm \
          \--normalisation $normalisation \
          \--saliency-input $input
        fi
      done
    done
  done
done

# random
filename=$filename_prefix\random_caffe.npy
if [[ ! -e  $filename ]] || [[ $force == true ]]
then
  echo $filename
  GLOG_minloglevel=1 python compare_pruning_techniques_caffe.py \
  \--filename $filename \
  \--arch $masked_prototoxt \
  \--arch-saliency $saliency_prototxt \
  \--pretrained $caffe_models/$arch/original.caffemodel \
  \--stop-acc 10.0 \
  \--characterise $characterise \
  \--retrain $retrain \
  \--input-channels-only $input_channels \
  \--test-interval $test_interval \
  \--test-size $test_size \
  \--eval-size $eval_size \
  \--train-size $train_size \
  \--method random
fi

saliency_method="apoz"
for normalisation in $normalisation_python
do
  filename=$filename_prefix$saliency_method-$normalisation\_caffe.npy
  if [[ ! -e  $filename ]] || [[ $force == true ]]
  then
    echo $filename
    GLOG_minloglevel=1 python compare_pruning_techniques_caffe.py \
    \--filename $filename \
    \--arch $masked_prototoxt \
    \--arch-saliency $saliency_prototxt \
    \--pretrained $caffe_models/$arch/original.caffemodel \
    \--stop-acc 10.0 \
    \--characterise $characterise \
    \--retrain $retrain \
    \--input-channels-only $input_channels \
    \--test-interval $test_interval \
    \--test-size $test_size \
    \--eval-size $eval_size \
    \--train-size $train_size \
    \--method $saliency_method \
    \--normalisation $normalisation \
    \--saliency-input ACTIVATION
  fi
done
