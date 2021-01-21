#! /bin/bash

prune() {
  input=$1
  saliency_method=$2
  norm=$3
  scaling=$4
  input_lower=${input,,}
  saliency_method_lower=${saliency_method,,}
  norm_lower=${norm,,}
  for (( i=1; i<=16; i++ ))
    do
    if [[ $i == 1 ]] && [[ $characterise == false ]] && [[ $retrain == false ]]
    then
      use_stop_acc=10.0
    else
      use_stop_acc=$stop_acc
    fi
    filename=$filename_prefix$input_lower-$saliency_method_lower-$norm_lower\_norm-$scaling\_caffe_iter$i.npy
    skip=false
    if [[ ! -e $filename ]] || [[ $force == true ]] && [[ $skip == false ]]
    then
      filename_partial=$filename\.partial
      if [[ ! -e $filename_partial ]]
      then
        touch $filename_partial
        echo $filename
        GLOG_minloglevel=1 python compare_pruning_techniques_caffe.py \
        \--filename $filename \
        \--arch $arch \
        \--dataset $dataset \
        \--stop-acc $use_stop_acc \
        \--characterise $characterise \
        \--retrain False \
        \--test-interval $test_interval \
        \--test-size $test_size \
        \--eval-size $eval_size \
        \--train-size $train_size \
        \--saliency-pointwise $saliency_method \
        \--saliency-norm $norm \
        \--scaling $scaling \
        \--saliency-input $input \
        \--saliency-scale $saliency_scale
        rm $filename_partial
      fi
    fi
  done
}

default_save_path=$arch-$dataset/results/prune
filename_prefix=$default_save_path/summary_
if [[ $characterise == true ]]
then
  filename_prefix=$filename_prefix\characterise_
else
  filename_prefix=$filename_prefix\sensitivity_
fi
mkdir -p $default_save_path
