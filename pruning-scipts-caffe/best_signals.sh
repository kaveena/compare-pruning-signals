#! /bin/bash

prune() {
  input=$1
  saliency_method=$2
  saliency_norm=$3
  scaling=$4
  input_lower=${input,,}
  saliency_method_lower=${saliency_method,,}
  norm_lower=${norm,,}
  for (( i=1; i<=20; i++ ))
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

saliency_input="ACTIVATION WEIGHT"
saliency_norm="NONE L1 L2 ABS_SUM SQR_SUM"
saliency_caffe="TAYLOR HESSIAN_DIAG_APPROX1 HESSIAN_DIAG_APPROX2 TAYLOR_2ND_APPROX1 TAYLOR_2ND_APPROX2 
AVERAGE_INPUT AVERAGE_GRADIENT APOZ"
scaling_python="l0_normalisation l1_normalisation l2_normalisation no_normalisation weights_removed l0_normalisation_adjusted"
default_save_path=$arch-$dataset/results/prune
force=false
characterise=false
while getopts ":fircoqp" arg; do
  case $arg in
    c ) # Display help.
      characterise=true
      retrain=false
      ;;
    f ) # Display help.
      force=true
      ;;
  esac
done
filename_prefix=$default_save_path/summary_
if [[ $characterise == true ]]
then
  filename_prefix=$filename_prefix\characterise_
else
  filename_prefix=$filename_prefix\sensitivity_
fi

mkdir -p $default_save_path
#  prune ACTIVATION TAYLOR_2ND_APPROX2 ABS_SUM weights_removed
