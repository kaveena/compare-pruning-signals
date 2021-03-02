#! /bin/bash
caffe_models=caffe-pruned-models
cifar10_data=../caffe-cifar-10-training
saliency_input="ACTIVATION WEIGHT"
saliency_norm="L1 ABS_SUM"
saliency_python="TAYLOR TAYLOR_2ND_APPROX2"
saliency_available=$saliency_caffe$saliency_python
scaling_python="l1_normalisation no_normalisation weights_removed"
scaling_saliency="none_scale"

default_save_path=$arch-$dataset/results/prune
force=false
input_channels=false
output_channels=false
skip_input_channels=false
skip_output_channels=false
retrain=false
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
    i ) # Display help.
      input_channels=true
      ;;
    o ) # Display help.
      output_channels=true
      ;;
    q ) # Display help.
      skip_input_channels=true
      ;;
    p ) # Display help.
      skip_output_channels=true
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
  if [[ $output_channels == true ]]
  then
    filename_prefix=$filename_prefix\input_output_channels_
  else
    filename_prefix=$filename_prefix\input_channels_
  fi
fi

saliency_scale=none_scale

mkdir -p $default_save_path
for (( i=1; i<=$iterations; i++ ))
  do
  if [[ $i == 1 ]] && [[ $characterise == false ]] && [[ $retrain == false ]]
  then
    use_stop_acc=10.0
  else
    use_stop_acc=$stop_acc
  fi
  input_lower=${input,,}
  saliency_method_lower=${saliency_method,,}
  norm_lower=${norm,,}
  filename=$filename_prefix\_activation_var_gradient_mean-none_norm-no_normalisation\_python_iter$i.npy
  if [[ ! -e $filename ]] || [[ $force == true ]]
  then
    filename_partial=$filename\.partial
    if [[ ! -e $filename_partial ]]
    then
      touch $filename_partial
      echo $filename
      GLOG_minloglevel=1 python compare_pruning_techniques_variance.py \
      \--filename $filename \
      \--arch $arch \
      \--dataset $dataset \
      \--stop-acc $use_stop_acc \
      \--characterise $characterise \
      \--retrain $retrain \
      \--test-interval $test_interval \
      \--test-size $test_size \
      \--eval-size $eval_size \
      \--train-size $train_size \
      \--scaling $scaling 
      rm $filename_partial
    fi
  fi
done
