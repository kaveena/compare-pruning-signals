#! /bin/bash
caffe_models=caffe-pruned-models
cifar10_data=../caffe-cifar-10-training
saliency_input="ACTIVATION WEIGHT"
#saliency_norm="NONE L1 L2 ABS_SUM SQR_SUM"
saliency_norm="L2 ABS_SUM "
#saliency_caffe="TAYLOR HESSIAN_DIAG HESSIAN_DIAG_APPROX2 TAYLOR_2ND TAYLOR_2ND_APPROX2 WEIGHT_AVG DIFF_AVG"
saliency_caffe="TAYLOR TAYLOR_2ND WEIGHT_AVG"
saliency_python="random apoz"
saliency_available=$saliency_caffe$saliency_python
#scaling_python="l0_normalisation l1_normalisation l2_normalisation no_normalisation weights_removed l0_normalisation_adjusted"
scaling_python="l2_normalisation no_normalisation weights_removed l0_normalisation_adjusted"
scaling_saliency="none_scale"
masked_prototoxt=$caffe_models\/$arch\/solver\-gpu.prototxt
saliency_prototxt=$caffe_models/$arch/masked-one-saliency.prototxt
default_save_path=$arch/results/prune
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
for input in $saliency_input
do
  for saliency_method in $saliency_caffe
  do
    for norm in $saliency_norm
    do
      for scaling in $scaling_python
      do
        input_lower=${input,,}
        saliency_method_lower=${saliency_method,,}
        norm_lower=${norm,,}
        filename=$filename_prefix$input_lower-$saliency_method_lower-$norm_lower\_norm-$scaling\_caffe.npy
        if [[ $skip_input_channels == true ]] && [[ $skip_output_channels == true ]]
        then 
          filename=$filename_prefix$input_lower-$saliency_method_lower-$norm_lower\_norm-$scaling\_skip_input_channels\_skip_output_channels\_caffe.npy
        elif [[ $skip_input_channels == true ]] && [[ $skip_output_channels == false ]]
        then 
          filename=$filename_prefix$input_lower-$saliency_method_lower-$norm_lower\_norm-$scaling\_skip_input_channels\_caffe.npy
        elif [[ $skip_input_channels == false ]] && [[ $skip_output_channels == true ]]
        then 
          filename=$filename_prefix$input_lower-$saliency_method_lower-$norm_lower\_norm-$scaling\_skip_output_channels\_caffe.npy
        fi
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
          \--input-channels $input_channels \
          \--output-channels $output_channels \
          \--skip-input-channels $skip_input_channels \
          \--skip-output-channels $skip_output_channels \
          \--test-interval $test_interval \
          \--test-size $test_size \
          \--eval-size $eval_size \
          \--train-size $train_size \
          \--method $saliency_method \
          \--saliency-norm $norm \
          \--scaling $scaling \
          \--saliency-input $input \
          \--saliency-scale $saliency_scale
        fi
      done
    done
  done
done
