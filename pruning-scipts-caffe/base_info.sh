#! /bin/bash
caffe_models=caffe-pruned-models
cifar10_data=../caffe-cifar-10-training
saliency_input="ACTIVATION WEIGHT"
saliency_norm="NONE L1 L2 ABS_SUM SQR_SUM"
saliency_caffe="TAYLOR HESSIAN_DIAG_APPROX1 HESSIAN_DIAG_APPROX2 TAYLOR_2ND_APPROX1 TAYLOR_2ND_APPROX2 
AVERAGE_INPUT AVERAGE_GRADIENT APOZ"
saliency_python="random"
saliency_available=$saliency_caffe$saliency_python
#scaling_python="l0_normalisation l1_normalisation l2_normalisation no_normalisation weights_removed l0_normalisation_adjusted"
scaling_python="l2_normalisation no_normalisation weights_removed l0_normalisation_adjusted l1_normalisation"
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
          filename=$filename_prefix$input_lower-$saliency_method_lower-$norm_lower\_norm-$scaling\_caffe_iter$i.npy
          skip=false
          if [[ $saliency_method == HESSIAN_DIAG_APPROX2 ]]
          then 
            if [[ $norm == NONE ]] || [[ $norm == L1 ]] || [[ $norm == ABS_SUM ]]
            then
              skip=true
            fi
          fi
          if [[ $saliency_method == TAYLOR ]] && [[ $input == WEIGHT ]]
          then 
            if [[ $norm == NONE ]] || [[ $norm == ABS_SUM ]] || [[ $norm == SQR_SUM ]]
            then
              if [[ $scaling == no_normalisation ]] || [[ $scaling == l1_normalisation ]] || [[ $scaling == l2_normalisation ]] || [[ $scaling == weights_removed ]]
              then
                skip=true
              fi
            fi
          fi
          if [[ $saliency_method == APOZ ]]
          then 
            if [[ $norm == ABS_SUM ]] || [[ $norm == L2 ]] || [[ $norm == L1 ]]
            then
              skip=true
            fi
          fi
          if [[ $skip_input_channels == true ]] && [[ $skip_output_channels == true ]]
          then 
            filename=$filename_prefix$input_lower-$saliency_method_lower-$norm_lower\_norm-$scaling\_skip_input_channels\_skip_output_channels\_caffe_iter$i.npy
          elif [[ $skip_input_channels == true ]] && [[ $skip_output_channels == false ]]
          then 
            filename=$filename_prefix$input_lower-$saliency_method_lower-$norm_lower\_norm-$scaling\_skip_input_channels\_caffe_iter$i.npy
          elif [[ $skip_input_channels == false ]] && [[ $skip_output_channels == true ]]
          then 
            filename=$filename_prefix$input_lower-$saliency_method_lower-$norm_lower\_norm-$scaling\_skip_output_channels\_caffe_iter$i.npy
          fi
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
              \--retrain $retrain \
              \--input-channels $input_channels \
              \--output-channels $output_channels \
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
      done
    done
  done
done
