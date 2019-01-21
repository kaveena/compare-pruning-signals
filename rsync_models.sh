all_src="starbuck sweetness toothbrush "
for src in $all_src
do
  rsync \-uvrtP kaveena@$src:~/compare-pruning-signals/caffe-pruned-models/* ./caffe-pruned-models/
done
