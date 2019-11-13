
#get path to dataset of h5 files
dataset=$1
result_path="results"
for ds in $dataset*; do
  if [ -d "$ds" ]; then
    blocks=$ds/blocks.csv
    rel=$ds/relations.csv
    echo "handling $ds $blocks" 
    result_ds=$(basename $ds)
    mkdir -p $result_path/$result_ds
    if [ ! -f $result_path/$result_ds/nodes.h5 ]; then
      python3 ml-scripts/extract-features.py $ds $result_path/$result_ds
    else
      echo "Already found $result_path/nodes.h5, remove it if wanna recalculate"
    fi
  fi
done
