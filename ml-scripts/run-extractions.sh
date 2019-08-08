
#get path to dataset of h5 files
dataset=$1
result_path="results"
for ds in $dataset*; do
  if [ -d "$ds" ]; then
    blocks=$ds/blocks.csv
    rel=$ds/relations.csv
    echo "handling $ds $blocks" 
    mkdir -p $result_path/$ds
    python3 ml-scripts/extract-features.py $ds $result_path/$ds
  fi
done
