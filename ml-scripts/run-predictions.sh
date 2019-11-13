
#get path to dataset of h5 files
dataset=$1
for ds in $dataset*; do
  if [ -d "$ds" ]; then
    blocks=$ds/node_data.h5
    rel=$ds/edges.h5
    echo "handling $ds $blocks" 
    if [ ! -f $ds/result.json ]; then
      python3 ml-scripts/localize-protection.py $ds
    else 
      echo "Already found results for $ds, remove $ds/result.json to recalculate"
    fi
  fi
done
