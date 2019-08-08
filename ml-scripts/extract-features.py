import pandas as pd
import os
import numpy as np
from tfidf import *
import sys
#two inputs from the user seg-datasets and output folder paths
data_path="/Users/mohsen-tum/Projects/stellargraph/smwyg"
output_path=data_path
if(len(sys.argv)>1):
  data_path = sys.argv[1]
  output_path = sys.argv[2]
data_dir = os.path.expanduser(data_path)
feature_names = ["w_{}".format(ii) for ii in range(64)]
column_names =  ['uid']+feature_names  + ["subject"]
node_data = pd.read_csv(os.path.join(data_dir, "blocks.csv"),lineterminator='\r', sep=';', header=None,index_col=False,dtype={'uid':object}, names=column_names)
node_data.set_index('uid', inplace=True)
edgelist = pd.read_csv(os.path.join(data_dir, "relations.csv"),index_col=False, sep=';', header=None, names=["source", "target","label"],dtype={'source': object, 'target':object})

print('Writing to edges.h5...')
edgelist.to_hdf(os.path.join(output_path,'edges.h5'),key='edges',mode='w')
#TFIDF begin of stuff

tfidf_count = 200 
print("TFIDF on basic block instructions...")
tfidfdf = extractTFIDFMemoryFriendly(node_data['w_63'], maxfeatures=tfidf_count)
print("Merging TFIDF vector with the node_data attributes...")
tfidfdf = tfidfdf.rename(columns=lambda x: 'w_'+str(int(x)+64))
tfidf_cols = tfidfdf.columns.values.tolist()
tfidfdf['uid'] = node_data.index
tfidfdf.set_index('uid',inplace=True)
feature_names = feature_names + tfidf_cols
del node_data["w_63"]
feature_names.remove('w_63')
node_data =pd.concat([node_data,tfidfdf], axis=1)

#TFIDF end of stuff
print('Writing to nodes.h5...')
node_data.to_hdf(os.path.join(output_path,'nodes.h5'),key='node_data',mode='w')
print('Done, Bye!')
