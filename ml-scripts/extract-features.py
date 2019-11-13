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
print("data_path:{} output_path:{}".format(data_path,output_path))
data_dir = os.path.expanduser(data_path)
edgelist = pd.DataFrame()
#chunk_size = 75000000 
chunk_size = 975000000 
read_size = 0
#chunk_temp = []
appendf = False 
#print('Concating chunks')
#edgelist = pd.concat(chunk_temp)
#del chunk_temp
#print('Writing to edges.h5...')
#edgelist.to_hdf(os.path.join(output_path,'edges.h5'),key='edges',mode='w')
#del edgelist
feature_names = ["w_{}".format(ii) for ii in range(64)]
column_names =  ['uid']+feature_names+["program"]+["subject"]
node_data = pd.read_csv(os.path.join(data_dir, "blocks.csv"),lineterminator='\r', sep=';', header=None,index_col=False,dtype={'uid':object}, names=column_names)
node_data.set_index('uid', inplace=True)
print('Finished reading nodes!')


print(node_data['program'])
#exit(1)
store = pd.HDFStore(os.path.join(output_path,'edges.h5'))
for chunk in pd.read_csv(os.path.join(data_dir, "relations.csv"),index_col=False, sep=';', header=None, names=["source", "target","label"],dtype={'source': object, 'target':object},chunksize=chunk_size):
  #chunk_temp.append(chunk)
#  if appendf:
#    store.append('edges',chunk,format='t',data_columns=True)
#  else:
    #chunk.to_hdf(os.path.join(output_path,'edges.h5'),key='edges',mode='w', format='t')
  store.put('edges',chunk,format='t',append=True,data_columns=True)
  appendf=True
  read_size = read_size + chunk_size
  print('Read node elements:',read_size)


#TFIDF begin of stuff

tfidf_count = 200 
print("TFIDF on basic block instructions...")
tfidfdf = extractTFIDFMemoryFriendly(node_data['w_63'], maxfeatures=tfidf_count,data_path=output_path)
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
