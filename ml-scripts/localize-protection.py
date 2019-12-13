import networkx as nx
#from networkx.drawing.nx_pydot import write_dot
import pygraphviz
from networkx.drawing.nx_agraph import write_dot
import pandas as pd
import os
import numpy as np
import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from stellargraph.core.graph import StellarGraph
from stellargraph.layer.hinsage import HinSAGE
from stellargraph.mapper.node_mappers import HinSAGENodeGenerator
import sys
from keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn import metrics as skmetrics
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
####################################################
#input the path to h5 files node_data.h5 and edges.h5

EPOCH=7
N_KFOLDS=10

def select_features(node_data):
  tfidf_count = 200
  svf_count=64#64
  feature_names = ["w_{}".format(ii) for ii in range(svf_count+tfidf_count)]
  feature_names.remove('w_63')
  node_data=node_data.filter(feature_names+['subject']) #program
  print(node_data) 
  return node_data, feature_names

def read_data(data_dir):
  print('Reading nodes')
  node_data=pd.read_hdf(os.path.join(data_dir, "nodes.h5"),key='node_data')
  print('Reading edges')
  chunks = []
  read_size = 0
  #edges= pd.read_hdf(os.path.join(data_dir, "edges.h5"),key='edges')
  #Gnx = nx.from_pandas_edgelist(edges, edge_attr="label")
  chunk_size=10000000
  #chunk_size=100000000
  Gnx = nx.Graph()
  for chunk in pd.read_hdf(os.path.join(data_dir, "edges.h5"),key='edges',chunksize=chunk_size):
    chunk=chunk.loc[chunk['label'].isin(['cfg'])]#['svgc','svgd','scc','cfg']
    g = nx.from_pandas_edgelist(chunk, edge_attr="label")
    chunks.append(g)
    read_size = read_size + chunk_size
    print('Read edge chunk ', read_size)
  for chunk in chunks:
    Gnx.add_edges_from(chunk.edges(data=True))
  nx.set_node_attributes(Gnx, "block", "label")
  return node_data, Gnx

def average_classifiers(classifiers):
  labels=['sc_guard','cfi_verify','oh_verify','none']
  metric='fscore'
  tmp={}
  for lbl in labels:
    tmp[lbl]=[]
  for array in classifiers:
    for scores in array:
      if np.isnan(scores[metric]):
          continue
      tmp[scores['label']].append(scores[metric])
  output={}
  for label in labels:
    output[label]=np.mean(tmp[label])
  print(output)
  return output
 
def main():
  data_path = "/Users/mohsen-tum/Projects/stellargraph/smwyg"
  if len(sys.argv)>1:
    print("datapath:{}".format(sys.argv[1]))
    data_path = sys.argv[1]

  data_dir = os.path.expanduser(data_path)
  node_data, Gnx = read_data(data_dir)
  node_data, feature_names = select_features(node_data)
  
  #load all paths
  if len(sys.argv)>2:
    print(sys.argv[2])
    for directory in sys.argv[2].split(';'):
      print("appending {}".format(directory))
      tmpnode = pd.read_hdf(os.path.join(directory,'nodes.h5'),key='node_data')
      node_data = pd.concat([node_data,tmpnode])
      del tmpnode
      tmpedge = pd.read_hdf(os.path.join(directory,'edges.h5'),key='edges')
      edgelist = pd.concat([edgelist,tmpedge])
      del tmpedge

  #node_data.set_index('uid', inplace=True)
  column_names =  ['uid']+feature_names  +['program']+ ["subject"]
  print("feeding edges to the graph...")

  #Gnx = nx.from_pandas_edgelist(edgelist, edge_attr="label")
  #nx.set_node_attributes(Gnx, "block", "label")

  set(node_data["subject"])
  print('spliting train and test data...')
  
  all_features = node_data[feature_names]
  #train_data, test_data = model_selection.train_test_split(node_data, train_size=0.8, test_size=0.2, stratify=node_data['subject'].values.ravel())
  skf = StratifiedKFold(n_splits=N_KFOLDS, random_state=12321, shuffle=False)
  classifier_results=[]
  output_results = {}
  i=1
  for train_index, test_index in skf.split(node_data, node_data['subject'].values.ravel()):
    print('Cross fold validation iteration {}'.format(i))
    i+=1
    train_data=node_data.iloc[train_index]
    test_data=node_data.iloc[test_index]
    generator, model,x_inp,x_out, history, target_encoding, out_result = train_model(Gnx,train_data,test_data,all_features)
    classifier_results.append(out_result['classifier'])
    plot_history(history,data_path)
    output_results=out_result
  
  
  output_results['Kfold_results']=classifier_results
  output_results['classifier']=average_classifiers(classifier_results)
  all_nodes = node_data.index
  all_mapper = generator.flow(all_nodes)

  
  #save_model(data_dir, model)
  #save results
  import json
  with open(os.path.join(data_path,'result.json'), 'w') as fp:
    json.dump(output_results, fp)
  print('Finished {} results are here:'.format(os.path.join(data_path,'result.json')))
  subjects = node_data[["subject"]]
  embed_nodes(target_encoding, all_mapper,subjects,node_data.index,x_inp,x_out,data_path)
  print('Finished creating Node Embeddings! Bye!')

#we pass all features (both train and test) as the graph keeps both of them, the test features are used in model testing
def train_model(Gnx,train_data, test_data, all_features):
  output_results={}
  from collections import Counter
  #TODO: save size of dataset, train_data, and test data
  #save the count of each subject in the blocks
  print(len(train_data), len(test_data))
  subject_groups_train = Counter(train_data['subject'])
  subject_groups_test = Counter(test_data['subject'])
  output_results['train_size'] = len(train_data)
  output_results['test_size'] = len(test_data)
  output_results['subject_groups_train'] = subject_groups_train
  output_results['subject_groups_test'] = subject_groups_test

  #node_features = train_data[feature_names]
  #print (feature_names)
  G = sg.StellarGraph(Gnx, node_features=all_features)
  #TODO: save graph info
  print(G.info())
  print("writing graph.dot")
  #write_dot(Gnx,"graph.dot")
  output_results['graph_info']=G.info()
  print("building the graph generator...")

  batch_size = 50; num_samples = [10, 5]
  generator = GraphSAGENodeGenerator(G, batch_size, num_samples)
  #generator = HinSAGENodeGenerator(G, batch_size, num_samples)

  target_encoding = feature_extraction.DictVectorizer(sparse=False)
  train_targets = target_encoding.fit_transform(train_data[["subject"]].to_dict('records'))
  class_weights = class_weight.compute_class_weight('balanced',np.unique(train_targets),train_targets[:,0])
  test_targets = target_encoding.transform(test_data[["subject"]].to_dict('records'))
  train_gen = generator.flow(train_data.index, train_targets, shuffle=True)
  graphsage_model = GraphSAGE(
  #graphsage_model = HinSAGE(
      #layer_sizes=[32, 32],
      layer_sizes=[80, 80],
      generator=train_gen,
      bias=True,
      dropout=0.5,
  )
  print("building model...")
  x_inp, x_out = graphsage_model.build(flatten_output=True)
  prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

  model = Model(inputs=x_inp, outputs=prediction)
  print("compiling model...")
  model.compile(
      optimizer=optimizers.Adam(lr=0.005),
      loss=losses.categorical_crossentropy,
      metrics=["acc",metrics.categorical_accuracy],
  )
  print("testing the model...")
  test_gen = generator.flow(test_data.index, test_targets)
  history = model.fit_generator(
      train_gen,
      epochs=EPOCH,
      validation_data=test_gen,
      verbose=2,
      shuffle=True,
      class_weight=class_weights,
  )
  # save test metrics
  test_metrics = model.evaluate_generator(test_gen)
  print("\nTest Set Metrics:")
  output_results['test_metrics'] = []
  for name, val in zip(model.metrics_names, test_metrics):
    output_results['test_metrics'].append({'name':name, 'val:':val})
    print("\t{}: {:0.4f}".format(name, val))
  
  test_nodes = test_data.index
  test_mapper = generator.flow(test_nodes)
  test_predictions = model.predict_generator(test_mapper)
  node_predictions = target_encoding.inverse_transform(test_predictions)
  results = pd.DataFrame(node_predictions, index=test_nodes).idxmax(axis=1)
  df = pd.DataFrame({"Predicted": results, "True": test_data['subject']}) #, "program":test_data['program']})
  clean_result_labels = df["Predicted"].map(lambda x: x.replace('subject=',''))
  # save predicted labels
  pred_labels = np.unique(clean_result_labels.values)
  #pred_program = np.unique(df['program'].values)
  # save predictions per label
  precision, recall, f1, _ = skmetrics.precision_recall_fscore_support(df['True'].values,clean_result_labels.values,average=None,labels=pred_labels)
  output_results['classifier'] =[]
  for lbl, prec, rec, fm in zip(pred_labels, precision,recall,f1):
    output_results['classifier'].append({'label':lbl, 'precision':prec, 'recall':rec, 'fscore':fm})
  print(output_results['classifier'])
  print(pred_labels)
  print('precision: {}'.format(precision))
  print('recall: {}'.format(recall))
  print('fscore: {}'.format(f1))
  
  return generator,model,x_inp,x_out, history, target_encoding, output_results

def plot_history(history,data_path):
    metrics = sorted(history.history.keys())
    metrics = metrics[:len(metrics)//2]
    for m in metrics:
        # summarize history for metric m
        plt.plot(history.history[m])
        plt.plot(history.history['val_' + m])
        plt.title(m)
        plt.ylabel(m)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        #plt.show()
        plt.savefig(os.path.join(data_path,str(m)+".pdf"))


def save_model(data_path,model):
  model.save(os.path.join(data_path,"model.h5"))

def write_fscore_to_csv(program_classifiers):
  import csv
  classes = {}
  fscores =[]
  for classifier in program_classifiers:
    program = classifier['program']
    for score in classifier['classes']:
      if score['label'] not in classes:
        classes[score['label']]=[]
      classes[score['label']].append((program,score['fscore'],score['samples']))
          
  for label, scores in classes.items():
    with open(os.path.join(data_path,'programs_fscore_{}.csv'.format(label)), mode='w') as fscore_file:
      fscore_writer = csv.writer(fscore_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
      for program, score, samples in scores:
        fscore_writer.writerow([program,score,samples])
      array_fscore=[x[1] for x in scores]
      array_samples= [x[2] for x in scores]
      fscore_writer.writerow(['mean',np.mean(array_fscore), np.mean(array_samples)])
      fscore_writer.writerow(['std',np.std(array_fscore), np.std(array_samples)])
      fscore_writer.writerow(['median',np.median(array_fscore), np.median(array_samples)])

def calc_fscore_per_program(df, programs):
  program_classifiers = []
  for program in programs:
    df_program = df[(df.program == program)]
    print(len(df),len(df_program))
    predictions = df_program['Predicted'].map(lambda x: x.replace('subject=',''))
    ground_truth= df_program['True']
    #print(ground_truth)
    pred_labels = np.unique(predictions.values)
    precision, recall, f1, _ = skmetrics.precision_recall_fscore_support(ground_truth.values,predictions.values,labels=pred_labels)
    classes =[]
    counts = Counter(df_program['True'])
    print("Check this:",program,counts)
    for lbl, prec, rec, fm in zip(pred_labels, precision,recall,f1):
        classes.append({'label':lbl, 'precision':prec, 'recall':rec, 'fscore':fm, 'samples':counts[lbl]})
    program_classifiers.append({'program':program,'classes':classes})
  return program_classifiers 


#program_scores = calc_fscore_per_program(df, pred_program)
#write_fscore_to_csv(program_scores)
#for nid, pred, true in zip(df.index, df["Predicted"], df["True"]):
#    Gnx.node[nid]["subject"] = true
#    Gnx.node[nid]["PREDICTED_subject"] = pred.split("=")[-1]

#for nid in train_data.index:
#    Gnx.node[nid]["isTrain"] = True
    
#for nid in test_data.index:
#    Gnx.node[nid]["isTrain"] = False
#for nid in Gnx.nodes():
#    Gnx.node[nid]["isCorrect"] = Gnx.node[nid]["subject"] == Gnx.node[nid]["PREDICTED_subject"]
#pred_fname = "pred_n={}.graphml".format(num_samples)
#nx.write_graphml(Gnx, os.path.join(data_dir,pred_fname))



#Node embedding

def embed_nodes(target_encoding,all_mapper,subjects,all_index, x_inp, x_out, data_path):
  embedding_model = Model(inputs=x_inp, outputs=x_out)
  emb = embedding_model.predict_generator(all_mapper)
  emb.shape

  from sklearn.decomposition import PCA
  from sklearn.manifold import TSNE
  import pandas as pd
  import numpy as np
  X = emb
  y = np.argmax(target_encoding.transform(subjects.to_dict('records')), axis=1)
  if X.shape[1] > 2:
    transform = PCA#TSNE  

    trans = transform(n_components=2)
    emb_transformed = pd.DataFrame(trans.fit_transform(X), index=all_index)
    emb_transformed['label'] = y
  else:
    emb_transformed = pd.DataFrame(X, index=all_index)
    emb_transformed = emb_transformed.rename(columns = {'0':0, '1':1})
    emb_transformed['label'] = y
  alpha = 0.7

  fig, ax = plt.subplots(figsize=(7,7))
  ax.scatter(emb_transformed[0], emb_transformed[1], c=emb_transformed['label'].astype("category"), 
            cmap="jet", alpha=alpha)
  ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
  #TODO. fix the header of the plot to reflect the dataset name
  plt.title('{} visualization of embeddings'.format(transform.__name__))
  # save the embeddings plot 
  #plt.show()
  plt.savefig(os.path.join(data_path,transform.__name__+'.pdf'))

if __name__== "__main__":
  main()


