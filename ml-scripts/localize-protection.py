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

####################################################
#input the path to h5 files node_data.h5 and edges.h5
output_results = {}
data_path = "/Users/mohsen-tum/Projects/stellargraph/smwyg"
if len(sys.argv)>0:
  data_path = sys.argv[1]

data_dir = os.path.expanduser(data_path)
tfidf_count = 200 
feature_names = ["w_{}".format(ii) for ii in range(64+tfidf_count)]
feature_names.remove('w_63')
column_names =  ['uid']+feature_names  + ["subject"]
edgelist = pd.read_hdf(os.path.join(data_dir, "edges.h5"),key='edges')
node_data=pd.read_hdf(os.path.join(data_dir, "nodes.h5"),key='node_data')
#print(node_data)
#print(edgelist)
#node_data.set_index('uid', inplace=True)
print("feeding edges to the graph...")
Gnx = nx.from_pandas_edgelist(edgelist, edge_attr="label")
nx.set_node_attributes(Gnx, "block", "label")
set(node_data["subject"])
print('spliting train and test data...')
train_data, test_data = model_selection.train_test_split(node_data, train_size=0.8, test_size=None, stratify=node_data['subject'])
from collections import Counter
#TODO: save size of dataset, train_data, and test data
#save the count of each subject in the blocks
print(len(train_data), len(test_data))
subject_groups = Counter(train_data['subject'])
print(subject_groups)
output_results.train_size = len(train_data)
output_results.test_size = len(test_data)
output_results.subject_groups = subject_groups
node_features = node_data[feature_names]
print (node_features)
G = sg.StellarGraph(Gnx, node_features=node_features)
#TODO: save graph info
print(G.info())
print("writing graph.dot")
#write_dot(Gnx,"graph.dot")
output_results.graph_info=G.info()
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
    layer_sizes=[32, 32],
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
    epochs=35,
    validation_data=test_gen,
    verbose=2,
    shuffle=True,
    class_weight=class_weights,
)
import matplotlib.pyplot as plt
#%matplotlib inline

def plot_history(history):
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
plot_history(history)
# save test metrics
test_metrics = model.evaluate_generator(test_gen)
print("\nTest Set Metrics:")
output_results.test_metrics = []
for name, val in zip(model.metrics_names, test_metrics):
    output_results.test_metrics.append({'name':name, 'val:':val})
    print("\t{}: {:0.4f}".format(name, val))
all_nodes = node_data.index
all_mapper = generator.flow(all_nodes)
all_predictions = model.predict_generator(all_mapper)
node_predictions = target_encoding.inverse_transform(all_predictions)
results = pd.DataFrame(node_predictions, index=all_nodes).idxmax(axis=1)
df = pd.DataFrame({"Predicted": results, "True": node_data['subject']})
#print(df.head(100))


clean_result_labels = df["Predicted"].map(lambda x: x.replace('subject=',''))
# save predicted labels
pred_labels = np.unique(clean_result_labels.values)
# save predictions per label
precision, recall, f1, _ = skmetrics.precision_recall_fscore_support(df['True'].values,clean_result_labels.values,labels=pred_labels)
output_results.classifier = zip(pred_labels, precision,recall,f1)
print(pred_labels)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(f1))
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


embedding_model = Model(inputs=x_inp, outputs=x_out)
emb = embedding_model.predict_generator(all_mapper)
emb.shape

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
X = emb
y = np.argmax(target_encoding.transform(node_data[["subject"]].to_dict('records')), axis=1)
if X.shape[1] > 2:
    transform = PCA #TSNE  

    trans = transform(n_components=2)
    emb_transformed = pd.DataFrame(trans.fit_transform(X), index=node_data.index)
    emb_transformed['label'] = y
else:
    emb_transformed = pd.DataFrame(X, index=node_data.index)
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




#save results
import json
with open(os.path.join(data_path,'result.json'), 'w') as fp:
    json.dump(output_results, fp)
print('Finished {}, BYE!'.format(os.path.join(data_path,'result.json')))
