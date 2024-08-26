import _pickle as pkl
from sklearn.linear_model import LogisticRegression
import sys
import numpy as np
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.Interpretability.deepaid.interpreters.graph_embed_link import GraphEmbedLink
edge_list_file = 'tests/deepaid/graph_link_embed_apt/data/train_ben.edgelist'
node_name_map = pkl.load(open('tests/deepaid/graph_link_embed_apt/data/node_map.pkl','rb'))
embeddings = np.load('tests/deepaid/graph_link_embed_apt/data/embeddings.npy')
model = pkl.load(open('tests/deepaid/graph_link_embed_apt/save/clf.pkl','rb'))
interpretper = GraphEmbedLink(model,input_size=64,edge_list_file=edge_list_file,node_name_map=node_name_map,embeddings=embeddings)
anomaly_embedding = np.load('tests/deepaid/graph_link_embed_apt/data/anomaly_embedding.npy')[83]
anomaly_node_pair = np.load('tests/deepaid/graph_link_embed_apt/data/anomaly_node_pair.npy')[83]
print('anomaly_node_pair',anomaly_node_pair[0],anomaly_node_pair[1])
item = interpretper(anomaly_embedding,anomaly_node_pair)
print('reference_node_pair',item[-2],item[-1])