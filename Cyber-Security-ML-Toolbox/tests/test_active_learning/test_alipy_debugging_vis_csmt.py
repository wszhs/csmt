import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import copy
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from csmt.active_learning.alipy import ToolBox
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from csmt.get_model_data import get_datasets,parse_arguments,get_raw_datasets
import visual
import pandas as pd

seed=20
np.random.seed(seed)
random.seed(seed)

arguments = sys.argv[1:]
options = parse_arguments(arguments)
X_,y_,mask=get_raw_datasets(options)

X_raw=X_.values
y_raw=y_.values

# Define our PCA transformer and fit it onto our raw dataset.
de = PCA(n_components=2, random_state=seed)
# de = TSNE(n_components=2, random_state=seed)
transformed_data = de.fit_transform(X=X_raw)
# Isolate the data we'll need for plotting.
x_component, y_component = transformed_data[:, 0], transformed_data[:, 1]

# visual.plot_data(x_component, y_component,y_raw)


alibox = ToolBox(X=X_raw, y=y_raw, query_type='AllLabels')

# Split data
alibox.split_AL(test_ratio=0.3, initial_label_rate=0.002, split_count=2)
# print(alibox.label_idx[0])

# Use the default Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# model = LogisticRegression()
stopping_criterion = alibox.get_stopping_criterion('num_of_queries', 100)

# Use random strategy
Query_result = []
for round in range(1):
    
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    # visual.plot_is_label(transformed_data,label_ind)
    # QueryStrategy = alibox.get_query_strategy(strategy_name='QueryInstanceUncertainty')
    # QueryStrategy=alibox.get_query_strategy(strategy_name='QueryInstanceDensityWeighted',uncertainty_meansure='entropy')
    QueryStrategy = alibox.get_query_strategy(strategy_name='QueryInstanceRandom')
    # QueryStrategy = alibox.get_query_strategy(strategy_name='QueryInstanceCoresetGreedy',train_idx=train_idx)
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round) 

    count=0
    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind = QueryStrategy.select(label_ind, unlab_ind, model=None, batch_size=1)
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)
        # Update model and calc performance according to the model you are using
        model.fit(X=X_raw[label_ind.index, :], y=y_raw[label_ind.index])
        pred = model.predict(X_raw[test_idx, :])
        is_correct = (pred == y_raw[test_idx])
        accuracy = alibox.calc_performance_metric(y_true=y_raw[test_idx],
                                                  y_pred=pred,
                                                  performance_metric='accuracy_score')
        
        # if count%10==0:
        #     visual.plot_is_correct(x_component[test_idx],y_component[test_idx],is_correct,accuracy)

        # Save intermediate results to file
        st = alibox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
        count=count+1
    # Reset the progress in stopping criterion object
    
    visual.plot_is_label(transformed_data,label_ind)
    
    # series = pd.Series(label_ind) 
    # series.to_csv('csmt/datasets/data/CIC-IDS-2017/ids17_botnet_index.csv')

    stopping_criterion.reset()
    Query_result.append(copy.deepcopy(saver))


X=X_raw[label_ind.index, :]
y=y_raw[label_ind.index]
import seaborn as sns 
X_de= PCA(n_components=2).fit_transform(X)
df = pd.DataFrame(data = X_de, columns = ['comp0', 'comp1'])
df['label']=y
sns.jointplot(x='comp0', y='comp1', data=df,hue='label')
plt.show()


# from csmt.active_learning.alipy.experiment import ExperimentAnalyser
# # get the query results
# anal1 = ExperimentAnalyser(x_axis='num_of_queries')
# anal1.add_method('Qurry', Query_result)
# # set plot parameters
# anal1.plot_learning_curves(title='Learning curves', std_area=True,show=False)
# plt.title('',fontproperties='Times New Roman',fontsize=1)
# plt.yticks(fontproperties='Times New Roman',fontsize=12)
# plt.xticks(fontproperties='Times New Roman',fontsize=12)
# plt.xlabel('Number of queries',fontproperties='Times New Roman',fontsize=14)
# plt.ylabel('Accuracy',fontproperties='Times New Roman',fontsize=14)
# plt.legend(loc=4,prop='Times New Roman')
# plt.show()
