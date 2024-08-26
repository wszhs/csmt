import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import copy
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np
from csmt.active_learning.alipy import ToolBox
from csmt.classifiers.scores import get_class_scores
from csmt.get_model_data import get_datasets,parse_arguments,get_raw_datasets
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict,models_load
from csmt.active_learning.alipy.experiment import ExperimentAnalyser
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.manifold import TSNE

seed=20
np.random.seed(seed)
random.seed(seed)

arguments = sys.argv[1:]
options = parse_arguments(arguments)
X,y,mask=get_raw_datasets(options)
mm=MinMaxScaler()
X=mm.fit_transform(X)
y=y.values

alibox = ToolBox(X=X, y=y, query_type='AllLabels')

# Split data
alibox.split_AL(test_ratio=0.3, initial_label_rate=0.01, split_count=10)

arguments = sys.argv[1:]
options = parse_arguments(arguments)
datasets_name=options.datasets
orig_models_name=options.algorithms

stopping_criterion = alibox.get_stopping_criterion('num_of_queries', 200)

# strategy_name_arr=['QueryInstanceUncertainty']
strategy_name_arr=['QueryInstanceRandom','QueryInstanceUncertainty','QueryInstanceCoresetGreedy']


def strategy_query(strategy_name):
    _result = []
    for round in range(2):
        # Get the data split of one fold experiment
        train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
        if strategy_name in ('QueryInstanceGraphDensity','QueryInstanceCoresetGreedy'):
            _Strategy = alibox.get_query_strategy(strategy_name=strategy_name,train_idx=train_idx)
        else:
            _Strategy = alibox.get_query_strategy(strategy_name=strategy_name)
        # Get intermediate results saver for one fold experiment
        saver = alibox.get_stateio(round)

        while not stopping_criterion.is_stop():
            # Select a subset of Uind according to the query strategy
            # Passing model=None to use the default model for evaluating the committees' disagreement
            select_ind = _Strategy.select(label_ind, unlab_ind, model=None, batch_size=1)
            label_ind.update(select_ind)
            unlab_ind.difference_update(select_ind)

            # csmt 接口
            trained_models=models_train(datasets_name,orig_models_name,X[label_ind.index, :], y[label_ind.index],X[label_ind.index, :], y[label_ind.index])
            y_test,y_pred=models_predict(trained_models,X[test_idx, :], y[test_idx])
            for i in range(0,len(orig_models_name)):
                y_pred=np.argmax(y_pred[i], axis=1)
            accuracy = metrics.accuracy_score(y_test, y_pred)

            # Save intermediate results to file
            st = alibox.State(select_index=select_ind, performance=accuracy)
            saver.add_state(st)
            saver.save()

            # Passing the current progress to stopping criterion object
            stopping_criterion.update_information(saver)
        # Reset the progress in stopping criterion object
        stopping_criterion.reset()
        _result.append(copy.deepcopy(saver))
    return _result
    
# uncertain+Coreset
uncertain_core_result = []
for round in range(2):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    UncertainStrategy = alibox.get_query_strategy(strategy_name='QueryInstanceUncertainty')
    CoresetGreedyStrategy = alibox.get_query_strategy(strategy_name='QueryInstanceCoresetGreedy',train_idx=train_idx)
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round)

    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_uncertain_ind = UncertainStrategy.select(label_ind, unlab_ind, model=None, batch_size=1)
        label_ind.update(select_uncertain_ind)
        unlab_ind.difference_update(select_uncertain_ind)
        
        select_core_ind = CoresetGreedyStrategy.select(label_ind, unlab_ind, model=None, batch_size=1)
        label_ind.update(select_core_ind)
        unlab_ind.difference_update(select_core_ind)

        # csmt 接口
        trained_models=models_train(datasets_name,orig_models_name,X[label_ind.index, :], y[label_ind.index],X[label_ind.index, :], y[label_ind.index])
        y_test,y_pred=models_predict(trained_models,X[test_idx, :], y[test_idx])
        for i in range(0,len(orig_models_name)):
            y_pred=np.argmax(y_pred[i], axis=1)
        result=get_class_scores(y_test, y_pred)
        accuracy=result[0]

        # Save intermediate results to file
        st_uncertain = alibox.State(select_index=select_uncertain_ind, performance=accuracy)
        saver.add_state(st_uncertain)
        
        st_core = alibox.State(select_index=select_core_ind, performance=accuracy)
        saver.add_state(st_core)
        
        saver.save()

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    uncertain_core_result.append(copy.deepcopy(saver))

# get the query results
anal1 = ExperimentAnalyser(x_axis='num_of_queries')
for _strategy in strategy_name_arr:
    result=strategy_query(_strategy)
    anal1.add_method(_strategy,result)
anal1.add_method('uncertain_coreset',uncertain_core_result)

# set plot parameters
anal1.plot_learning_curves(title='Learning curves', std_area=True,show=False)
plt.title('Learning curves',fontproperties='Times New Roman',fontsize=14)
plt.yticks(fontproperties='Times New Roman',fontsize=12)
plt.xticks(fontproperties='Times New Roman',fontsize=12)
plt.xlabel('Number of queries',fontproperties='Times New Roman',fontsize=14)
plt.ylabel('Performance',fontproperties='Times New Roman',fontsize=14)
plt.legend(loc=4,prop='Times New Roman')
plt.show()