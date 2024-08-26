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
from csmt.get_model_data import get_datasets,parse_arguments,get_raw_datasets


seed=20
np.random.seed(seed)
random.seed(seed)

arguments = sys.argv[1:]
options = parse_arguments(arguments)
X,y,mask=get_raw_datasets(options)
mm=MinMaxScaler()
X=mm.fit_transform(X)
y=y

alibox = ToolBox(X=X, y=y, query_type='AllLabels')

# Split data
alibox.split_AL(test_ratio=0.3, initial_label_rate=0.05, split_count=10)

# Use the default Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# model=RandomForestClassifier()

# The cost budget is 50 times querying
stopping_criterion = alibox.get_stopping_criterion('num_of_queries', 200)

# Use random strategy
Random_result = []
for round in range(2):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    RandomStrategy = alibox.get_query_strategy(strategy_name='QueryInstanceRandom')
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round) 

    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind = RandomStrategy.select(label_ind, unlab_ind, model=None, batch_size=1)
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)
        # Update model and calc performance according to the model you are using
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                  y_pred=pred,
                                                  performance_metric='accuracy_score')

        # Save intermediate results to file
        st = alibox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    Random_result.append(copy.deepcopy(saver))

# # Use Uncertain strategy
Uncertain_result = []
for round in range(2):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    UncertainStrategy = alibox.get_query_strategy(strategy_name='QueryInstanceUncertainty')
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round)

    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind = UncertainStrategy.select(label_ind, unlab_ind, model=None, batch_size=1)
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)

        # Update model and calc performance according to the model you are using
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                  y_pred=pred,
                                                  performance_metric='accuracy_score')

        # Save intermediate results to file
        st = alibox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    Uncertain_result.append(copy.deepcopy(saver))

# # Use CoresetGreedy strategy
CoresetGreedy_result = []
for round in range(2):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    CoresetGreedyStrategy = alibox.get_query_strategy(strategy_name='QueryInstanceCoresetGreedy',train_idx=train_idx)
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round)

    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind = CoresetGreedyStrategy.select(label_ind, unlab_ind, model=None, batch_size=1)
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)

        # Update model and calc performance according to the model you are using
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                  y_pred=pred,
                                                  performance_metric='accuracy_score')

        # Save intermediate results to file
        st = alibox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    CoresetGreedy_result.append(copy.deepcopy(saver))

# Use QueryInstanceDensityWeighted strategy
Density_result = []
for round in range(2):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    DensityStrategy = alibox.get_query_strategy(strategy_name='QueryInstanceDensityWeighted',uncertainty_meansure='entropy')
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round)

    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind = DensityStrategy.select(label_ind, unlab_ind, model=model, batch_size=1)
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)

        # Update model and calc performance according to the model you are using
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                  y_pred=pred,
                                                  performance_metric='accuracy_score')

        # Save intermediate results to file
        st = alibox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    Density_result.append(copy.deepcopy(saver))


from csmt.active_learning.alipy.experiment import ExperimentAnalyser
# get the query results
anal1 = ExperimentAnalyser(x_axis='num_of_queries')
anal1.add_method('RaS', Random_result)
anal1.add_method('UnS',Uncertain_result)
anal1.add_method('ReS', CoresetGreedy_result)
anal1.add_method('Density', Density_result)
# set plot parameters
anal1.plot_learning_curves(title='Learning curves', std_area=True,show=False)
plt.title('',fontproperties='Times New Roman',fontsize=1)
plt.yticks(fontproperties='Times New Roman',fontsize=12)
plt.xticks(fontproperties='Times New Roman',fontsize=12)
plt.xlabel('Number of queries',fontproperties='Times New Roman',fontsize=14)
plt.ylabel('Accuracy',fontproperties='Times New Roman',fontsize=14)
plt.legend(loc=4,prop='Times New Roman')
plt.show()
