import numpy as np
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,get_raw_datasets
from csmt.figure.visualml.plot_importance import plot_feature_importance_all
from sklearn.feature_selection import SelectKBest,VarianceThreshold,mutual_info_classif
import matplotlib.pyplot as plt
from csmt.figure.visualml.plot_importance import plot_vec
from minepy import MINE

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    evasion_algorithm=options.evasion_algorithm
    
    X,y,mask=get_raw_datasets(options)
    
    print(X)
    # X_train,y_train,X_val,y_val,X_test,y_test,constraints=get_datasets(datasets_name)

    
    # step1 方差
    # calculate variances of x
    # selector = VarianceThreshold()
    # se = selector.fit_transform(X)
    # selector.variances_
    
    # drop the features with variance < 1e-3
    # de_v = np.where(selector.variances_<1e-3)[0]
    # x_de = np.delete(X,de_v,1)
    
    # step2 互信息
    # MI=mutual_info_classif(X,y)
    # print(MI)

    def mic(x,y):
        m = MINE()
        m.compute_score(x,y)
        return m.mic()
    
    # step3 MIC
    mic_arr=[]
    for i in range(X.shape[1]):
        mic_value=mic(X.iloc[:,i],y)
        mic_arr.append(mic_value)
        
    plot_vec(mic_arr,'MIC_importance')

    
    # pick the first n main features
    # n = 60
    # # function: calculate MIC
    # def mic(x,y):
    #     m = MINE()
    #     m.compute_score(x,y)
    #     return (m.mic(),0.5)
    # X_selection = SelectKBest(lambda X, Y: tuple(map(tuple,np.array(list(map(lambda x:mic(x, Y), 
    #                 X.T))).T)),k=n).fit_transform(X,y)
    # print(X_selection)
    
    
