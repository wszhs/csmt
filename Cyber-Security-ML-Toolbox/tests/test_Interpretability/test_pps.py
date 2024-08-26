import numpy as np
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,get_raw_datasets
from csmt.figure.visualml.plot_importance import plot_feature_importance_all
from sklearn.feature_selection import SelectKBest,VarianceThreshold,mutual_info_classif
import matplotlib.pyplot as plt
import ppscore as pps

if __name__=='__main__':
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    datasets_name=options.datasets
    orig_models_name=options.algorithms
    evasion_algorithm=options.evasion_algorithm
    X,y,mask=get_raw_datasets(options)
    df=X
    df['label']=y
    
    print(df)
    
    # f1=pps.score(df, "Flow Duration", "label")
    # # print(f1)
    
    # fall=pps.predictors(df, "label")
    # print(fall)
    
    # import seaborn as sns
    # predictors_df = pps.predictors(df, y="label")
    # sns.barplot(data=predictors_df, x="x", y="ppscore")
    # print(predictors_df)
    # plt.show()
    
    # import seaborn as sns
    # matrix_df = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
    # sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
    # plt.show()
    