import sys
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
# matplotlib.style.use('seaborn-dark')
import numpy as np
from matplotlib import rcParams
config={
    "font.family":'Times New Roman'
}
rcParams.update(config)


def Fidelity_Conciseness():

    sheet_name='port-rf'
    df=pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/AL-SHAP/AL-SHAP.xlsx',sheet_name=sheet_name)
    print(df)
    
    x=[0,0.2,0.4,0.6,0.8]
    x_label = [0.2,0.4,0.6,0.8,1.0]
    color=['#d8d9dc','#bcd6e6','#5e98c2','#4272b1','#285085']
    # color=['#f3cab3','#df9478','#c15a4c','#9e282f','#5e0e20']
    hatch=['','','','','']
    plt.figure(figsize=(7,5), dpi= 80)
    plt.bar(x=[i + 0.2 for i in x], height=(0.55-df['fidelity'])/0.55*100, hatch=hatch[0],color=color[3], width=.14, label='fidelity', alpha=1)
    plt.gca().set_xticks(x_label)
    plt.yticks(fontsize=30,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticklabels(df['Method'], rotation=0, horizontalalignment= 'center',fontdict={'size':26})
    plt.gca().set_ylabel('Fidelity(%)',fontdict={'size':25},labelpad=-15)
    plt.show()
    
def Efficiency_Evaluation():
    
    sheet_name='port-mlp'
    df=pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/AL-SHAP/AL-SHAP.xlsx',sheet_name=sheet_name)
    print(df)
    
    x=[0,0.2,0.4,0.6,0.8]
    x_label = [0.2,0.4,0.6,0.8,1.0]
    color=['#d8d9dc','#bcd6e6','#5e98c2','#4272b1','#285085']
    # color=['#f3cab3','#df9478','#c15a4c','#9e282f','#5e0e20']
    hatch=['','','','','']
    plt.figure(figsize=(7,5), dpi= 80)
    plt.bar(x=[i + 0.2 for i in x], height=df['time'], hatch=hatch[0],color=color[2], width=.14, label='time', alpha=1)
    plt.gca().set_xticks(x_label)
    plt.yticks(fontsize=30,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticklabels(df['Method'], rotation=0, horizontalalignment= 'center',fontdict={'size':28})
    plt.gca().set_ylabel('Running-Time(s)',fontdict={'size':25},labelpad=-2)
    plt.show()
    
# Fidelity_Conciseness()
Efficiency_Evaluation()