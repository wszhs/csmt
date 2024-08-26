import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
import pandas as pd

def get_nash():
    y_arr=[]
    x_arr=[]
    COUNT=5
    sheet_name='ctu13-a'
    df=pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/NashAE/4-收敛性.xlsx',sheet_name=sheet_name)
    y=np.array([])
    x=np.array([])
    name_arr=['NashRL','NashAE']
    for j in range(len(name_arr)):
        for i in range(COUNT):
            y_i=df[name_arr[j]+str(i+1)].values*100
            x_i=range(len(y_i))
            y=np.concatenate((y,y_i))
            x=np.concatenate((x,x_i))
        y_arr.append(y)
        x_arr.append(x)
    return y_arr,x_arr

y_arr,x_arr=get_nash()

name_arr=['NashRL','NashAE']
for i in range(len(y_arr)):
    sns.lineplot(x=x_arr[i],y=y_arr[i],label=name_arr[i])

plt.xlabel("Iteration")
plt.xticks(fontsize=23)
plt.ylabel("Detection Rate (%)",fontsize=20)
plt.yticks(fontsize=20,rotation=0, horizontalalignment= 'right',)
plt.show()