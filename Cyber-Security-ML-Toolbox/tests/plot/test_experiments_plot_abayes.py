
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

def plot_transfer():
    sheet_name='cptc2018'
    my_color=['#161d15','#53565c','#858b92','#d8d9dc','#bcd6e6','#5e98c2','#4272b1','#285085']
    # my_color=['#161d15','#53565c','#75809c','#f2e6dc','#dfbac8','#a784a7','#965454']
    df = pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/Robust_priority/不确定.xlsx',sheet_name=sheet_name)
    # print(df)
    plt.figure(figsize=(15,10), dpi= 80)
    y_label=df['budget'].values
    df=df.drop(['budget'], axis=1)
    h = sns.heatmap(df,vmin=15,vmax=55, xticklabels = df.columns, yticklabels = y_label, cmap='RdBu_r',
                        annot=True,fmt='.2f',annot_kws={'size':28},cbar=False)

    cb=h.figure.colorbar(h.collections[0]) #显示colorbar
    cb.ax.tick_params(labelsize=40) #设置colorbar刻度字体大小。

    plt.xticks(fontsize=38)
    plt.yticks(fontsize=40,rotation=0, horizontalalignment= 'right')
    
    plt.gca().set_ylabel('Actual attack budget',fontdict={'size':32})
    plt.gca().set_xlabel('Defender estimate of attack budget',fontdict={'size':32},labelpad=-2)
    
    plt.show()
    # plt.savefig('/Users/zhanghangsheng/Desktop/TDSC/ETA/images/experiment/'+sheet_name+'_transfer.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    
def plot_priority():
    sheet_name='cptc2018-2.2'
    df=pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/Robust_priority/Robust_priority.xlsx',sheet_name=sheet_name)
    x=[1,2,3]
    x_label = [1.5, 2.5, 3.5]
    
    color=['#f3cab3','#f8cab3','#df9478','#c15a4c','#9e282f','#5e0e20']
    hatch=['','','','','','']
    plt.figure(figsize=(15,7), dpi= 80)
    plt.bar(x=[i + 0.15 for i in x], height=df['Uniform'], hatch=hatch[0],color=color[0], width=.14, label='Uniform', alpha=0.6)
    plt.bar(x=[i + 0.3 for i in x], height=df['Greedy'], color=color[1],hatch=hatch[1], width=.14, label='Greedy', alpha=1)
    plt.bar(x=[i + 0.45 for i in x], height=df['GAIN'], color=color[2],hatch=hatch[2], width=.14, label='GAIN', alpha=1)
    plt.bar(x=[i + 0.6 for i in x], height=df['RIO'], color=color[3],hatch=hatch[3], width=.14, label='RIO', alpha=1)
    plt.bar(x=[i + 0.75 for i in x], height=df['ARL'], color=color[4],hatch=hatch[4],width=.14, label='ARL', alpha=1)
    plt.bar(x=[i + 0.9 for i in x], height=df['ABayes'], color=color[5],hatch=hatch[5],width=.14, label='ABayes', alpha=1)

    plt.yticks(fontsize=35,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df['Method'] , rotation=0, horizontalalignment= 'center',fontdict={'size':25})
    plt.legend(loc='upper center',ncol=6,bbox_to_anchor=(0.5,1.15),prop={'size':20},edgecolor='black')
    plt.gca().set_ylabel('Defense Loss',fontdict={'size':32})
    plt.gca().set_xlabel('Actual attack budget',fontdict={'size':32},labelpad=-6)
    # plt.gca().set_xlabel('',fontdict={'size':28}) 
    # plt.savefig('/Users/zhanghangsheng/Desktop/TDSC/ETA/images/experiment/'+'different_'+sheet_name+'.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    plt.show()
    
def plot_Nash_Ens():
    sheet_name='cptc2018'
    name_arr=['ARL','ABayes']
    def get_nash():
        y_arr=[]
        x_arr=[]
        COUNT=5
        df=pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/Robust_priority/算法效率评估.xlsx',sheet_name=sheet_name)
        y=np.array([])
        x=np.array([])
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
    for i in range(len(y_arr)):
        sns.lineplot(x=x_arr[i],y=y_arr[i],label=name_arr[i])

    plt.xlabel("Iteration",fontsize=20,labelpad=-6)
    plt.xticks(fontsize=23)
    plt.ylabel("Detection Rate (%)",fontsize=20)
    plt.yticks(fontsize=20,rotation=0, horizontalalignment= 'right',)
    plt.legend(prop={'size':20},fancybox=True,shadow=True)
    plt.show()
    # plt.savefig('/Users/zhanghangsheng/Documents/my_note/my_knowledge_map/my_paper/PhD_thesis/image/NashAE/'+sheet_name+'_nash.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)


    
if __name__=='__main__':

    # plot_priority()
    # plot_transfer()
    plot_Nash_Ens()



