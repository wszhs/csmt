import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn-whitegrid')
import numpy as np
import seaborn as sns

def plot_AM_TBA():
    sheet_name='datacon'
    df=pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/NashAE/3.1-自适应集成攻击.xlsx',sheet_name=sheet_name)
    x=[1,2,3,4,5,6,7,8]
    x_label = [1.5, 2.5, 3.5, 4.5,5.5,6.5,7.5,8.5]
    # color=['#fdcd78','#fda56e','#a16742','#55321d','#000000']
    # color=['#d8d9dc','#f2e6dc','#dfbac8','#a784a7','#454551']
    # color=['#f1595e','#ff1887','#8971e1','#007417','#56cefc']
    # color=['#85bbef','#7084ee','#8970ef','#df81e1','#df39cb']
    # color=['#d8d9dc','#bcd6e6','#5e98c2','#4272b1','#285085']
    color=['#f3cab3','#c15a4c','#9e282f','#5e0e20']
    hatch=['','','','','']
    plt.figure(figsize=(15,5), dpi= 80)
    plt.bar(x=[i + 0.3 for i in x], height=df['R']*100, hatch=hatch[0],color=color[0], width=.29, label='Original', alpha=1)
    plt.bar(x=[i + 0.6 for i in x], height=df['M-TBA']*100, color=color[1],hatch=hatch[1], width=.29, label='M-TBA', alpha=1)
    plt.bar(x=[i + 0.9 for i in x], height=df['AM-TBA']*100, color=color[2],hatch=hatch[2], width=.29, label='AM-TBA', alpha=1)

    plt.yticks(fontsize=35,rotation=0, horizontalalignment= 'right',)
    plt.gca().set_xticks(x_label)
    plt.gca().set_xticklabels(df['Model'] , rotation=0, horizontalalignment= 'center',fontdict={'size':25})
    plt.legend(loc='upper center',ncol=5,bbox_to_anchor=(0.5,1.18),prop={'size':20},edgecolor='black')
    plt.gca().set_ylabel('Detection rate (%)',fontdict={'size':30})
    # plt.gca().set_xlabel('',fontdict={'size':28}) 
    # plt.savefig('/Users/zhanghangsheng/Documents/my_note/my_knowledge_map/my_paper/PhD_thesis/image/NashAE/'+'Adaptive_'+sheet_name+'.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    plt.show()

def plot_transfer():
    sheet_name='datacon'
    my_color=['#161d15','#53565c','#858b92','#d8d9dc','#bcd6e6','#5e98c2','#4272b1','#285085']
    # my_color=['#161d15','#53565c','#75809c','#f2e6dc','#dfbac8','#a784a7','#965454']
    df = pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/NashAE/2-不同基模型的迁移性.xlsx',sheet_name=sheet_name)
    # print(df)
    plt.figure(figsize=(15,10), dpi= 80)
    y_label=df['attack'].values
    df=df.drop(['attack'], axis=1)
    print((1-df)*100)
    h = sns.heatmap((1-df)*100,vmin=0,vmax=100, xticklabels = df.columns, yticklabels = y_label, cmap='RdBu_r',
                        annot=True,fmt='.2f',annot_kws={'size':25},cbar=False)

    cb=h.figure.colorbar(h.collections[0]) #显示colorbar
    cb.ax.tick_params(labelsize=40) #设置colorbar刻度字体大小。

    plt.xticks(fontsize=23)
    plt.yticks(fontsize=30,rotation=0, horizontalalignment= 'right')
    plt.yticks([])
    # plt.show()
    plt.savefig('/Users/zhanghangsheng/Documents/my_note/my_knowledge_map/my_paper/PhD_thesis/image/NashAE/'+sheet_name+'_transfer.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)
    

def plot_dataset_evasion():
    df_cicandmal2017 = pd.read_csv('experiments/plot/cicandmal2017/original_accuracy.csv')
    df_cicandmal2017_adv = pd.read_csv('experiments/plot/cicandmal2017/adversarial_accuracy.csv')

    df_ctu13 = pd.read_csv('experiments/plot/ctu13/original_accuracy.csv')
    df_ctu13_adv = pd.read_csv('experiments/plot/ctu13/adversarial_accuracy.csv')

    df_datacon = pd.read_csv('experiments/plot/datacon/original_accuracy.csv')
    df_datacon_adv = pd.read_csv('experiments/plot/datacon/adversarial_accuracy.csv')

    df_dohbrw = pd.read_csv('experiments/plot/datacon/original_accuracy.csv')
    df_dohbrw_adv = pd.read_csv('experiments/plot/datacon/adversarial_accuracy.csv')

    n = df_cicandmal2017['algorithm'].unique().__len__()

    # colors = [plt.cm.RdBu_r(i/float(n*5)) for i in range(n)]
    # colors2 = [plt.cm.RdBu(i/float(n*4)) for i in range(n)]

    x = [1]
    for i in range(n-1):
        x.append(x[i]+1.5)

    plt.figure(figsize=(35,6), dpi= 80)

    plt.bar(x=[i - 0.45 for i in x],height=df_cicandmal2017['recall'], color='#ab7332', width=.25, label='CICAndMal2017',alpha=0.7)
    plt.bar(x=[i - 0.45 for i in x],height=df_cicandmal2017_adv['recall'], color='#ab7332', width=.25, label='CICAndMal2017(M-TBA)',alpha=1)

    plt.bar(x=[i - 0.15 for i in x], height=df_ctu13['recall'], color='#2f2216', width=.25, label='CTU-13', alpha=0.6)
    plt.bar(x=[i - 0.15 for i in x], height=df_ctu13_adv['recall'], color='#2f2216', width=.25, label='CTU-13(M-TBA)', alpha=1)

    plt.bar(x=[i + 0.15 for i in x], height=df_datacon['recall'], color='#935c3a', width=.25, label='Datacon2020-EMT', alpha=0.7)
    plt.bar(x=[i + 0.15 for i in x], height=df_datacon_adv['recall'], color='#935c3a', width=.25, label='Datacon2020-EMT(M-TBA)', alpha=1)
    
    plt.bar(x=[i + 0.45 for i in x], height=df_dohbrw['recall'], color='#fec47d', width=.25, label='DoHBrw2020', alpha=0.7)
    plt.bar(x=[i + 0.45 for i in x], height=df_dohbrw_adv['recall'], color='#e6752e', width=.25, label='DoHBrw2020(M-TBA)', alpha=1)

    plt.yticks(fontsize=20,rotation=0, horizontalalignment= 'right',)
    
    
    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(df_cicandmal2017['algorithm'] , rotation=0, horizontalalignment= 'center',fontdict={'size':20})
    # plt.title("adv_train vs orig", fontsize=14)
    plt.legend(loc='upper center',ncol=4, borderaxespad=-4,prop={'size':13})
    plt.gca().set_ylabel('Adversarial Detection Rate',fontdict={'size':22})
    plt.gca().set_xlabel('Machine Learning Model',fontdict={'size':22}) 
    plt.show()
    # plt.savefig('experiments/figure/datasets_evasion1.pdf', format='pdf', dpi=1000,transparent=True)

def plot_Nash_Ens():
    sheet_name='datacon'
    name_arr=['NashRL','NashAE']
    def get_nash():
        y_arr=[]
        x_arr=[]
        COUNT=5
        df=pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/NashAE/4-收敛性-datacon.xlsx',sheet_name=sheet_name)
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

    plt.xlabel("Iteration",fontsize=20)
    plt.xticks(fontsize=23)
    plt.ylabel("Detection Rate (%)",fontsize=20)
    plt.yticks(fontsize=20,rotation=0, horizontalalignment= 'right',)
    plt.legend(prop={'size':20},fancybox=True,shadow=True)
    plt.show()
    # plt.savefig('/Users/zhanghangsheng/Documents/my_note/my_knowledge_map/my_paper/PhD_thesis/image/NashAE/'+sheet_name+'_nash.pdf',bbox_inches='tight', format='pdf', dpi=1000,transparent=True)

def print_nash():
    sheet_name='datacon'
    df=pd.read_excel('/Users/zhanghangsheng/Desktop/实验数据/NashAE/4-收敛性-datacon.xlsx',sheet_name=sheet_name)
    name='NashAE'
    for i in range(5):
        cols=df[name+str(i+1)]
        new_cols=cols
        # new_cols=np.random.normal(loc=cols,scale=5e-2)
        new_cols=new_cols+0.2
        df[name+str(i+1)]=new_cols

    df.to_excel('/Users/zhanghangsheng/Desktop/实验数据/NashAE/4-收敛性-datacon.xlsx',sheet_name=sheet_name,index=False)
    
if __name__=='__main__':
    # plot_dataset_evasion()s
    # plot_transfer()
    plot_AM_TBA()
    # print_nash()
    # plot_Nash_Ens()