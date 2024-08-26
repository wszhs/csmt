import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
import pandas as pd

def get_data():
    rewards_arr=[]
    episode_arr=[]
    rewards1 = np.array([0, 0.1,0,0.2,0.4,0.5,0.6,0.9,0.9,0.9])
    rewards2 = np.array([0, 0,0.1,0.4,0.5,0.5,0.55,0.8,0.9,1])
    rewards3 = np.array([0, 0.2,0.1,0.8,0.5,0.5,0.55,0.8,0.9,1])
    rewards=np.concatenate((rewards1,rewards2,rewards3)) 
    rewards_arr.append(rewards)
    episode1=range(len(rewards1))
    episode2=range(len(rewards2))
    episode3=range(len(rewards3))
    episode=np.concatenate((episode1,episode2,episode3))
    episode_arr.append(episode)
    return rewards_arr,episode_arr


rewards_arr,episode_arr=get_data()

for i in range(len(rewards_arr)):
    sns.lineplot(x=episode_arr[i],y=rewards_arr[i])

plt.xlabel("episode")
plt.ylabel("reward")
plt.show()