import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np


if __name__=='__main__':
    boston = load_boston()
    X = boston.data
    y = boston.target
    feature_names = boston.feature_names

    fig, axs = plt.subplots(nrows = 3, ncols=5, figsize=(15, 6))
    for i, (ax, col) in enumerate(zip(axs.flat, feature_names)):
        x = X[:,i]
        pf = np.polyfit(x, y, 1)
        p = np.poly1d(pf)
        ax.plot(x, y, 'o')
        ax.plot(x, p(x),"r--")
        ax.set_title(col + ' vs Y')
        ax.set_xlabel(col)
        ax.set_ylabel('Y')
    plt.show()