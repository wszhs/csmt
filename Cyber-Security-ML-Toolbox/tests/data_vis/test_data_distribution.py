import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)

sns.set_style("darkgrid")
x = np.random.normal(size=200)


# sns.distplot(x, color='y')

# sns.distplot(x, kde=False, rug=True, bins=20)

# sns.distplot(x, hist=False, rug=True, color='g')

# sns.kdeplot(x)
# sns.kdeplot(x, bw=.25, label="bw:0.25")
# sns.kdeplot(x, bw=3, label="bw:3")
# plt.legend()

# sns.kdeplot(x, shade=True, cut=0, color='y')
# sns.rugplot(x, color='y')

# x = np.random.gamma(7, size=200)
# sns.distplot(x, kde=False, fit=stats.gamma)


# #设置均值与协方差矩阵
# mean, cov = [1.7, 130], [(1, .65),(.65, 1)]
# data = np.random.multivariate_normal(mean, cov, 200)
# #生成结构化格式
# df = pd.DataFrame(data, columns=["x", "y"])
# sns.jointplot(x='x', y='y', data=df)

iris = sns.load_dataset("iris")
print(iris)
# sns.set_palette(sns.color_palette("BuGn_d"))
# sns.pairplot(iris)

sns.set_palette(sns.color_palette("RdPu"))
g = sns.PairGrid(iris, hue="species")
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)
g = g.add_legend()


plt.show()

