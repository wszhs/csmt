import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")

from csmt.Interpretability.interpret import set_visualize_provider
from csmt.Interpretability.interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from csmt.Interpretability.interpret.glassbox import ClassificationTree
from csmt.Interpretability.interpret import show

seed = 1
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

dt = ClassificationTree(random_state=seed)
dt.fit(X_train, y_train)

dt_global = dt.explain_global()
show(dt_global)

dt_local = dt.explain_local(X_test[:5], y_test[:5])
show(dt_local)