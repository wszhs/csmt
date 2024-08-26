import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")

from csmt.Interpretability.interpret import set_visualize_provider
from csmt.Interpretability.interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from csmt.Interpretability.interpret.glassbox import DecisionListClassifier
from csmt.Interpretability.interpret import show

seed = 1
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

dl = DecisionListClassifier(random_state=seed)
dl.fit(X_train, y_train)

dl_global = dl.explain_global()
show(dl_global)

dl_local = dl.explain_local(X_test[:5], y_test[:5])
show(dl_local)