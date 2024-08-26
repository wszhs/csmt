import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")

from csmt.Interpretability.interpret import set_visualize_provider
from csmt.Interpretability.interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from csmt.Interpretability.interpret import show
from csmt.Interpretability.interpret.blackbox import LimeTabular

seed = 1
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

pca = PCA()
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

blackbox_model = Pipeline([('pca', pca), ('rf', rf)])
blackbox_model.fit(X_train, y_train)

lime = LimeTabular(predict_fn=blackbox_model.predict_proba, data=X_train)
lime_local = lime.explain_local(X_test[:5], y_test[:5])

show(lime_local)