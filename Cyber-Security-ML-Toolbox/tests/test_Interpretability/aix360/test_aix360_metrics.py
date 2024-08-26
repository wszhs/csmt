import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.get_model_data import get_datasets,parse_arguments,models_train,print_results,models_predict
import numpy as np
import torch
import random
from lime.lime_tabular import LimeTabularExplainer
from csmt.Interpretability.aix360.metrics import faithfulness_metric, monotonicity_metric
import sklearn
import sklearn.datasets

iris = sklearn.datasets.load_iris()
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, 
                                                                                  iris.target, 
                                                                                  train_size=0.80)

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train, labels_train)

score=sklearn.metrics.accuracy_score(labels_test, rf.predict(test))
print(score)

explainer = LimeTabularExplainer(train, 
                             feature_names=iris.feature_names, 
                             class_names=iris.target_names, 
                             discretize_continuous=True)

i = np.random.randint(0, test.shape[0])
exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=4, top_labels=1)

exp.as_map()

predicted_class = rf.predict(test[i].reshape(1,-1))[0]
le = exp.local_exp[predicted_class]
m = exp.as_map()
x = test[i]
coefs = np.zeros(x.shape[0])

for v in le:
    coefs[v[0]] = v[1]
base = np.zeros(x.shape[0])

print(coefs)
print("Faithfulness: ", faithfulness_metric(rf, x, coefs, base))
print("Monotonity: ", monotonicity_metric(rf, x, coefs, base))

ncases = test.shape[0]
fait = np.zeros(ncases)
for i in range(ncases):
    predicted_class = rf.predict(test[i].reshape(1,-1))[0]
    exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=4, top_labels=1)
    le = exp.local_exp[predicted_class]
    m = exp.as_map()
    
    x = test[i]
    coefs = np.zeros(x.shape[0])
    
    for v in le:
        coefs[v[0]] = v[1]
    fait[i] = faithfulness_metric(rf, test[i], coefs, base)

print("Faithfulness metric mean: ",np.mean(fait))
print("Faithfulness metric std. dev.:", np.std(fait))