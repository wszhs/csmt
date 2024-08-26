
from joblib import dump, load
import numpy as np

class AbstractModel:
    """
    Base model that all other models should inherit from.
    Expects that classifier algorithm is initialized during construction.
    """  
    def data_reshape(self,X):
        models_list=['RNN','LSTM']
        if X.ndim==2 and (self.classifier.model.__class__.__name__ in models_list):
            X=X.reshape(X.shape[0],X.shape[1],1)
        if X.ndim==3 and (self.classifier.model.__class__.__name__ not in models_list):
            X=X.reshape(X.shape[0],-1)
        return X

    def train(self, X_train, y_train,X_val,y_val):
        
        X_train=self.data_reshape(X_train)
        X_val=self.data_reshape(X_val)
        self.classifier.fit(X_train, y_train,X_val,y_val)

    def predict(self, X):
        X=self.data_reshape(X)
        return self.classifier.predict(X)

    def predict_label(self, X):
        X=self.data_reshape(X)
        pred=self.classifier.predict(X)
        label=np.argmax(pred, axis=1)
        return label

    #仅限异常检测模型
    def predict_anomaly(self,X,y):
        X=self.data_reshape(X)
        # from sklearn.metrics import roc_auc_score,precision_score
        from csmt.classifiers.anomaly_detection.pyod.utils.data import evaluate_print
        anomaly_scores=self.classifier.decision_function(X)
        roc=evaluate_print('model', y, anomaly_scores)
        # roc=np.round(roc_auc_score(y, anomaly_scores), decimals=4)
        print(roc)

    def save(self, path):
        dump(self.classifier, path)

    def load(self, path):
        self.classifier = load(path)
