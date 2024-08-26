'''
Author: your name
Date: 2021-04-01 17:32:59
LastEditTime: 2021-08-03 14:26:22
LastEditors: your name
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/attacks/evasion/gradient.py
'''
from csmt.attacks.evasion import FastGradientMethod,CarliniLInfMethod,CarliniL2Method,ProjectedGradientDescent,SaliencyMapMethod,DeepFool,BasicIterativeMethod,UniversalPerturbation
import numpy as np
import copy

class GradientEvasionAttack():
    def __init__(self,estimator,eps,eps_step,max_iter,norm,upper,lower,feature_importance,mask=None):
        self.estimator=estimator
        self.eps=eps
        self.eps_step=eps_step
        self.max_iter=max_iter
        self.norm=norm
        self.upper=upper
        self.lower=lower
        self.mask=mask
        self.feature_importance=feature_importance

class UniversalAttack(GradientEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=UniversalPerturbation(classifier=self.estimator.classifier,eps=self.eps,norm=self.norm)
        X_adv,X_adv_path=attack.generate(X,y,mask=self.mask)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class FGSMAttack(GradientEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=FastGradientMethod(estimator=self.estimator.classifier,eps=self.eps,norm=self.norm)
        X_adv,X_adv_path=attack.generate(X,y,mask=self.mask)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class PGDAttack(GradientEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=ProjectedGradientDescent(estimator=self.estimator.classifier,eps=self.eps,eps_step=self.eps_step,max_iter=self.max_iter,norm=self.norm,verbose=False)
        X_adv,X_adv_path=attack.generate(X,y,mask=self.mask)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class BIMAttack(GradientEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=BasicIterativeMethod(estimator=self.estimator.classifier,eps=self.eps,eps_step=self.eps_step,max_iter=self.max_iter)
        X_adv,X_adv_path=attack.generate(X,y,mask=self.mask)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class CWAttack(GradientEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=CarliniLInfMethod(classifier=self.estimator.classifier,eps=self.eps)
        X_adv,X_adv_path=attack.generate(X,y,mask=self.mask)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class JSMAAttack(GradientEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=SaliencyMapMethod(classifier=self.estimator.classifier,theta= 0.1,gamma= 1)
        X_adv,X_adv_path=attack.generate(X,y)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class DeepFoolAttack(GradientEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=DeepFool(classifier=self.estimator.classifier,max_iter=10,epsilon=0.01)
        X_adv,X_adv_path=attack.generate(X,y,mask=self.mask)
        y_adv=y
        return X_adv,y_adv,X_adv_path



