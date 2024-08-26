
from csmt.attacks.evasion import ZeroOrderOptMethod,HopSkipJump,BoundaryMethod,\
    DEEvasionAttack,GAEvasionAttack,\
    ZOSGDMethod,ZONESMethod,ZOSCDMethod,ZOAdaMMMethod,\
    BayesOptMethod,GradFreeMethod,OpenboxMethod,MimicryMethod,GANMethod

class GradientFreeEvasionAttack():
    def __init__(self,estimator,eps,eps_step,max_iter,norm,estimator_features):
        self.estimator=estimator
        self.eps=eps
        self.eps_step=eps_step
        self.max_iter=max_iter
        self.norm=norm
        self.estimator_features=estimator_features

class ZOOAttack(GradientFreeEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=ZeroOrderOptMethod(estimator=self.estimator.classifier)
        X_adv,X_adv_path=attack.generate(X,y,**kwargs)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class HSJAttack(GradientFreeEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=HopSkipJump(estimator=self.estimator.classifier)
        X_adv,X_adv_path=attack.generate(X,y,**kwargs)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class BoundAttack(GradientFreeEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=BoundaryMethod(estimator=self.estimator.classifier)
        X_adv,X_adv_path=attack.generate(X,y,**kwargs)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class DEAttack(GradientFreeEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=DEEvasionAttack(estimator=self.estimator.classifier,eps=self.eps,eps_step=self.eps_step,max_iter=self.max_iter,norm=self.norm)
        X_adv,X_adv_path=attack.generate(X,y,**kwargs)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class GAAttack(GradientFreeEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=GAEvasionAttack(estimator=self.estimator.classifier,eps=self.eps,eps_step=self.eps_step,max_iter=self.max_iter,norm=self.norm)
        X_adv,X_adv_path=attack.generate(X,y,**kwargs)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class ZOSGDAttack(GradientFreeEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=ZOSGDMethod(estimator=self.estimator.classifier,eps=self.eps,eps_step=self.eps_step,max_iter=self.max_iter,norm=self.norm,estimator_features=self.estimator_features)
        X_adv,X_adv_path=attack.generate(X,y,**kwargs)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class ZONESAttack(GradientFreeEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=ZONESMethod(estimator=self.estimator.classifier,eps=self.eps,eps_step=self.eps_step,max_iter=self.max_iter,norm=self.norm)
        X_adv,X_adv_path=attack.generate(X,y,**kwargs)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class ZOSCDAttack(GradientFreeEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=ZOSCDMethod(estimator=self.estimator.classifier,eps=self.eps,eps_step=self.eps_step,max_iter=self.max_iter,norm=self.norm)
        X_adv,X_adv_path=attack.generate(X,y,**kwargs)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class ZOAdaMMAttack(GradientFreeEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=ZOAdaMMMethod(estimator=self.estimator.classifier,eps=self.eps,eps_step=self.eps_step,max_iter=self.max_iter,norm=self.norm)
        X_adv,X_adv_path=attack.generate(X,y,**kwargs)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class GANAttack(GradientFreeEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=GANMethod(estimator=self.estimator.classifier,eps=self.eps,eps_step=self.eps_step,max_iter=self.max_iter,norm=self.norm)
        X_adv,X_adv_path=attack.generate(X,y,**kwargs)
        y_adv=y
        return X_adv,y_adv,X_adv_path
    
class MimicryAttack(GradientFreeEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=MimicryMethod(estimator=self.estimator.classifier,eps=self.eps,eps_step=self.eps_step,max_iter=self.max_iter,norm=self.norm)
        X_adv,X_adv_path=attack.generate(X,y,**kwargs)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class BayesAttack(GradientFreeEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=BayesOptMethod(estimator=self.estimator.classifier,eps=self.eps,eps_step=self.eps_step,max_iter=self.max_iter,norm=self.norm)
        X_adv,X_adv_path=attack.generate(X,y,**kwargs)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class GradFreeAttack(GradientFreeEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=GradFreeMethod(estimator=self.estimator.classifier,eps=self.eps,eps_step=self.eps_step,max_iter=self.max_iter,norm=self.norm)
        X_adv,X_adv_path=attack.generate(X,y,**kwargs)
        y_adv=y
        return X_adv,y_adv,X_adv_path

class OpenboxAttack(GradientFreeEvasionAttack):
    def generate(self,X,y,**kwargs):
        attack=OpenboxMethod(estimator=self.estimator.classifier,eps=self.eps,eps_step=self.eps_step,max_iter=self.max_iter,norm=self.norm)
        X_adv,X_adv_path=attack.generate(X,y,**kwargs)
        y_adv=y
        return X_adv,y_adv,X_adv_path




