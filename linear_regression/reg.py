import pandas as pd
import numpy as np
import sympy
class linear_reg:
    def __init__(self,learning_rate,n_iteration):
        self.learning_rate=learning_rate
        self.n_iteration=n_iteration
        self.weight=np.array([])
    def fit(self,x,y):
        (n_sample,n_feature)=x.shape
        f0=np.ones((n_sample,1))
        xnew=np.concatenate((f0,x),axis=1)
        w=np.zeros((n_feature+1))
        for it in range(self.n_iteration):
            diff=float(1/n_sample)*np.dot(xnew.T,np.dot(xnew,w)-y)
            w=w-self.learning_rate*diff
        self.weight=w
    def predict(self,x_test):
        (n_sample,n_feature)=x_test.shape
        f0=np.ones((n_sample,1))
        xnew=np.concatenate((f0,x_test),axis=1)
        return np.dot(xnew,self.weight)
    def coeff(self):
        return self.weight


    
        


    