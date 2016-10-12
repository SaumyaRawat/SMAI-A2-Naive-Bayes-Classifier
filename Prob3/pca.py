#!/bin/python

import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
import array
from scipy.sparse import csr_matrix, linalg, lil_matrix
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
from copy import deepcopy
import global_variables as gv

gv.init()
lines = open('dataset/dorothea_train.data', 'r').readlines()
dataList = [line.rstrip('\n') for line in lines]
k = len(max(dataList,key=len))
#data = [[]]

#x=for i in range(len(dataList)):
    #dataList[0].extend([data0]*(k-len(dataList[i])))
cols = np.arange(k)

train_data = pd.read_csv(
    filepath_or_buffer='dataset/dorothea_train.data', 
    header=None,
    sep=" ",
    names=cols,
    engine = 'python')

train_data=train_data.fillna(0)
X = lil_matrix((train_data.shape[0],100001))
for i in range(train_data.shape[0]):
   X[i,train_data[i]] = 1


#train_labels = pd.read_csv(filepath_or_buffer='dataset/dorothea_train.labels', header=Non

#Standardizing the data so that all features are on the same scale
X_std = MaxAbsScaler().fit_transform(X)
#M = X_std.dot(X_std.T) #covariance matrix
      
#Computing covariance matrix
#mean_v = X_std.T.mean()
#norm = linalg.norm(X_std)
#s_cov_mat = X_std.dot(X_std.T.conjugate()) / norm

#cov_mat=np.cov(X_std.toarray().T)

#nsamples = X_std.shape[1]
#mean_mat = np.outer(np.ones((nsamples,1)),mean_v)
#cov_mat = X_std.T - mean_mat
#cov_mat = np.dot(cov_mat,cov_mat.T)/(nsamples -1)
    
#EigenDecomposition : 
k=100
#k=500
#k=1000

#eVals, eVecs = linalg.eigs(M,k,which='LM')
#tmp = X_std.T.dot(eVecs).T #this is the compact trick
#V = tmp[::-1] #reverse since last eigenvectors are the ones we want
#S = np.sqrt(eVals)[::-1] #reverse since eigenvalues are in increasing order

#Y=S.T.dot(X_std)
# Make a list of (eigenvalue, eigenvector) tuples
# d×k-dimensional eigenvector matrix W.
#W = [(abs(eVals[i]), eVecs[:,i]) for i in range(len(eVals))]
#Y=X×W
#Y=X_std.dot(eVecs.real)
#Y=np.dot(eVecs.T, X_std.T).T

#Kernel_PCA since d>>n
S=X_std.dot(X_std.T)
val,vec=linalg.eigs(S,k,which='LM')
# d×k-dimensional eigenvector matrix W.
W=X_std.T.dot(vec)
Y=X_std.dot(W)

pca_X = deepcopy(Y)