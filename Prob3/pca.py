#!/bin/python

import pandas as pd
from sklearn.preprocessing import KernelCenterer
import numpy as np
import array
from scipy.sparse import csr_matrix, linalg, lil_matrix
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
from copy import deepcopy



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
K=X.dot(X.T)

#Centering Kernel since data has to be standardizied
kern_cent = KernelCenterer()
S = kern_cent.fit_transform(K.toarray())

#val,vec=linalg.eigs(S,k,which='LM')

eig_vals, eig_vecs = np.linalg.eig(S)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
vec = np.array([ eig_pairs[i][1] for i in range(k)])
vec = vec.T # to make eigen vector matrix nxk

# d×k-dimensional eigenvector matrix W.
W=X.T.dot(vec)
Y=X.dot(W)

global pca_X
pca_X = deepcopy(Y)