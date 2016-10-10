#!/bin/python
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
from numpy import *
import array
from scipy.sparse import csr_matrix, linalg
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls

lines = open('dataset/dorothea_train.data', 'r').readlines()
dataList = [line.rstrip('\n') for line in lines]
k = len(max(dataList,key=len))
cols = np.arange(k)

train_data = pd.read_csv(
    filepath_or_buffer='dataset/dorothea_train.data', 
    header=None,
    sep=" ",
    names=cols,
    engine = 'python')

train_data=train_data.fillna(0)
X = csr_matrix(train_data.values)

train_labels=np.loadtxt('dataset/dorothea_train.labels')

class1_indices = np.where(train_labels==-1)
class2_indices = np.where(train_labels==1)

# Variables
n1=len(class1_indices[0])
n2=len(class2_indices[0])
n=n1+n2

K=X.dot(X.T)

class1=(train_labels==-1).astype(int)
class2=(train_labels==1).astype(int)
classes=class1+class2


#Compute scatter matrices
## Between Class Scatter Matrix
P1=(class1.T/n1-classes.T/n)*(class1.T/n1-classes.T/n).T
P2=(class2.T/n2-classes.T/n)*(class2.T/n2-classes.T/n).T
SB=K*(n1*P1+n2*P2)*K.T

## Within Class Scatter Matrix
Q1=(np.eye(n,n)-np.tile(class1.T/n1,(n,1))) * np.diag(class1) * (np.eye(n,n)-np.tile(class1/n1,(n,1))) * np.diag(class1).T
Q2=(np.eye(n,n)-np.tile(class2.T/n1,(n,1))) * np.diag(class2) * (np.eye(n,n)-np.tile(class2/n2,(n,1))) * np.diag(class2).T
SW=K*(Q1+Q2)*K.T

val,vec=np.linalg.eig(SW)

#Chosen 10^-8 because its the closest value to singularity
#v=SW+np.power(10,8)*minval*np.eye(len(SW))*SB;
#max_val,max_vec=linalg.eigs(csr_matrix(v),1,which='LM')

#W=X*max_vec;
#W=W/np.norm(W,2);

#Y=X.dot(v)
#Y1=X[:, np.ravel(class1_indices)]*v
#Y2=X[:, np.ravel(class2_indices)]*v
