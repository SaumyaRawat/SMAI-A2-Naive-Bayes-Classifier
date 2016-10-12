#!/bin/python

#import global_variables as gv
from pca import pca_X
import numpy as np

global train_labels
train_labels=np.loadtxt('dataset/dorothea_train.labels')
class_indices = []
class_indices.append(np.where(train_labels==-1))
class_indices.append(np.where(train_labels==1))

#replace class label -1 with 0 for easier indexing
train_labels[class_indices[0]] = 0

#dimension
n = pca_X.shape[0]
d = pca_X.shape[1]

#pca_X = np.real(pca_X)

#Calculate n dimensional means

overall_mean = np.mean(pca_X, axis=0)
mean_vectors = []
for class_no in range(2):
    mean_vectors.append(np.mean(pca_X[class_indices[class_no]], axis=0))


#Calculate within class scatter matrix
S_W = np.zeros((d,d))
for class_no,mean_vector in zip(range(2), mean_vectors):
    scatter_matrix = np.zeros((d,d))                 
    for row in pca_X[class_indices[class_no]]:
        row, mean_vector = row.reshape(d,1), mean_vector.reshape(d,1)
        scatter_matrix += (row-mean_vector).dot((row-mean_vector).T)
    S_W += scatter_matrix                             


#Calculate between class scatter matrix
S_B = np.zeros((d,d))
for i,mean_vector in enumerate(mean_vectors):  
    n = pca_X[class_indices[i],:].shape[0]
    mean_vector = mean_vector.reshape(d,1)
    overall_mean = overall_mean.reshape(d,1) # make column vector
    S_B += n * (mean_vector - overall_mean).dot((mean_vector - overall_mean).T)

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Construct KxD eigenvector matrix W
W = eig_pairs[0][1]

global lda_X
lda_X = np.real(pca_X.dot(W))
