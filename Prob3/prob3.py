import csv
import random
import math
import numpy as np
from lda import lda_X, train_labels
from pca import pca_X
import warnings


class Model:
	def __init__(self, X, labels):
		self.dataset = X
		self.class_labels = labels
		self.test_class_labels = list(labels)
		self.trainingSet= []
		self.testSet = []
		self.summaries = {}


	def k_fold_split(self,splitRatio):
		trainSize = int(len(self.dataset) * splitRatio)
		trainSet = []
		copy = list(self.dataset)
		while len(trainSet) < trainSize:
			index = random.randrange(len(copy))
			trainSet.append(copy.pop(index))
			self.test_class_labels.pop(index)
		self.trainingSet = trainSet
		self.testSet = copy
	
	def seperate_by_class(self):
		separated = {}
		for i in range(len(self.dataset)):
			vector = self.dataset[i]
			label = self.class_labels[i]
			if (label not in separated):
				separated[label] = []
			separated[label].append(vector)
		return separated
	
	def summarize_pca(self,instances):          #Parameters of Gaussian
		summaries = [(np.mean(attribute), np.std(attribute,ddof=1)) for attribute in zip(*instances)]
		return summaries

	def summarize_lda(self,instances):          #Parameters of Gaussian
		summaries = [(np.mean(instances), np.std(instances))]
		return summaries

	def train_classifier(self,data_flag):
		separated = self.seperate_by_class()
		summaries = {}
		for classValue, instances in separated.items():
		 if data_flag is 0:
			 summaries[classValue] = self.summarize_pca(instances)
		 elif data_flag is 1:
			 summaries[classValue] = self.summarize_lda(instances)
		self.summaries = summaries
	

	def calc_probability(self,x, mean, stdev):
		exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
		return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
	
	def calc_class_probabilities(self, inputVector):
		probabilities = {}
		for classValue, classSummaries in self.summaries.items():
			probabilities[classValue] = 1
			for i in range(len(classSummaries)):
				mean, stdev = classSummaries[i]
				if type(inputVector) == np.float64 :
					x = inputVector
				else:
					x = inputVector[i]
				probabilities[classValue] *= self.calc_probability(x, mean, stdev)
		return probabilities
				
	def predict(self, inputVector):
		probabilities = self.calc_class_probabilities(inputVector)
		bestLabel, bestProb = None, -1
		for classValue, probability in probabilities.items():
			if bestLabel is None or probability > bestProb:       #Assign the label with the highest probability
				bestProb = probability
				bestLabel = classValue
		return bestLabel
	
	def test_classifier(self):
		predictions = []
		for i in range(len(self.testSet)):
			result = self.predict(self.testSet[i])
			predictions.append(result)
		return predictions
	
	def get_accuracy(self,predictions):
		correct = 0
		for i in range(len(self.testSet)):
			if self.test_class_labels[i] == predictions[i]:
				correct += 1
		return (correct/float(len(self.testSet))) * 100.0

if __name__ == "__main__":    
		warnings.simplefilter("error")
		
		# PCA
		model_pca = Model(pca_X,train_labels)
		splitRatio = 0.5
		model_pca.k_fold_split(splitRatio)
		print('Size of train=',len(model_pca.trainingSet),' and test=',len(model_pca.testSet))
		model_pca.train_classifier(0)
		predictions = model_pca.test_classifier()
		accuracy_pca = model_pca.get_accuracy(predictions)
		print('Accuracy after PCA(K==500): ',accuracy_pca)
		
		# LDA
		model_lda = Model(lda_X,train_labels)
		splitRatio = 0.5
		model_lda.k_fold_split(splitRatio)
		print('Size of train=',len(model_lda.trainingSet),' and test=',len(model_lda.testSet))
		model_lda.train_classifier(1)
		predictions = model_lda.test_classifier()
		accuracy_lda = model_lda.get_accuracy(predictions)
		print('Accuracy after LDA: ',accuracy_lda)