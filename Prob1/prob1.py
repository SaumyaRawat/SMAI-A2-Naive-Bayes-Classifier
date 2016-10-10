import csv
import random
import math
import numpy as np

class Model:
    def __init__(self, file):
        self.trainingFile = file
        self.dataset = []
        self.trainingSet= []
        self.testSet = []
        self.summaries = {}


    def load_database(self):
           with open(self.trainingFile, 'r') as csvfile:
               reader = csv.reader(csvfile, delimiter=',')
               next(reader) # Skip first row
               dataset = list(reader)
               for i in range(len(dataset)):
                   try:
                       try_list = [float(x) for x in dataset[i]]
                   except ValueError as e:
                        print("error",e,"on line",i)
                   dataset[i] = [float(j) for j in dataset[i]]
           self.dataset = dataset
    
    def k_fold_split(self,splitRatio):
    	trainSize = int(len(self.dataset) * splitRatio)
    	trainSet = []
    	copy = list(self.dataset)
    	while len(trainSet) < trainSize:
    		index = random.randrange(len(copy))
    		trainSet.append(copy.pop(index))
    	self.trainingSet = trainSet
    	self.testSet = copy
    
    def seperate_by_class(self):
    	separated = {}
    	for i in range(len(self.dataset)):
    		vector = self.dataset[i]
    		if (vector[-1] not in separated):
    			separated[vector[-1]] = []
    		separated[vector[-1]].append(vector)
    	return separated
    
    def summarize(self,dataset):
    	summaries = [(np.mean(attribute), np.std(attribute,ddof=1)) for attribute in zip(*dataset)]
    	del summaries[-1]
    	return summaries
    
    def summarize_by_class(self):
    	separated = self.seperate_by_class()
    	summaries = {}
    	for classValue, instances in separated.items():
    		summaries[classValue] = self.summarize(instances)
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
    			x = inputVector[i]
    			probabilities[classValue] *= self.calc_probability(x, mean, stdev)
    	return probabilities
    			
    def predict(self, inputVector):
    	probabilities = self.calc_class_probabilities(inputVector)
    	bestLabel, bestProb = None, -1
    	for classValue, probability in probabilities.items():
    		if bestLabel is None or probability > bestProb:
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
    		if self.testSet[i][-1] == predictions[i]:
    			correct += 1
    	return (correct/float(len(self.testSet))) * 100.0

if __name__ == "__main__":
        model = Model("dataset/train_data.csv")
        model.load_database()
        splitRatio = 0.7
        model.k_fold_split(splitRatio)
        print('Size of train=',len(model.trainingSet),' and test=',len(model.testSet))
        model.summarize_by_class()
        predictions = model.test_classifier()
        accuracy = model.get_accuracy(predictions)
        print('Accuracy: ',accuracy)