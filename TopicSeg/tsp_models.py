import re
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

'''
This script aims to build Topic Score Predictor which is used to assign sequence with a topic confidence score, also considered as probabilities;
The Topic Score Predictor could be built by using Naive Bayes, Linear Support Vector Machine, decision tree and Nearest Neighbors;
class predictors(model="nb",dataset_file="../Datasets/LabeledDataset.txt").load_predictor() 
return the corresponding classifier based on selected model.
'''

# Initialize the dictionary of all words from training set
def __build_dictionary(dataset):
    allwords = []
    alllabels=[]
    for line in dataset:
        label, Text = re.split(" +", line, 1)
        words = re.split(" +", Text)
        alllabels.append(label)
        for word in words:
            allwords.append(word)
    return list(set(alllabels)),list(set(allwords))

def __lable2id(label,labels_list):
    for i in range(len(labels_list)):
        if label == labels_list[i]:
            return i
    raise Exception('Error lable %s' % (label))

# Convert each document into BOW feature vector
def dataset_initial(dataset):
    labels_list,dictionary=__build_dictionary(dataset)
    doc_labels = []
    docvectors = []
    for line in dataset:
        docvector = [0] * len(dictionary)
        label, Text = re.split(" +", line, 1)
        doc_labels.append(__lable2id(label,labels_list))
        words = re.split(" +", Text)
        for word in words:
            if word in dictionary:
                docvector[dictionary.index(word)] += 1
        docvectors.append(docvector)
    return np.array(docvectors), np.array(doc_labels),dictionary,labels_list

class predictors(object):

    def __init__(self,model="nb",dataset_file="../Datasets/LabeledDataset.txt"):

        if model=="nb": self.predictor=MultinomialNB()
        elif model=="svm": self.predictor=SVC(kernel='linear', probability=True)
        elif model=="dtree": self.predictor=DecisionTreeClassifier()
        elif model=="kn": self.predictor=KNeighborsClassifier()
        else:
            raise Exception('Error model %s' % (model))

        self.__dataset_X,self.__dataset_y,self.dictionary,self.labels=dataset_initial([line.strip() for line in open(dataset_file)])

    def load_predictor(self):
        self.predictor.fit(self.__dataset_X,self.__dataset_y)
        return self.predictor

    def get_dictionary(self):
        return self.dictionary,self.labels





