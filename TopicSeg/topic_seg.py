from TopicSeg.tsp_models import predictors
import re
import numpy as np

class segmenter(object):

    def __init__(self,predictor_model='nb',
                 dataset_file="../Datasets/LabeledDataset.txt",
                 labels_dic={'A':'History',
                             'B':'Labs',
                             'C':'Medications',
                             'D':'PhysicalExam',
                             'E':'Courses'},
                 threshold=0.5):
        predictor=predictors(predictor_model,dataset_file)
        self.__predictor=predictor.load_predictor()
        self.__dictionary,self.__labels=predictor.get_dictionary()
        self.__labels_dic=labels_dic
        self.__threshold=threshold

    def __datatoVector(self,line):
        words = re.split(" +", line)
        docvector = [0] * len(self.__dictionary)
        for word in words:
            if word in self.__dictionary:
                docvector[self.__dictionary.index(word)] += 1
        return np.array([docvector])

    def __ifboundary(self,diff_value):

        ifboundary=0
        score=np.min(diff_value)+(np.sum(diff_value)-np.min(diff_value))/4
        if(score<self.__threshold*(-1)):
            ifboundary=1
        return ifboundary

    def __BoundaryFinder(self,text):
        values = []; noteline = ""; count=0;
        last_prob=np.zeros(len(self.__labels_dic))
        for line in text:
            # Convert each line to vector
            linevector=self.__datatoVector(line)
            # Use pre-trained model to obtain the probabilities
            current_prob=self.__predictor.predict_log_proba(linevector)[0]
            preValues=current_prob+last_prob
            last_prob=preValues
            preValues = max(preValues) - preValues
            diff = np.zeros(len(self.__labels_dic))
            if (len(values) is not 0):
                diff = (preValues - values[len(values)-1])
            if (self.__ifboundary(diff)):
                break
            values.append(preValues)
            count += 1
        return count, self.__labels[preValues.tolist().index(0)]


    def get_Boundary_Position(self,notelines):
        BoundaryPositions=[]
        T=notelines
        while 1:
            index, label = self.__BoundaryFinder(T)
            T = T[index:]
            while index:
                BoundaryPositions.append(label)
                index=index-1
            if (T == []): break
        return BoundaryPositions

    def get_Seg_index(self,notelines):
        count = 0
        Segindex = []
        BoundaryPositions=self.get_Boundary_Position(notelines)
        lastele = BoundaryPositions[0]
        for ele in BoundaryPositions:
            if ele == lastele: count = count + 1
            else:
                Segindex.append([lastele, count])
                count = 1
                lastele = ele
        return Segindex

    def print_Segs(self,notelines):
        lines = notelines
        Segindex=self.get_Seg_index(notelines)
        for ele in Segindex:
            print("========" + self.__labels_dic.get(ele[0]) + "==========")
            for line in lines[:ele[1]]:
                print(line)
                print("\n")
            lines = lines[ele[1]:]
