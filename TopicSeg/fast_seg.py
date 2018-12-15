import re
import numpy as np

class nbsegmenter(object):
    def __init__(self,
                 model='../Models/model_nbbow.out',
                 topics=['History','Labs','Medications','PhysicalExam','Courses'],
                 labels=['A', 'B', 'C', 'D', 'E'],
                 seged='../Output/segmented.txt'):
        self.__model=model
        self.__scores={}
        self.__topics=topics
        self.__topics_len=len(topics)
        self.__output=seged
        self.__labels=labels

    def __loadnbmodel(self):
        f = open(self.__model)
        for line in f:
            word, counts = line.strip().rsplit('\t', 1)
            self.__scores[word] = eval(counts)
        f.close()

    def __BoundaryFinder(self,text):
        values = []; noteline = ""; count = 0;
        for line in text:
            noteline += line + " " + "\n"
            words = re.split(' ', noteline)
            preValues = np.zeros(self.__topics_len)
            pre = np.zeros(self.__topics_len)
            for word in words:
                if word in self.__scores.keys():
                    preValues+=np.asarray(self.__scores[word])
            preValues=max(preValues)-preValues
            if (len(values) is not 0):
                pre = preValues - values[len(values)-1]
            if ((max(pre) + min(pre)) < 0):
                break
            values.append(preValues)
            count += 1
        return count, self.__topics[preValues.tolist().index(0)]

    def get_Boundary_Position(self,note):
        self.__loadnbmodel()
        BoundaryPostion=[]
        T=note
        while 1:
            index, label = self.__BoundaryFinder(T)
            T = T[index:]
            while index:
                BoundaryPostion.append(self.__labels[self.__topics.index(label)])
                index=index-1
            if (T == []):
                break
        return BoundaryPostion

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
            print("========" + self.__topics [self.__labels.index(ele[0])] + "==========")
            for line in lines[:ele[1]]:
                print(line)
                print("\n")
            lines = lines[ele[1]:]
