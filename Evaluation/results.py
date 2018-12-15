import re
from TopicSeg.topic_seg import segmenter
from TopicSeg import seg_evaluators
import numpy as np

def initial_test_data(textlines):
    boundary_position=[]; document_sentences=[]
    for line in textlines:
        label, sentence=re.split(" +", line,1)
        boundary_position.append(label)
        document_sentences.append(sentence)
    return boundary_position,document_sentences

class document(object):

    def __init__(self,files_dir,seger):
        self.__reference_boundary,self.__document=initial_test_data([line.strip() for line in open(files_dir)])
        self.__predict_boudanry=seger.get_Boundary_Position(self.__document)

    def get_this_doc_score(self):
        return seg_evaluators.evaluateSegments(self.__reference_boundary,self.__predict_boudanry)

if __name__=="__main__":
    initial_test_data([line.strip() for line in open('../Datasets/Eval_dataset/1.txt')])
    test_dataset_size=50
    i=1;seger = segmenter(predictor_model='svm')
    score_list=[]
    while i<test_dataset_size+1:
        doc_file="../Datasets/Eval_dataset/"+str(i)+'.txt'
        doc=document(doc_file,seger)
        score=doc.get_this_doc_score()
        print(score);
        score_list.append(score); i+=1
    score_matrix=np.asarray(score_list)
    print("The average F_1 score is: %f" % float(np.sum(score_matrix,axis=0)[0]/50))
    print("The average pk score is: %f" % float(np.sum(score_matrix,axis=0)[1]/ 50))
    print("The average windiff score is: %f" % float(np.sum(score_matrix,axis=0)[2]/ 50))
    print("The average Bs score is: %f" % float(np.sum(score_matrix,axis=0)[3] / 50))
    print("The average Ss score is: %f" % float(np.sum(score_matrix,axis=0)[4]/ 50))


'''
The following is the nb-log-based results score obtained by testing on 50 clinical notes:
The average F_1 score is: 0.918194
The average pk score is: 0.091042
The average windiff score is: 0.112194
The average Bs score is: 0.653154
The average Ss score is: 0.971656
'''
# score_matrix = np.asarray([[1., 0., 0., 1., 1.],
#                           [0.84482759, 0.05769231, 0.13461538, 0.7, 0.97368421],
#                           [0.90625, 0.11320755, 0.20754717, 0.66666667, 0.98412698],
#                           [0.92307692, 0.09090909, 0.12727273, 0.64285714, 0.95689655],
#                           [0.98684211, 0.01428571, 0.01428571, 0.9, 0.99333333],
#                           [1., 0., 0., 1., 1.],
#                           [0.98571429, 0.03125, 0.03125, 0.9, 0.99275362],
#                           [0.95744681, 0.06818182, 0.07954545, 0.64285714, 0.97311828],
#                           [0.91891892, 0.11111111, 0.11111111, 0.66666667, 0.94736842],
#                           [0.96774194, 0.03529412, 0.04705882, 0.58333333, 0.97282609],
#                           [0.9787234, 0.02325581, 0.02325581, 0.83333333, 0.98924731],
#                           [0.98630137, 0.01515152, 0.01515152, 0.875, 0.99305556],
#                           [0.93548387, 0.10526316, 0.14035088, 0.5, 0.95081967],
#                           [0.94318182, 0.04878049, 0.09756098, 0.5, 0.95402299],
#                           [0.96703297, 0.03529412, 0.04705882, 0.64285714, 0.97222222],
#                           [0.984375, 0.01724138, 0.01724138, 0.875, 0.99206349],
#                           [0.82258065, 0.10344828, 0.13793103, 0.57142857, 0.95081967],
#                           [0.94117647, 0.06349206, 0.07936508, 0.75, 0.97761194],
#                           [0.71428571, 0.11842105, 0.21052632, 0.5, 0.97590361],
#                           [0.96875, 0.05084746, 0.06779661, 0.83333333, 0.98412698],
#                           [0.94680851, 0.10344828, 0.10344828, 0.66666667, 0.97849462],
#                           [0.88607595, 0.08108108, 0.13513514, 0.57142857, 0.96153846],
#                           [0.98529412, 0.01612903, 0.03225806, 0.9, 0.99253731],
#                           [0.96491228, 0.03773585, 0.05660377, 0.78571429, 0.97321429],
#                           [0.54347826, 0.39130435, 0.39130435, 0., 0.97802198],
#                           [0.91052632, 0.14772727, 0.17045455, 0.5, 0.97883598],
#                           [0.95901639, 0.05309735, 0.08849558, 0.64285714, 0.97933884],
#                           [0.91099476, 0.14689266, 0.16949153, 0.5, 0.97894737],
#                           [0.99295775, 0.00746269, 0.01492537, 0.9375, 0.9964539],
#                           [0.97033898, 0.09333333, 0.15111111, 0.36363636, 0.97021277],
#                           [0.81012658, 0.17073171, 0.19512195, 0.41666667, 0.92045455],
#                           [0.8, 0.16129032, 0.19354839, 0.5, 0.94117647],
#                           [0.8, 0.32911392, 0.32911392, 0.2, 0.90243902],
#                           [0.89795918, 0.02325581, 0.1627907, 0.5, 0.96875],
#                           [0.94444444, 0.15686275, 0.15686275, 0.57142857, 0.94339623],
#                           [0.97916667, 0.02325581, 0.04651163, 0.875, 0.9893617],
#                           [1., 0., 0., 1., 1.],
#                           [0.73214286, 0.27118644, 0.28813559, 0.35714286, 0.92741935],
#                           [0.97478992, 0.16363636, 0.16363636, 0.5, 0.96666667],
#                           [1., 0., 0., 1., 1.],
#                           [1., 0., 0., 1., 1.],
#                           [0.77586207, 0.27118644, 0.27118644, 0.2, 0.93846154],
#                           [0.94285714, 0.03125, 0.078125, 0.6, 0.97101449],
#                           [0.97727273, 0.02439024, 0.04878049, 0.83333333, 0.98850575],
#                           [0.92156863, 0.17894737, 0.17894737, 0.35714286, 0.95544554],
#                           [0.9382716, 0.08108108, 0.09459459, 0.58333333, 0.96875],
#                           [0.85294118, 0.15625, 0.171875, 0.5625, 0.94776119],
#                           [0.78787879, 0.26229508, 0.26229508, 0.5, 0.94615385],
#                           [0.98666667, 0.02898551, 0.02898551, 0.9, 0.99324324],
#                           [0.98461538, 0.03703704, 0.03703704, 0.75, 0.9921875]])



