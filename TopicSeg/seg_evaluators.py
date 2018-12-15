
import segeval
from decimal import *
import numpy as np


'''
This script aims to do evaluation of topic segmentation;
For evaluating the segmentation algorithms, the metric of traditional windows-based measurement P_k and Windiff are applied in this script;
In addition, Boundary similarity and Segmentation similarity proposed by Fournier are also used
For more information, please visit the following link:
https://segeval.readthedocs.io/en/latest/api/#
Precision, recall and F score are also important for assessing the algorithm.
'''
# The format of the reference and hypothesis sequences could be the following:
# string-based: ['a','a','b','b','b','b','c','c','c']
# int: [1,1,2,2,2,2,3,3,]

def __scorecount(reference,hypothesis,label):
    true_positives=0; true_negatives=0; false_positives=0; false_negatives=0;
    new_reference=[1 if ref==label else 0 for ref in reference]
    new_hypothesis=[1 if hyp==label else 0 for hyp in hypothesis]
    # Counting true positives, true negatives, false positives and false negatives
    for (ref,hyp) in zip(new_reference,new_hypothesis):
        if ref==hyp:
            if ref==1: true_positives+=1
            else: true_negatives+=1
        else:
            if ref==1:false_negatives+=1
            else: false_positives+=1
    return true_positives,true_negatives,false_positives,false_negatives

def __getscores(reference,hypothesis):
    # Check if two sequences have the same format and length for further process
    if (len(reference)!=len(hypothesis)):
        print("Error! The length of hypothesis doesn't match the length of reference!")
        raise SystemExit
    labels_set=set(reference)
    true_positives = 0; true_negatives = 0; false_positives = 0; false_negatives = 0
    # Calculating each labels' precision, recall and F score, return average values finally.
    for each_label in labels_set:
        tp,tn,fp,fn=__scorecount(reference,hypothesis,each_label)
        true_positives+=tp
        true_negatives+=tn
        false_positives+=fp
        false_negatives+=fn
    # Calculating average precision, recall and F-1 score.
    precision=true_positives/(true_positives+false_positives)
    recall=true_positives/(true_positives+false_negatives)
    F_1=2*precision*recall/(precision+recall)
    return precision,recall,F_1

def __initialization(reference,hypothesis):
    if (len(reference)!=len(hypothesis)):
        print("Error! The length of hypothesis doesn't match the length of reference!")
        raise SystemExit
    # Initializing the format of the reference and hypothesis sequences for feeding in the SegEval
    reference_boundary=segeval.convert_positions_to_masses(reference)
    hypothesis_boundary=segeval.convert_positions_to_masses(hypothesis)
    return reference_boundary,hypothesis_boundary

def get_Pk_score(reference,hypothesis):
    ref,hyp=__initialization(reference,hypothesis)
    # Evaluate algorithm using pk metric
    return segeval.pk(ref,hyp)

def get_Windiff_socre(reference,hypothesis):
    ref, hyp = __initialization(reference, hypothesis)
    # Evaluate algorithm using window diff metric
    return segeval.window_diff(ref,hyp)

def get_Boundary_similarity(reference,hypothesis):
    ref, hyp = __initialization(reference, hypothesis)
    # Evaluate algorithm using B (boundary similarity) metric
    return segeval.boundary_similarity(ref,hyp)

def get_Segmentation_similarity(reference,hypothesis):
    ref, hyp = __initialization(reference, hypothesis)
    # Evaluate algorithm using S (segmentation similarity) metric
    return segeval.segmentation_similarity(ref,hyp)

def get_F1_score(reference,hypothesis):
    # Evaluate algorithm using F1 metric
    return __getscores(reference,hypothesis)[2]

def evaluateSegments(reference,hypothesis):
    ref, hyp = __initialization(reference, hypothesis)
    score=np.array([__getscores(reference,hypothesis)[2],\
           float(segeval.pk(ref, hyp)),\
           float(segeval.window_diff(ref, hyp)),\
           float(segeval.boundary_similarity(ref, hyp)),\
           float(segeval.segmentation_similarity(ref, hyp))])
    # Return pk, windiff, boundary_sim, segmentation_sim and F_1 score.
    return score



