# Boundary Detection by Determining the Difference of Classification Probabilities of Sequences: Topic Segmentation
-------------------------------------------------------------------------------------------------
Topic segmentation plays an important role in the process of information extraction. It could be used to automatically split large articles into small segments with desired topics. The accumulated confidence scores of each sequence from text, predicted by a specific classification model, show significant feature. Let's take a look at the following example.

## Example
The following is a short clinical notes which has been tokenized into 23 sequences. (This short clinical notes is quoted from I2B2- NLP Dataset. For detail information please visit: https://www.i2b2.org/) 
> 1). History of present illness \
2). This is a 54-year-old female with a history of cardiomyopathy, hypertension, diabetes…… \
3). The patient then reportedly went into VFib and was shocked once by EMS ……. \
4). She was intubated, received amiodarone and dopamine, as her BP ……\
5). In the ED, a portable chest x-ray revealed diffuse bilateral ……\
6). Pt was transferred to the ICU for further management. \
7). Of note, she was recently hospitalized at Somver Vasky University Of ……\
8). She was then asymptomatic at that time. \
9). A fistulogram and angioplasty of her right AV fistula was performed on……\
10). She has since received dialysis treatments with no complication.\
         -------------------------------------------------------------------------------------------------------\
11). Home medications\
12). At the time of admission include amitriptyline 25 mg p.o. bedtime \
13). enteric-coated aspirin 325 mg p.o. daily\
14). enalapril 20 mg p.o. b.i.d., \
15). Lasix 200 mg p.o. b.i.d., \
16). Losartan 50 mg p.o. daily, \
17). Toprol-XL 200 mg p.o. b.i.d., \
18). Advair Diskus 250/50 one puff inhaler b.i.d., \
19). insulin NPH 50 units q.a.m. subcu and 25 units q.p.m. subcu, \
20). insulin lispro 18 units subcu at dinner time, \
21). Protonix 40 mg p.o. daily, \
22). sevelamer 1200 mg p.o. t.i.d., \
23). tramadol 25 mg p.o. q.6 h. p.r.n. pain.

Suppose we want to split a clinical note into five categories: History, Medications, Hospital Course, Laboratories and Physical Examinations, which are the most well-known topics in Electrical Medical Records (EMR) that is the systematized collection of patient and population electronically-stored health information in a digital format. We applied a pre-trained classifier for predicting confidence scores (the probabilities of belonging to each topic) of each sequence and assigning it with a score vector ![scorevector](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cfn_phv%20v%3D%5B%7Bscore%7D_%7Bhistory%7D%2Cscore_%7Blabs%7D%2Cscore_%7Bmeds%7D%2Cscore_%7BPE%7D%2Cscore_%7Bcourse%7D%5D).
After obtaining all the score vectors from all sequences, we could calculate the accumulated confidence score of the location of current sequence by simply adding current sequence’s score with its all previous sequences’ score. \
Formulaically：\
	Pre-trained classifier models respectively assign ![sequence](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cfn_phv%20s_1%2Cs_2%2Cs_3%2C......%2Cs_t) with a 5-dimensional vector ![scorevector](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cfn_phv%20v%3D%5B%7Bscore%7D_%7Bhistory%7D%2Cscore_%7Blabs%7D%2Cscore_%7Bmeds%7D%2Cscore_%7BPE%7D%2Cscore_%7Bcourse%7D%5D) and obtain a t-dimensional accumulated score vector ![scorevector](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cfn_phv%20%5Cmathbf%7B%5Crho%7D%20%3D%5Bas_1%2Cas_2%2C...%2Cas_t%5D). Each element in vector ρ could be obtained by using: ![vectorp](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cfn_phv%20as_t%3Dv_1&plus;v_2&plus;...&plus;v_t).\
Where:
![C_k](https://latex.codecogs.com/gif.latex?C_k): class ![k](https://latex.codecogs.com/gif.latex?k%20%5Csubset%20%5Cleft%20%5C%7B%20history%2C%20labs%2C%20medications%2C%20physical%20exams%2C%20hospital%20courses%20%5Cright%20%5C%7D)\
![s_i](https://latex.codecogs.com/gif.latex?s_i): the ![i^th](https://latex.codecogs.com/gif.latex?i%5E%7Bth%7D) tokenized sentence or sequence s;

The following two pictures show the accumulated score (![as](https://latex.codecogs.com/gif.latex?%5Cboldsymbol%7B%5Cmathbf%7B%5Cmathit%7Bas%7D%7D%7D)) of probability in vector ![scorevector](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cfn_phv%20%5Cmathbf%7B%5Crho%7D%20%3D%5Bas_1%2Cas_2%2C...%2Cas_t%5D) obtained using the classifiers based on NB (left) and SVM (right) models. Vertical axis represents the score of the vector 𝐯 while horizontal axis refer to the location of each sequence.\
![nb-svm1](https://raw.githubusercontent.com/patrick013/TopicSeg/master/Images/nb-svm1.png)


Let's take a close look at the variety of these five scores through initializing each accumulated score vector ![as](https://latex.codecogs.com/gif.latex?%5Cboldsymbol%7B%5Cmathbf%7B%5Cmathit%7Bas%7D%7D%7D) by being subtracted by the maximum value in the vector:\
![as_initial](https://latex.codecogs.com/gif.latex?%5Cmathbf%7Bas_i%5E%27%7D%3Dmax%28%5Cmathbf%7Bas_i%7D%29-%5Cmathbf%7Bas_i%7D)\
A new vector ![pho_initial](https://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Crho_%7Binitial%7D%7D%3D%5B%5Cmathbf%7Bas_1%5E%27%7D%2C%5Cmathbf%7Bas_2%5E%27%7D%2C...%2C%5Cmathbf%7Bas_t%5E%27%7D%5D) would be obtained. The following shows ![phoinitial](https://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Crho_%7Binitial%7D%7D) based on on NB (left) and SVM (right) models.\
![nb1](https://raw.githubusercontent.com/patrick013/TopicSeg/master/Images/nb2.png)
![svm1](https://raw.githubusercontent.com/patrick013/TopicSeg/master/Images/svm2.png)

Then take a backward difference of the former formula. As for the vector ρ_initial, it could be the following:\
![pho_initial_](https://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Crho_%7Binitial%7D%7D%5E%7B%27%7D%3D%5Cleft%20%5Clfloor%20%5Cmathbf%7Bas_2%5E%27%7D%20-%5Cmathbf%7Bas_1%5E%27%7D%2C%5Cmathbf%7Bas_3%5E%27%7D-%5Cmathbf%7Bas_2%5E%27%7D%2C...%2C%20%5Cmathbf%7Bas_t%5E%27%7D-%5Cmathbf%7Bas_%7Bt-1%7D%7D%5E%27%20%5Cright%20%5Crfloor)\
Finally, it could be seen as follow:\
![nb1](https://raw.githubusercontent.com/patrick013/TopicSeg/master/Images/nb3.png)
![svm1](https://raw.githubusercontent.com/patrick013/TopicSeg/master/Images/svm3.png)\
It is clear to see that there is a sharp change at the postion of sequence 10 where is the ground truth boundary.

## Algorithm

### Topic Score Predictor
Topic score predictor could give each sequence or sentence scores five score each of which is the probability of belonging to corresponding topic. Finally, the segmenter could predict tokenized sequences’ (![s_i](https://latex.codecogs.com/gif.latex?s_i)) probability ![P_(C_k )(s_i)](https://latex.codecogs.com/gif.latex?P_%7BC_k%7D%28s_i%20%29) of belonging to each category ![C_k](https://latex.codecogs.com/gif.latex?C_k)  for boundary detection by determining the difference of ![P_(C_k )(s_i)](https://latex.codecogs.com/gif.latex?P_%7BC_k%7D%28s_i%20%29) with respect to ![s_i](https://latex.codecogs.com/gif.latex?s_i), while ![s_i](https://latex.codecogs.com/gif.latex?s_i)   represents ![i^th](https://latex.codecogs.com/gif.latex?i%5E%7Bth%7D) tokenized sequences. \
In this project, Naive Bayes and Linear SVM models with features of BOW are mainly emploied for training Topic Score Predictor. Other types of classification models might also work.

### Boundary Detection
> 1. Tokenize text ![T](https://latex.codecogs.com/gif.latex?T) into n (number of tokenized sentences) sentences ![s_i](https://latex.codecogs.com/gif.latex?s_i);
> 2. Let ![t=1](https://latex.codecogs.com/gif.latex?t%3D1);
> 3. Topic score pretictor respectively assigns each sequence score and obtains vector ![scorevector](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cfn_phv%20%5Cmathbf%7B%5Crho%7D%20%3D%5Bas_1%2Cas_2%2C...%2Cas_t%5D)
> 4. Analyzing vector ![pho](https://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Crho%20%7D), \
If return 0, let ![t=t+1](https://latex.codecogs.com/gif.latex?t%3Dt&plus;1), \
If ![t>i](https://latex.codecogs.com/gif.latex?t%3Ei), ![sehment](https://latex.codecogs.com/gif.latex?s_1&plus;...&plus;s_%7Bt-1%7D) would be a segment;\
Else, go back to step 3;\
If return 1 and ![topic_index](https://latex.codecogs.com/gif.latex?topic_%7Bindex%7D), ![boundary_index=t-1](https://latex.codecogs.com/gif.latex?boundary_%7Bindex%7D%3Dt-1). In other words, ![s_1+⋯+s_(boundary_index)](https://latex.codecogs.com/gif.latex?s_1&plus;...&plus;s_%7Bboundary_%7Bindex%7D%7D)  is a segment which discusses about ![topic_index](https://latex.codecogs.com/gif.latex?topic_%7Bindex%7D). Simultaneously, let ![T=s_t+⋯+s_i](https://latex.codecogs.com/gif.latex?T%3Ds_t&plus;...&plus;s_i) and go back to step 1 to segment the rest text;
> 5. Segmentation finished. 

##### Analyzing vector ![pho](https://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Crho%20%7D)

The idea of analyzing idea ![pho](https://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Crho%20%7D) is to detect the variety of each topic score with the detected sequences location changing for boundary detection.\
Based on the example above, it is obvious to see that analyzing the vector ![corevector](https://latex.codecogs.com/gif.latex?%5Cmathbf%7Bas_t%27-as_%7Bt-1%7D%27%7D): \
>![analysevector](https://raw.githubusercontent.com/patrick013/TopicSeg/master/Images/analysep.png)\
The best threshold currently tested is 0.3 for NB-based topic score predictor and for SVM-based predictor. 

## Usage
Class *TopicSeg.topic_seg.segmenter(predictor_model='nb', dataset_file="../Datasets/LabeledDataset.txt",labels_dic, threshold=0.3)*

> ### Parameters:
> * **predictor_model**: String, optional(default='nb')
>   * Classification model for Topic Score Predictor
> * **dataset_file**: String, optional(dafault="../Datasets/LabeledDataset.txt")     
>   * Directory of the labelled dataset. This parameter should be changed for different segmentation tasks. The default directory here is used to achieve topic segmentation of clinical notes.
> * **labels_dic**: dictionary, optional(default={'A':'History','B':'Labs','C':'Medications','D':'PhysicalExam','E':'Courses'})
>   * Labels used in the dataset. This parameter should be changed for different segmentation tasks. The default directory here is used to achieve topic segmentation of clinical notes.
> * **threshold**:float, optional(default=0.3,range=[0,1])
>   * Threshold of analyzing vector ![pho](https://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Crho%20%7D). 

> ### Methods
> * **get_Boundary_Position(notelines)**: Get all the boundaries and corresponding topic labels sequence.
>   * Parameter: notelines: the list of text to be segmented.
>   * Return: List of detected boundaries and topic labels.
> * **get_Seg_index(notelines)**: Get all the boundaries' index.
>   * Parameter: notelines: the list of text to be segmented.
>   * Return: List of boundaries' index.
> * **print_Segs(notelines)**: Print out the segmented text.
>   * Parameter: notelines: the list of text to be segmented.

### Example
```python
>>> from TopicSeg.topic_seg import segmenter
>>> mysegmenter=nbsegmenter()
>>> boudanry_postion=mysegmenter.get_Boundary_Position(notelines)
>>> print(boudanry_postion)
>>> ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'C', 'D', 'D', 'D', 'D', 'D', 'E', 'E', 'E', 'E']
>>> boudanry_index=mysegmenter.get_Seg_index(notelines)
>>> print(boudanry_index)
>>> [['A',8],['C',2],['D',5],['E',4]]
>>> mysegmenter.print_Segs(nontelines)
>>> ========History==========
>>> history of present illness ...
>>> ========Medications==========
>>> The medications ....
```



## Please Cite
> Ruan, Wei, and Won-sook Lee. "Boundary Detection by Determining the Difference of Classification Probabilities of Sequences: Topic Segmentation of Clinical Notes." 2018 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2018.
> Ruan, Wei, et al. "Pictorial Visualization of EMR Summary Interface and Medical Information Extraction of Clinical Notes." 2018 IEEE International Conference on Computational Intelligence and Virtual Environments for Measurement Systems and Applications (CIVEMSA). IEEE, 2018.





