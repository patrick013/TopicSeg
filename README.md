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
23). tramadol 25 mg p.o. q.6 h. p.r.n. pain.\

Suppose we want to split a clinical note into five categories: History, Medications, Hospital Course, Laboratories and Physical Examinations, which are the most well-known topics in Electrical Medical Records (EMR) that is the systematized collection of patient and population electronically-stored health information in a digital format. We applied a pre-trained classifier for predicting confidence scores (the probabilities of belonging to each topic) of each sequence and assigning it with a score vector $$v=[{score}_{history},score_{labs},score_{meds},score_{PE},score_{course}]$$. After obtaining all the score vectors from all sequences, we could calculate the accumulated confidence score of the location of current sequence by simply adding current sequence’s score with its all previous sequences’ score. \
Formulaically：
	Pre-trained classifier models respectively assign $$s_1,s_2,s_3,……,s_t$$ with a 5-dimensional vector $$v=[{score}_{history},score_{labs},score_{meds},score_{PE},score_{course}]$$ and obtain a t-dimensional accumulated score vector $$ρ=[as_1,as_2,……,as_t]$$. Each element in vector ρ could be obtained by using: $$as_t=v_1+v_2+⋯+v_t$$
