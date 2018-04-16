# Deep_treatment_recommender
Code for submitted PKDD paper  
We are continuously maintaining and updating this repository  

The codes include two different parts:
+ Synthetic patient record generation algorithm
+ RNNs for activity recommendation

We are not able to provide the authentic data as it is allowed by the protocol of our project.

Alternatively, we provided several synthetic desensitized patient records generated using the algorithm we proposed in this paper. Here is a summary of the data provided:
1. 100 synthetic intubation patient records
2. 1000 synthetic intubation patient records
3. 5000 synthetic intubation patient records

More synthetic patient records can be generated using the synthetic patient record generator we provided. The algorithm details can be found in Alg.1 in our paper and the implementation is described below. 


## Context-Aware Deep Treatment Recommendation Framework



## Synthetic Patient Record Generator
As described in the paper, there are two steps to generate the synthetic patient records:
1. Align the activity traces to acquire the alignment matrix
2. Fit the alignment matrix to the Multivariate Bernoulli distribution and generate random samples

For step.1, we deployed the PIMA algorithm, details can be found in 2017 ICDM workshop paper "Process-Oriented Iterative Multiple Alignment for Medical Process Mining". For easier use and visualization, we implemented a Java App. The App takes the activity traces as input and outputs the alignment matrix. Visualization of the alignment is also supported. 

For step.2, the code can be found in folder "data_augmentation". Please note that a piece of R code is called from the python code. Hence the R packages are required to be installed in advance. 
