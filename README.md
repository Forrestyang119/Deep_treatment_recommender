# Deep_treatment_recommender
Code for submitted PKDD paper  
We are continuously maintaining and updating this repository  

The codes include two different parts:
+ RNNs for activity recommendation
+ Synthetic patient record generation algorithm

We are not able to provide the authentic data as it is not allowed by the protocol of our project.

Alternatively, we provided the synthetic desensitized patient records generated using the algorithm we proposed in this paper. Here is a summary of the data provided:
+ 1000 synthetic intubation patient records (1000 activity traces with patient attributes)

More synthetic patient records can be generated using the synthetic patient record generator we provided. The algorithm details can be found in Alg.1 in our paper and the implementation is described below. 

## Summary of Our Methods:
### Context-Aware Deep Treatment Recommendation Framework
![alt text](Docs/Fig1.png "Fig.1")
The recommender system (Fig. 1) built on an RNN. The RNN takes as input the concatenation of the activity embedding vectors $v^a$ (main input) and the activity attribute vectors $v^b$ (auxiliary input (optional), dynamic environmental context). The latent vector outputs from the RNN go through our attention layer and then merged with the patient attribute vector $v^x$ (auxiliary input, static environmental context).  For the final output, we used a densely connected layer after the merging layer followed by a top-k softmax activation function. The most probable k activities will be shown to the medical team as the recommended treatment for the next step (t+1). In practice, the dynamic contextual information will be updated by our sensor-based activity recognition system or by the nurse recorder who has access to the computerized decision support system.

### Synthetic Patient Record Generator
![alt text](Docs/Fig2.png "Fig.2")
As described in the paper, there are two steps to generate the synthetic patient records:
1. Align the activity traces to acquire the alignment matrix
2. Fit the alignment matrix to the Multivariate Bernoulli distribution and generate random samples

For step.1, we deployed the PIMA algorithm, details can be found in 2017 ICDM workshop paper "Process-Oriented Iterative Multiple Alignment for Medical Process Mining". For easier use and visualization, we implemented a Java App. The App takes the activity traces as input and outputs the alignment matrix. Visualization of the alignment is also supported. 

For step.2, the code can be found in folder "data_augmentation". Please note that a piece of R code is called from the python code. Hence the R packages are required to be installed in advance. 
