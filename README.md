# svm_project
Project in Molecular Life Science, protein structure predictor using svm.

Output
-
Model trained with single-sequence information
Model trained with multiple-sequence information

Input
-
Whole dataset
Reduced dataset

check_ws
-
Performs cross-validation with different window sizes

confusion_matrix
-
Performs cross-validation and creates a confusion matrix figure

create_model
-
Trains the final single-sequence information SVM

create_pssm_model
-
Trains the final multiple-sequence information SVM

models_crossval
-
SVM and Random forest functions, cross-validation

mp2
-
Model preparation, only functions. 

optimizeC
-
Check different C parameter values

poly
-
Polynomial kernel

predictor
-
Takes the model and predicts the topology of a fasta sequence 

pssm_crossval
-
Cross-validation for the multiple-sequence info SVM

pssm_prep
-
Creates frequency matrices from the PSSM

rbf
-
RBF kernel

reduce_dataset
-
Divides the globular sequences and TM sequences into two files, then takes one part TM and two parts globular

run_rf
-
Runs random forest 

wsgraph
-
Plots graphs
