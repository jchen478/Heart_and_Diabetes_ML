# Heart_and_Diabetes_ML
Dataset set contains diabetes and heart disease data
*arff contains dataset in train and test set

1. dt-learn.py
- Contains decision tree callable from command line
  dt-learn <train-set-file> <test-set-file> m
- The stopping criteria: when m reaches the leaf node
- tested on heart and diabetes data
  
2. bayes.py
- Contains Naive Bayes (NB) and Tree-augmented naive Bayes (TAN)
- Callable from command line
  bayes <train-set-file> <test-set-file> <n|t>
- The last argument is a single character specifying NB or TAN network
- NB.py and TAN.py include implementation for the networks
- tested on lymphography data (lymph, vote)
  
  
  
  
