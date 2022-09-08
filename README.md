# NLP--POS-parts-of-speech-tagging-
A type of NLP(natural language processing) done using POS. Using Bayesian nets. We first train the program on an input data set and then test its accuracy afterwards.

Key files and how to run the program

tagger.py - This file contains the POS(parts of speech tagger) implementaion
data/train-public.txt - this txt file contains large texts with the POS tags for each words to train our model 
data/train-public.ind - This file contains the start indices for each sentence of the training file above

data/text-public(small/lrage).txt - both these files are used to test how well our model tags the words after training.
data/text-public(small/lrage).ind - similarly this file contains the sentence start indices for the file above

grader.py - contains a scipt to test the program using the training files above.

