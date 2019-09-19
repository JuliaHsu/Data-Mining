# Drug Activity Prediction

CS 584 Data Mining Project 2- Part 1. (Feb., 12, 2019)
This is a team assignment that I work with Hung-Mao Chen.


## Project Description
Develop predictive models that can determine given a particular compound whether it is active (1) or not (0).
In this project, we implement feature selection techniques and experiment with various classification models. Furthermore, the data set is inbalanced; therefore, we use different techniques, such as oversampling, undersampling and SMOTE to balnce the data. 

## Data Description
The training dataset consists of 800 records and the test dataset consists of 350 records.
The attributes are binary type and as such are presented in a sparse matrix format within train.dat and test.dat.

### train_drugs.data: Training set (a sparse binary matrix, patterns in lines, features in columns: the number of the non-zero features is provided with class label 1 or 0 in the first column.

### test.data: Testing set (a sparse binary matrix, patterns in lines, features in columns: the number of non-zero features is provided).


## Dealing with data and implementations

1. Read document and revert the sparse binary matrix.
2.	Perform PCA for dimension reduction.
3. Over sampling to balance the training data by Synthetic Minority Over-sampling (SMOTE): use the imblearn.over_sampling.SMOTE to perform SMOTE.
4.	Build several classification models and use cross validation to find the best parameters for the data. Including SVM, decision tree, Naïve Bayes, Adaboost, Random Forest classifiers.
5.	Classifier Evaluation: Use f1 score to evaluate the models we built. The result of cross validation shows that Random Forest classifier is one of the best classifiers for this dataset.

## Conclusion
In this project we first learned how to read and convert the sparse matrix. We also noticed that the dimension of this data set is high, all data are sparse and dissimilar; thus, we used PCA that we learned in class to reduce the dimenssion.
In addition to the curse of dimensionality, the data has a problem of imbalnced, which can be solved by several techniques such as undersampling, oversampling, and SMOTE. After conducting several experiments, we chose SMOTE to generate new data to balnce the data set.
This project includes multiple concepts of data mining that we learned in class, those techniques help us build a good model to predict as accurate as possible. 

## Rank and accuracy score on leaderboard
11
0.74
