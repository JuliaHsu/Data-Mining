# True and Deceptive Hotel Reviews

CS 584 Data Mining Project 1. (Feb., 7, 2019)<br>
The main goal of this project is to predict the class (truthful or deceptive)) of hotel reviews by implement the k Nearest Neighbor Algorithm.

## Data
Training data consists of 1,600 reviews and exists in the train data file. Each file represents a review and the folder where it is contained (truthful or deceptive) represents its class.<br>
Testing data consists of 160 hotel reviews provided in the test file.

## Dealing with data

1. Read data in sorted order, and label the training data.
2. Convert training document to a matrix of TF-IDF features by using sklearn.feature_extraction.text.TfidfVectorizer. Stop words can also be removed.

## Implementations
1. Evaluate the similarity between each testing document and training document by computing the cosine distance of vectors.
2. Use “nlargest” function to choose k nearest points to determine the unknown point by majority vote. To improve the accuracy, I consider the “weight” between each point. The more similar point has a higher weight. 
3. Validation: Use cross validation to find the best K.

## Conclusion
The k nearest neighbor classification was introduced in this assignment, which provided me a better view and concept on classification and data mining. I have learned how to deal with text document and create tf-idf; in addition, the cross validation allows me to test my algorithms and tune the parameter, which is very helpful.<br>
(This project was assigned at the first class; mainly to help students have a better overview on Data Mining.) 

## Rank and accuracy score on leaderboard
10<br>
0.86
