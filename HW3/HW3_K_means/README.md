# Text Clustering

CS 584 Data Mining Project 3- Part 1. (March, 16, 2019)<br>
This is a team assignment that I work with Hung-Mao Chen.<br>

## Objectives
1. Implement the K-Means algorithm without using any library.
2. Deal with Text Data in Term-Document Sparse Matrix format
3. Design a distance function for Text Data
4. Handle the Curse of Dimensionality
5. Evaluate clustering solutions

## Data
Input data: consists of 8580 text records in sparse format. No labels are provided.

The format of this file is as follows:<br>
word/feature_id count word/feature_id count .... per line<br>
representing count of words in one document. Mapping of these features_ids to features are available in features.dat

## Implementations
#### Data:
1. Read feature and test data then revert the sparse matrix: read the features data test data. Combine them into a 2D array; therefore, the array size is 8580 * 126373.
2. Preprocess the data via TF-IDF: utilize the TfidfTransformer from sklearn to preprocess the data. By transforming the matrix to TF-IDF, features will be weighted, and the values of some features would be 0.
3. Perform SVD for dimension reduction: Since this is a sparse dataset, we reduce the dimension through SVD from sklearn.

### K-means clustering:
1. Initial centroids: select 7 data points randomly by applying k-means++ algorithm to avoid empty cluster issue.
2. We computed cosine similarity, correlation, and Chebyshev distance between each point and centroid, then assign the cluster with the centroid that is closest to the data point. We observed that measuring the correlation might be a better option for this dataset.
3. After assign cluster to each data point, recompute and update the centroids then repeat 1 and 2 steps.
4. Set the maximum iteration of updating centroids (running a and b) as 300.
5. Calculate the silhouette score and SSE.
6. Repeat 1 to 5 steps for 10 times, and return the best clustering result with the smallest SSE and highest silhouette score


# Result and Conclusions:
Please refer to the report.

## Rank and accuracy score on leaderboard
7<br>
0.67
