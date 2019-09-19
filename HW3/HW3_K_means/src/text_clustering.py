import re
import pandas as pd
import numpy as np
import scipy
import math
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from copy import deepcopy
from random import randint
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_selection import mutual_info_classif
import random
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
import matplotlib.pyplot as plt


features_path = './features.data'
test_path = './test.data'
features = []
NUM_OF_FEATURES = 126373
NUM_OF_RECORDS = 8580
test_data = [[0]* NUM_OF_FEATURES for i in range(NUM_OF_RECORDS)]


def main():
    sse_res = []
    read_data()
    test_tfidf = tf_idf()
    test_svd = svd(test_tfidf)
    # evaluate K-means clusters with K = 3 to 22
    # for K in range(3,22,1):
    #     sse_res.append(k_means(test_svd,K))
    # plot_sse(sse_res)
    
    k_means(test_svd,7)


def read_data():
    global features, test_data
    test_buf = []
    # read features data
    with open(features_path) as f:
        features = f.read().splitlines()
    
    with open(test_path) as f:
        data = f.read().splitlines()
        for row in range(NUM_OF_RECORDS):
            test_buf.append(data[row].split())
    for row in range(NUM_OF_RECORDS):
        l = len(test_buf[row])
        for col in range(0,l,2):
            feature_id = int(test_buf[row][col]) - 1
            count = int(test_buf[row][col+1])
            test_data[row][feature_id] = count
  
def tf_idf():
    tfidf_transformer = TfidfTransformer().fit(test_data)
    test_tfidf = tfidf_transformer.transform(test_data)
    return test_tfidf

def svd(test_tfidf):
    # In particular, truncated SVD works on term count/tf-idf matrices 
    # it is known as latent semantic analysis (LSA).
    # For LSA, a value of 100 is recommended.
    # 100 = 0.25064564; 500 = 0.48713001; 1000 = 0.62768052; 2000 = 0.77843753; 2500 = 0.82; 3000: 0.85
    svd = TruncatedSVD(n_components=3000, random_state=42)
    test_svd = svd.fit_transform(test_tfidf)
    print(svd.explained_variance_.cumsum())
    return test_svd
    # print(test_svd.shape)
    

def k_means_plus_plus(test_svd, K):
    c1 = randint(0,NUM_OF_RECORDS)
    C = [test_svd[c1]]
    for k in range(1, K):
        D2 = scipy.array([min([scipy.inner(c-x,c-x) for c in C]) for x in test_svd])
        probs = D2/D2.sum()
        cumprobs = probs.cumsum()
        r = scipy.rand()
        for j,p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        C.append(test_svd[i])
    return C

def k_means(test_svd,K):
    # store clusters for each record 
    itr = 0
    clusters = np.zeros((NUM_OF_RECORDS,10))
    scores = np.zeros(10)
    sse_res = np.zeros(10)
    while itr<10:
        print("k means "+ str(itr))
        # test_svd = normalize(test_svd)

        # test_svd.shape[1] = # features after reduction
        num_features = test_svd.shape[1]
        centers_old = np.zeros((K,num_features))
        # random K centers
        rand_points = k_means_plus_plus(test_svd,K)
        # print(rand_points)
        centers_new = np.array(rand_points)
        
    
        # store distance between each record and centers
        distance = np.zeros((NUM_OF_RECORDS,K))
        # print(centers_new[0])
        # check if there's any center was changed
        # error = np.linalg.norm(centers_new - centers_old)
        j = 0
        while j<300:
            for row in range(NUM_OF_RECORDS):
                for i in range(K):
                    # compute distance between each record and center 
                    distance[row][i] = get_correlation(test_svd[row],centers_new[i])

            # assign the points to the closest cluster
            # argmax: get the index of maximum cosine similarity
            clusters[:,itr] = np.argmax(distance, axis = 1)
            # get the closest chebyshev_dist
            # clusters[:,itr] = np.argmin(distance, axis = 1)
            centers_old = deepcopy(centers_new)
            # recompute the new k mean centers
            for i in range(K):
                centers_new[i] = np.mean(test_svd[clusters[:,itr] == i],axis =0)
            # print(centers_new)
            centers_new = check_empty_cluster(test_svd,clusters[:,itr],centers_new,K)
            j+=1 
        scores[itr] = sil_score(test_svd,clusters[:,itr])
        c, sse = compute_sse(test_svd,clusters[:,itr],centers_new,K)
        sse_res[itr] = sse
        print("score " + str(itr) + ": " + str(scores[itr]) )
        print("sse " + str(itr) + ": " + str(sse_res[itr]) )
        itr+=1
    # get the highest  silhouettescore 
    best_clusters1 = clusters[:, np.argmax(scores)]
    print("itr: "+ str(np.argmax(scores)) + " has the maximum silhouette score = " + str(np.amax(scores)))
    # get the minimum sse
    best_clusters2 = clusters[:,np.argmin(sse_res)]
    print("itr: "+ str(np.argmin(sse_res)) + " has the minimum sse  = " + str(np.amin(sse_res)))

    with open('output_sil.txt', 'w') as f:
        for i in range(best_clusters1.shape[0]):
            f.write("%s\n" % str(int(best_clusters1[i])+1))

    with open('output_sse.txt', 'w') as f:
        for i in range(best_clusters2.shape[0]):
            f.write("%s\n" % str(int(best_clusters2[i])+1))
    return np.amin(sse_res)
    
   

def compute_sse(test_svd, clusters, centroids,K):
        distance = np.zeros(test_svd.shape[0])
        for k in range(K):
            distance[clusters == k] = np.linalg.norm(test_svd[clusters == k] - centroids[k], axis=1)  
        distance = np.square(distance)
        sse = np.sum(distance)
        return np.argmax(distance), sse

def check_empty_cluster(test_svd,clusters,centers_new,K):
    # print(clusters)
    for k in range(K):
        if k in clusters[:]:
           pass
        else:
            c, sse = compute_sse(test_svd,clusters,centers_new,K)
            centers_new [k] = random.choice(test_svd[clusters == c])
            print("empty cluster: "+ str(k))
            print(centers_new [k])
    return centers_new

def get_correlation(v1,v2):
    corr = np.corrcoef(v1, v2)
    return corr[0][1]

def get_cosine_sim(v1, v2):
    
    return np.dot(v1,v2)/(np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))

def get_chebyshev_dist(v1,v2):
    return distance.chebyshev(v1,v2)

def sil_score(data, predicted):
    # score between -1 and 1, the bigger the better
    # score = silhouette_score(data, predicted, metric='cosine')
    score = silhouette_score(data, predicted, metric='correlation')
    # score = silhouette_score(data, predicted, metric='chebyshev')
    return score

def plot_sse(sse_res):
    plt.figure()
    filename ="sse_res"
    x = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    plt.plot(x, sse_res)
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.xlabel("K")
    plt.ylabel("SSE")
    plt.savefig(filename)




main()



    