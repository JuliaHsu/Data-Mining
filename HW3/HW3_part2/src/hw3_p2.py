import re
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import pylab as pl
from sklearn.cluster import DBSCAN
import scipy
from sklearn.metrics import silhouette_score



test1_path ="./test1.data"
test2_path ="./test2.data"
NUM_OF_FEATURES = 5
NUM_OF_RECORDS = 750
test_data1 = np.zeros((NUM_OF_RECORDS,NUM_OF_FEATURES))
test_data2 = np.zeros((NUM_OF_RECORDS,NUM_OF_FEATURES))
proximity_matrix1 = np.zeros((NUM_OF_RECORDS,NUM_OF_RECORDS))
proximity_matrix2 = np.zeros((NUM_OF_RECORDS,NUM_OF_RECORDS))


def read_data():
    global test_data1, test_data2
    test1_buf = []
    test2_buf = []
    with open(test1_path) as f:
        data = f.read().splitlines()
        for row in range(NUM_OF_RECORDS):
            test1_buf.append(data[row].split())
    test_data1 = np.array(test1_buf)
    test_data1 = test_data1.astype(np.float)
    with open(test2_path) as f:
        data = f.read().splitlines()
        for row in range(NUM_OF_RECORDS):
            test2_buf.append(data[row].split())
    test_data2 = np.array(test2_buf)
    test_data2 = test_data2.astype(np.float)

def pca():
    test1_norm = normalize(test_data1)
    test2_norm = normalize(test_data2)
    pca = PCA(n_components=2)
    test1_pca = pca.fit_transform(test1_norm)
    test2_pca = pca.fit_transform(test2_norm)

    return test1_pca, test2_pca


def k_means(test1,test2):
    # initialize 
    incidence_matrix1 = np.zeros((NUM_OF_RECORDS,NUM_OF_RECORDS))
    incidence_matrix2 = np.zeros((NUM_OF_RECORDS,NUM_OF_RECORDS))
    y1_kmeans = np.zeros((NUM_OF_RECORDS,4))
    y2_kmeans = np.zeros((NUM_OF_RECORDS,4))
    sse_res1 = []
    sse_res2 = []
    corr1 = []
    corr2 = []
    sil_score1 = []
    sil_score2 = []
    for k in range(2,6):
        kmeans1 = KMeans(n_clusters=k,random_state=42).fit(test1)
        y1_kmeans[:,k-2] = kmeans1.predict(test1)
        centroids1 = kmeans1.cluster_centers_
        sse_res1.append(compute_sse(test_data1,y1_kmeans[:,k-2],centroids1, k))

        
        kmeans2 = KMeans(n_clusters=k,random_state=42).fit(test2)
        y2_kmeans[:,k-2] = kmeans2.predict(test2)
        centroids2 = kmeans2.cluster_centers_
        sse_res2.append(compute_sse(test_data2,y2_kmeans[:,k-2],centroids2, k))

        # incidence matrix for test 1 and test 2 data
        incidence_matrix1 = get_incidence_matrix(y1_kmeans[:,k-2],incidence_matrix1)
        incidence_matrix2 = get_incidence_matrix(y2_kmeans[:,k-2],incidence_matrix2)
        # compute correlation via proimity matrix and incidence matrix 
        corr1.append(get_correlation(proximity_matrix1,incidence_matrix1))
        corr2.append(get_correlation(proximity_matrix2,incidence_matrix2))
        # compute silhouette coefficient via original data and predicted clusters
        sil_score1.append(sil_score(test_data1,y1_kmeans[:,k-2]))
        sil_score2.append(sil_score(test_data2,y2_kmeans[:,k-2]))

    plot_sse(sse_res1,"test_1","kmeans")
    plot_sse(sse_res2,"test_2","kmeans")

    
    visualize_clusters("kmeans","1", test1_pca,y1_kmeans, corr1,sil_score1)
    visualize_clusters("kmeans","2", test2_pca,y2_kmeans, corr2, sil_score2)

    return y1_kmeans,y2_kmeans

def EM(test1,test2):
    incidence_matrix1 = np.zeros((NUM_OF_RECORDS,NUM_OF_RECORDS))
    incidence_matrix2 = np.zeros((NUM_OF_RECORDS,NUM_OF_RECORDS))
    y1_em = np.zeros((NUM_OF_RECORDS,4))
    y2_em = np.zeros((NUM_OF_RECORDS,4))
    corr1 = []
    corr2 = []
    sil_score1 = []
    sil_score2 = []
    for k in range(2,6):
        y1_em[:,k-2] = GaussianMixture(n_components=k,random_state=42).fit_predict(test1)
        y2_em[:,k-2] = GaussianMixture(n_components=k,random_state=42).fit_predict(test2)

        # incidence matrix for test 1 and test 2 data
        incidence_matrix1 = get_incidence_matrix(y1_em[:,k-2],incidence_matrix1)
        incidence_matrix2 = get_incidence_matrix(y2_em[:,k-2],incidence_matrix2)
        # compute correlation via proimity matrix and incidence matrix 
        corr1.append(get_correlation(proximity_matrix1,incidence_matrix1))
        corr2.append(get_correlation(proximity_matrix2,incidence_matrix2))
        # compute silhouette coefficient via original data and predicted clusters
        sil_score1.append(sil_score(test_data1,y1_em[:,k-2]))
        sil_score2.append(sil_score(test_data2,y2_em[:,k-2]))

    visualize_clusters("EM","1", test1_pca,y1_em,corr1,sil_score1)
    visualize_clusters("EM","2", test2_pca,y2_em,corr2,sil_score2)

def DBScan(test1,test2):
    corr1 = 0.0
    corr2 =  0.0
    max_corr1 = 0.0
    max_corr2 = 0.0
    y1_dbscan = []
    y2_dbscan = []
    sil_score1 = 0.0
    sil_score2 = 0.0
    max_idx1 = 0
    max_idx2 = 0
    incidence_matrix1 = np.zeros((NUM_OF_RECORDS,NUM_OF_RECORDS))
    incidence_matrix2 = np.zeros((NUM_OF_RECORDS,NUM_OF_RECORDS))
    # grid search for best eps
    for i in range(1,11):
        
        y1_dbscan.append(DBSCAN(eps = i).fit_predict(test1))
        eps_2 = float(i/10.0)
        y2_dbscan.append(DBSCAN(eps = eps_2).fit_predict(test2))
        

        incidence_matrix1 = get_incidence_matrix(y1_dbscan[i-1],incidence_matrix1)
        incidence_matrix2 = get_incidence_matrix(y2_dbscan[i-1],incidence_matrix2)
        corr1 = get_correlation(proximity_matrix1,incidence_matrix1)
        corr2 = get_correlation(proximity_matrix2,incidence_matrix2)


        if abs(corr1) > max_corr1:
            max_corr1 = corr1
            max_idx1 = i-1
        if abs(corr2) > max_corr2:
            max_corr2 = corr2
            max_idx2 = i-1
    
    sil_score1 = sil_score(test_data1,y1_dbscan[max_idx1])
    sil_score2 = sil_score(test_data2,y2_dbscan[max_idx2])

    # figure for test 1
    plt.figure()
    plt.title("DBSCAN test 1 ")
     
    plt.text(1, 1.5, "eps = "+ str(max_idx1+1))
    plt.text(1, 1.6, "correlation = " + str(max_corr1))
    plt.text(1, 1.7, "silhouette = " + str(sil_score1))

    plt.scatter(test1_pca[:, 0], test1_pca[:, 1], c=y1_dbscan[max_idx1], cmap='viridis')
    plt.savefig("test_1_dbscan",bbox_inches='tight')
    # figure for test 2
    plt.figure()
    plt.title("DBSCAN test 2")
    plt.text(1, 1.5, "eps = "+ str((max_idx2+1)/10.0))
    plt.text(1, 1.6, "correlation = "+ str(max_corr2))
    plt.text(1, 1.7, "silhouette = " + str(sil_score2))
    plt.scatter(test2_pca[:, 0], test2_pca[:, 1], c=y2_dbscan[max_idx2], cmap='viridis')
    plt.savefig("test_2_dbscan",bbox_inches='tight')


def compute_sse(test, y, centroids,K):
        distance = np.zeros(test.shape[0])
        for k in range(K):
            distance[y == k] = np.linalg.norm(test[y == k] - centroids[k], axis=1)  
        
        sse = np.sum(np.square(distance))
        return sse


def visualize_clusters(alg,test_set,test,y, corr, sil_score):
    filename = alg
    plt.figure()
    for k in range(2,6):
        plt.title("test_" +  test_set + " "+ alg+ " k = "+ str(k))
        plt.scatter(test[:, 0], test[:, 1], c=y[:,k-2], cmap='viridis')
        txt1 = plt.text(1,1.5,"correlation = "+ str(corr[k-2]))
        txt2 = plt.text(1,1.6,"silhouette = "+ str(sil_score[k-2]))
        filename ="test_"+ test_set+ "_" + alg + str(k)

        plt.savefig(filename,bbox_inches='tight')
        txt1.remove()
        txt2.remove()

def plot_sse(sse_res,test_set,alg):
    plt.figure()
    filename = test_set + "_" + alg + "_sse"
    x = [2,3,4,5]
    plt.plot(x, sse_res)
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.xlabel("K")
    plt.ylabel("SSE")
    plt.savefig(filename)

def get_proximity_matrix(proximity_matrix,test_data):
    for i in range(NUM_OF_RECORDS):
        for j in range(NUM_OF_RECORDS):
            # distance between each point
            v1 = np.array(test_data[i])
            v2 = np.array(test_data[j])
            proximity_matrix[i][j] = np.linalg.norm(v1-v2)
    # print(proximity_matrix)
    return proximity_matrix

def get_incidence_matrix(y,incidence_matrix):
    for i in range(NUM_OF_RECORDS):
        for j in range(NUM_OF_RECORDS):
            if y[i] == y[j]:
                incidence_matrix[i][j] = 1
            else:
                incidence_matrix[i][j] = 0
    # print(incidence_matrix)
    return incidence_matrix

def get_correlation(proximity_matrix,incidence_matrix):
    cov12 =0.0
    var1 =0.0
    var2 =0.0
    mean_d = np.mean(proximity_matrix)
    
    mean_c = np.mean(incidence_matrix)

    
    for i in range(NUM_OF_RECORDS):
        for j in range(NUM_OF_RECORDS):
            
            std1 = (proximity_matrix[i][j] - mean_d)
            std2 = (incidence_matrix[i][j] - mean_c)
            cov12 +=  (std1*std2)

            var1 += std1*std1
            # print(var1)
            var2 += std2*std2
            # print(var2)
    
    corr = cov12/(np.sqrt(var1)*np.sqrt(var2))
    print(corr)
    return corr
def sil_score(data, predicted):
    
    score = silhouette_score(data, predicted, metric='euclidean')
    return score

   
            
    



read_data()
proximity_matrix1 = get_proximity_matrix(proximity_matrix1,test_data1)
proximity_matrix2 = get_proximity_matrix(proximity_matrix2,test_data2)
test1_pca, test2_pca = pca()
k_means(test_data1,test_data2)
EM(test_data1,test_data2)
DBScan(test_data1,test_data2)


