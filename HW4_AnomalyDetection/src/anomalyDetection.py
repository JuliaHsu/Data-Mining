import glob
import numpy as np
from sklearn.preprocessing import StandardScaler,Normalizer,scale
from numpy import inf
from scipy.spatial import distance
from scipy.fftpack import fft
from sklearn.metrics import roc_auc_score,roc_curve,auc
import matplotlib.pyplot as plt
sample_path = "./base/"

true_class=[]

def read_sample():
    global true_class
    signals = []
    signals_test =[]
    baseline_dir = "./base/Mode"
    for c in "ABCD":
        baseline_dir = baseline_dir+c
        for i in range(0,200):
            with open(baseline_dir+"/File"+str(i)+".txt") as f:
                signal = f.readline()
                if signal!="\n":
                    signals.append(signal.rstrip().split('\t'))       
        baseline_dir = "./base/Mode"
    baseline = np.array(signals)
    baseline = baseline.astype(np.float)
    # print(signals)
    print(baseline.shape)
    # test_dir = "./base/Mode"
    
    # for c in "ABCD":
    #     test_dir = test_dir+c
    #     for i in range(0,25):
            
    #         with open(test_dir+"/File"+str(i)+".txt") as f:
    #             signal_test = f.readline()
    #             if signal_test!="\n":
    #                 signals_test.append(signal_test.rstrip().split('\t'))     
    #                 true_class.append(0)  
    #     test_dir = "./base/Mode"
    # test_dir = "./base/ModeM/"
    # for i in range(0,100):
    #     with open(test_dir+"/File"+str(i)+".txt") as f:
    #         signal_test = f.readline()
    #         if signal_test!="\n":
    #             signals_test.append(signal_test.rstrip().split('\t'))  
    #             true_class.append(1)     
    #     test_dir = "./base/ModeM"
    
    # private_test = np.array(signals_test)
    # private_test = private_test.astype(np.float)
    

    test_dir = "./TestWT/Data"
    for i in range(1,500):
        with open(test_dir+str(i)+".txt")as f:
            signal_test = f.readline()
            if signal_test!="\n":
                signals_test.append(signal_test.rstrip().split('\t')) 
        test_dir = "./TestWT/Data"
    test = np.array(signals_test)
    test = test.astype(np.float)
    print(test.shape)

    
    return baseline,test

def  fast_fourier_transform(baseline,test):
   
    normal_scaler = Normalizer().fit(baseline)
    baseline = normal_scaler.transform(baseline)
    test = normal_scaler.transform(test)
    
    baseline_fft = np.zeros(baseline.shape)
    test_fft = np.zeros(test.shape)
    for row in range (baseline.shape[0]):
        baseline_fft[row] = abs(np.fft.fft(baseline[row]))
    for row in range(test.shape[0]):
        test_fft[row] = abs(np.fft.fft(test[row]))
        
    
    print(baseline_fft)
    print(test_fft)

    return baseline_fft,test_fft


    
    
   

def get_k_dist(data1,data2,K):
    num_samples1 = data1.shape[0]
    num_samples2 = data2.shape[0]
    dist = np.zeros((num_samples1,num_samples2))

    k_dist = np.zeros(num_samples1)
    for q in range (num_samples1):
        for p in range(num_samples2):
            # dist[q][p] = distance.cosine(data1[q],data2[p])
            # dist[q][p] = distance.cityblock(data1[q],data2[p])
            
            dist[q][p] = distance.euclidean(data1[q],data2[p])
        
        d = np.sort(dist[q])
        if num_samples1 == num_samples2:
            k_dist[q] = d[K]
        else:
            k_dist[q] = d[K-1]
    print(dist)
    
    return k_dist, dist

def get_k_neighbors(data1,data2,dist,k_dist):
    num_samples1 = data1.shape[0]
    num_samples2 = data2.shape[0]
    k_neighbors = []
    for q in range (num_samples1):
        k_neighbors.append([])
        for p in range (num_samples2):
            if dist[q][p]<k_dist[q] and q!=p:
                k_neighbors[q].append(p)
    return k_neighbors



def get_lrd(data1,data2,k_neighbors,dist):
    num_samples1 = data1.shape[0]
    num_samples2 = data2.shape[0]

    reach_dist = np.zeros(num_samples1)
    lrd = []
    
    for q in range(num_samples1):
        nk = len(k_neighbors[q])
        rd = 0
        for p in k_neighbors[q]:
            rd += max(k_dist[p],dist[q][p])
            
        reach_dist[q] = rd
        lrd.append(nk/reach_dist[q])
   

    return lrd
    # print(reach_dist[1][1])

def get_lof(lrd_baseline,lrd,k_neighbors):
    #LOF
    lof = []
 

    for q in range(len(lrd)):
        lrd_sum = 0.0
        nk = len(k_neighbors[q])
        for p in k_neighbors[q]:
            lrd_sum += lrd_baseline[p]
        lof.append((lrd_sum/nk) / lrd[q])
    print(lof)
    
    return lof

def StrOUD(test,baseline,lof_baseline,lrd_baseline,k_neighbors,dist_test):
    lrd_test = get_lrd(test,baseline,k_neighbors,dist_test)
    lof_test = get_lof(lrd_baseline,lrd_test,k_neighbors)

    p_val =[]
    for i in range(len(lof_test)):
        b=0
        for j in range(len(lof_baseline)):
            if lof_baseline[j]>=lof_test[i]:
                b +=1
        p = (b+1)/(baseline.shape[0]+1)
        p_val.append(p)
    anomaly = []
    # prob = []
    # for i in range(len(p_val)):
    #     p = 1-p_val[i]
    #     prob.append(p)
    with open('output.txt', 'w') as f:
        for i in range(len(lof_test)):
            f.write("%s\n" % str(p_val[i]))
  
    for i in range(len(lof_test)):
        print(p_val[i])
        if p_val[i] <=0.05:
            anomaly.append(1)
        else:
            anomaly.append(0)

    return lrd_test,lof_test,p_val


baseline, test = read_sample()
baseline_fft, test_fft = fast_fourier_transform(baseline,test)
fpr_ls = []
tpr_ls = []
for k in range(8,9,1):
    k_dist,dist_baseline = get_k_dist(baseline_fft,baseline_fft,k)
    k_neighbors = get_k_neighbors(baseline_fft,baseline_fft,dist_baseline,k_dist)
    lrd_baseline = get_lrd(baseline_fft,baseline_fft,k_neighbors,dist_baseline)
    lof_baseline = get_lof(lrd_baseline,lrd_baseline,k_neighbors)

    k_dist_test, dist_test = get_k_dist(test_fft,baseline_fft,k)
    k_neighbors_test = get_k_neighbors(test_fft,baseline_fft,dist_test,k_dist_test)

    lrd_test,lof_test,p_val = StrOUD(test_fft,baseline_fft,lof_baseline,lrd_baseline,k_neighbors_test,dist_test)
#     # Compute ROC curve and ROC area for each class
#     fpr, tpr, thresholds = roc_curve(true_class, p_val,pos_label = 0)
#     fpr_ls.append(fpr)
#     tpr_ls.append(tpr)

#     print("K = "+str(k))
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr,tpr,label="K = "+ str(k)+ " auc= "+str(roc_auc))
#     plt.legend(loc=4)

# plt.savefig("roc_auc",bbox_inches='tight')
# plt.show()
