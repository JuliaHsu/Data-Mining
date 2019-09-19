import re
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
import sys
import scipy
from sklearn.model_selection import StratifiedKFold
path = "./libsvm/python"
sys.path.append(path)
path_tool = "./libsvm/tools"
sys.path.append(path_tool)
from svmutil import *
from grid import *
from info_gain import info_gain
train_path = './train_handwriting.data'
TRAIN_SIZE = 2000
train_data = []
PIXELS = 784
max_size = 0
train_classdf = []
train_class = []
traindf = []
col_name = []
info_gain_res=[]

def read_data():
    global max_size, train_class,traindf
    with open(train_path) as f:
        data = f.readlines()
    for row in range(TRAIN_SIZE):
        data[row] = data[row].replace('{','')
        data[row] = data[row].replace('}','')
        train_data.append(data[row].rstrip().split(','))
    # print(train_data)
    for row in range(len(train_data)):
        for col in range(len(train_data[row])):
            train_data[row][col] = train_data[row][col].split()
        if len(train_data[row]) > max_size:
            max_size = len(train_data[row])
    # print(max_size)
    # print(train_data)
    train_arr = np.asarray(train_data)
    # print(train_arr[0][0][0])

    train_dense = [[0]*(PIXELS) for i in range(TRAIN_SIZE)]
    for row in range(TRAIN_SIZE):
        for col in range (max_size):
            if col < len(train_arr[row]):
                pixel = int(train_arr[row][col][0])
                train_dense[row][pixel] = int(train_arr[row][col][1])
    
    for i in range(784):
        col_name.append(str(i))
    traindf = pd.DataFrame(train_dense,dtype='int',columns=col_name)
    train_classdf = traindf['524']
    train_class = train_classdf.iloc[:].tolist()
    # print(traindf['524'].value_counts())
    traindf = traindf.drop(['524'],axis=1)
    return traindf
    
    # print (traindf)
def get_info_gain(train,classes):
    global info_gain_res
    
    # info_gain_res = dict(zip(col_name,mutual_info_classif(traindf, train_class, discrete_features=True)))
    for i in range(PIXELS):
        if i==524:
            pass
        else:
            # train[str(i)] to access the dataframe column e.g, train['0']
            info_gain_res.append(info_gain.info_gain(classes, train[str(i)]))
    # info_gain_res = mutual_info_classif(traindf, train_class, discrete_features=True)
    with open('./info_gain.txt', 'w') as f:
        print(info_gain_res,file=f)
    return info_gain_res

def remove_attributes(info_gain_res,train,test):
    attributes_rm =[]
    for i in range(783):
        info_gain = float("{:.8f}".format(float(info_gain_res[i])))
        if info_gain<=0.0: 
            attributes_rm.append(i)
    train_red = train
    test_red = test
    train_red.drop(train_red.columns[attributes_rm], axis = 1, inplace = True) 
    test_red.drop(test_red.columns[attributes_rm], axis = 1, inplace = True) 
    # print (train_red)
    return train_red,test_red
def pca_reduction(train,test):
    # standardize dataset before applying PCA 
    scaler = StandardScaler()
    train_std = scaler.fit_transform(train) 
    test_std = scaler.transform(test)
    pca = PCA(.9)
    train_pca = pca.fit_transform(train_std)
    test_pca = pca.transform(test_std)
    print(train_pca.shape)
    return train_pca, test_pca
def split_data(trainset,classls,split_ratio):
    train_split, test_split, train_split_cls, test_split_cls = train_test_split(trainset, classls, test_size=split_ratio)
    # train_split_cls = X_class.iloc[:].tolist()
    # test_split_cls = y_class.iloc[:].tolist()
    print(train_split.shape)
    return train_split, test_split, train_split_cls, test_split_cls

def cross_validation(train_split,train_class_split):
    skf = StratifiedKFold(n_splits=10,shuffle = False)
    for train_index, test_index in skf.split(train_split,train_class_split):
        train_cross=[[0]*(train_split.shape[1]) for i in range(len(train_index))]
        train_cross_class=[]
        test_cross=[[0]*(train_split.shape[1]) for i in range(len(test_index))]
        test_cross_class=[]
        for i in range(len(train_index)):
            train_cross[i] = train_split[train_index[i]]  
            train_cross_class.append(train_class_split[train_index[i]])
        for j in range(len(test_index)):
            test_cross[j] = train_split[test_index[j]]
            test_cross_class.append(train_class_split[test_index[j]]) 
        svm_classifier(train_cross, test_cross,train_cross_class,test_cross_class)
def svm_classifier(train, test,train_cls,test_cls): 
    # convert np array to libsvm dataset format to conduct grid search
    print(train)
    with open('./train_libsvm.data','w') as f:
        for j in range(len(train_cls)):
            f.write(" ".join(
                  [str(int(train_cls[j]))] + ["{}:{}".format(i,train[j][i]) 
                  for i in range(train.shape[1]) if train[j][i] != 0])+"\n")
    # convert np array to scipy array
    train = scipy.asarray(train)  
    classes = scipy.asarray(train_cls)
    # generate problem of svm
    prob  = svm_problem(classes, train)
    # conduct grid search to find the best parameters for c and g
    # rate, param_dict = find_parameters("train_libsvm.data", '-log2c -5,15,2 -log2g 3,-15,-2 -v 5 -gnuplot null')
    # convert the best parameters dictionary to list 
    # param_str = '-t 0 -b 1'+ ' -c '+ str(param_dict.get('c')) + ' -g ' + str(param_dict.get('g'))
    param_str = '-t 0 -b 1 -c 8.0 -g 0.001953125'
    param = svm_parameter(param_str)
    # train
    m=svm_train(prob,param)
    # predict and evaluate the result with the test classes (cross validation)
    p_labs, p_acc, p_vals = svm_predict(test_cls, test, m)
    # p_labs, p_acc, p_vals = svm_predict(test, m)
    print(p_labs)
    # print(param_dict)
    return p_acc
def build_model_org(train_org):
    accur_org = 0.0
    # split data into 80% (for training and cross validation) and 20% (for testing) 
    train_split, test_split, train_split_cls, test_split_cls = split_data(train_org,train_class,1/5.0)
    # split the 80% of remaining data for cross validation
    train_cross, test_cross, train_cross_cls, test_cross_cls = split_data(train_split,train_split_cls,1/4.0)
    # conduct svm and cross validation with 20 % of remaining data for testing
    # find the best parameters for svm
    # {'c': 8.0, 'g': 0.001953125}
    
    # cross validation
    # svm_classifier(train_cross, test_cross, train_cross_cls, test_cross_cls)
    # build svm model by training 80% of data and use the 20 % of data to test 
    accur_org = svm_classifier(train_split,test_split,train_split_cls,test_split_cls)
    return accur_org, train_split.shape
def build_model_pca(traindf):
    accur_pca =0.0
    # split data into 80% (for training and cross validation) and 20% (for testing) 
    train_split, test_split, train_split_cls, test_split_cls = split_data(traindf,train_class,1/5.0)
    # split the 80% of remaining data for cross validation
    train_cross, test_cross, train_cross_cls, test_cross_cls = split_data(train_split,train_split_cls,1/4.0)
    
    train_cross_pca, test_cross_pca = pca_reduction(train_cross,test_cross)
    
    train_split_pca, test_split_pca = pca_reduction(train_split,test_split)
    # conduct svm and cross validation with 20 % of remaining data for testing
    # find the best parameters for svm
    # {'c': 8.0, 'g': 0.001953125}
    # print(train_cross)
    # cross validation 
    # svm_classifier(train_cross_pca, test_cross_pca, train_cross_cls, test_cross_cls)
    # build svm model by training 80% of data and use the 20 % of data to test 
    accur_pca= svm_classifier(train_split_pca,test_split_pca,train_split_cls,test_split_cls)
    return accur_pca, train_split_pca.shape

def build_model_ig(traindf):
    accur_ig =0.0
    # split data into 80% (for training and cross validation) and 20% (for testing) 
    train_split, test_split, train_split_cls, test_split_cls = split_data(traindf,train_class,1/5.0)
    # split the 80% of remaining data for cross validation
    train_cross, test_cross, train_cross_cls, test_cross_cls = split_data(train_split,train_split_cls,1/4.0)

    ig_cross = get_info_gain(train_cross,train_cross_cls)
    train_cross_red,test_cross_red = remove_attributes(ig_cross,train_cross,test_cross)
    train_cross_red = train_cross_red.values
    test_cross_red = test_cross_red.values
        # cross validation 
    svm_classifier(train_cross_red, test_cross_red, train_cross_cls, test_cross_cls)

    ig_split = get_info_gain(train_split,train_split_cls)
    train_split_red,test_split_red = remove_attributes(ig_split,train_split,test_split)
    
    train_split_red = train_split_red.values
    
    test_split_red = test_split_red.values
   
    # conduct svm and cross validation with 20 % of remaining data for testing
    # find the best parameters for svm
    # {'c': 8.0, 'g': 0.001953125}
    # print(train_cross)

    # build svm model by training 80% of data and use the 20 % of data to test 
    accur_ig = svm_classifier(train_split_red,test_split_red,train_split_cls,test_split_cls)
    return accur_ig, train_split_red.shape

traindf = read_data()
train_org = traindf.values
# info_gain_res = get_info_gain()
# train_red = remove_attributes(info_gain_res)

accur_org, train_split_org = build_model_org(train_org)
accur_pca, train_split_pca = build_model_pca(traindf)
accur_ig, train_split_ig = build_model_ig(traindf)
print("accuracy org = "+ str(accur_org) + " " + str(train_split_org))
print("accuracy ig = "+ str(accur_ig) + " " + str(train_split_ig))
print("accuracy pca = "+ str(accur_pca)+ " " + str(train_split_pca))








