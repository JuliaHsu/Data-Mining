import re
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold,StratifiedKFold
from scipy.stats import randint
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV,train_test_split
train_path = './churning.data'
TRAIN_SIZE = 9688
train_data = []
clf_num = 3
train_cls =[]
def read_data():
    global train_class,train_cls
    with open(train_path) as f:
        data = f.readlines()
    for row in range(TRAIN_SIZE):
        train_data.append(data[row].rstrip().split(','))
    train_arr = np.asarray(train_data)
    traindf = pd.DataFrame(train_arr,dtype='int')
    train_class = traindf.iloc[:,-1]
    traindf = traindf.iloc[:,:-1]
    return traindf
def count_class(classls):
    c1 =0
    c0 =0
    for row in range(len(classls)):
        if classls[row] == 1:
            c1+=1
        elif classls[row] ==0:
            c0+=1
    print("class 0 = "+ str(c0))
    print("class 1 = "+ str(c1))

def decision_tree_org(traindf,train_class):
    train_split, test_split, train_split_cls, test_split_cls = train_test_split(traindf, train_class, test_size=1/10)
    # create unpruned decision tree 
    # dt_unpruned_clf = tree.DecisionTreeClassifier(random_state=42)
    # # print the unpruned decision tree 
    # dt_unpruned_clf.fit(traindf,train_class)
    dt_pruned_resample(train_split,train_split_cls,test_split,test_split_cls)
    # cross validation
    # cross_val_predict return the whole result of each fold (size of 
    # examples will be the same as test size)
    # y_predict = cross_val_predict(dt_unpruned_clf,traindf,train_class,cv = 10)
    #  confusion matrix for the result 
    
    print("----- Decision Tree pruned org-----")
    

def cross_validation(traindf, train_class):
    cross_validation_res = np.zeros(shape=(clf_num,2))
    f1_score_arr = np.zeros(shape=(clf_num,10,2))
    
    # list for each clasifier function 
    # classifier_functions=[decision_tree_stump,decision_tree_unpruned,decision_tree_pruned,knn_claffifier]
    
    resample_functions=[oversample,undersample,smote_combine_resample]
    # train_indices,test_indices = split_data(traindf, train_class)
    skf = StratifiedKFold(n_splits=10,shuffle = False)
    i = 0
    for train_index, test_index in skf.split(traindf,train_class):
        X_train, X_test, y_train, y_test = traindf.iloc[train_index], traindf.iloc[test_index], train_class[train_index], train_class[test_index]
        for j in range(len(resample_functions)):
            train_resampled, class_resampled = resample_functions[j](X_train,y_train)
            count_class(class_resampled)
            score = dt_pruned_resample(train_resampled,class_resampled,X_test,y_test)
            f1_score_arr[j][i]=score
        i+=1
    # get average f1 score of each classifier
    for i in range(clf_num):
        cross_validation_res[i] = f1_score_arr[i].mean(axis=0)
    print(cross_validation_res) 

def smote_combine_resample(traindf,train_class):
    print("-- Decision Tree pruned (smote_combine_resample)--")
    # Combine over- and under-sampling using SMOTE and Tomek links.
    smote_tomek = SMOTETomek()
    train_resampled, class_resampled = smote_tomek.fit_resample(traindf,train_class)
    return train_resampled, class_resampled

def oversample(traindf,train_class):
    print("-- Decision Tree pruned (oversample)--")
    smote = RandomOverSampler()
    train_resampled, class_resampled = smote.fit_resample(traindf,train_class)
    return train_resampled, class_resampled
def undersample(traindf, train_class):
    print("-- Decision Tree pruned (undersample)--")
    cc = RandomUnderSampler()
    train_resampled, class_resampled = cc.fit_resample(traindf,train_class)
    return train_resampled, class_resampled


def dt_pruned_resample(train_resampled,class_resampled,testdf,test_class):
    param_dist = {"criterion": ["gini", "entropy"],"min_samples_split": randint(2, 50),
                  "max_depth": randint(1, 10),"min_samples_leaf": randint(2, 10),"max_leaf_nodes": randint(2,10)}
    dt_pruned_clf = tree.DecisionTreeClassifier()
    dt_pruned_random_search = RandomizedSearchCV(dt_pruned_clf, param_distributions=param_dist, cv=10,scoring='f1',n_iter=50)
    dt_pruned_random_search.fit(train_resampled, class_resampled)
    
    f1_score =0.0
    
    # predict
    # predict = dt_pruned_clf.predict(testdf)
    predict = dt_pruned_random_search.predict(testdf)
    f1_score = evaluation(test_class,predict)
    print(f1_score)
    return f1_score

def evaluation(test_class,predict):
    conf_matrix = confusion_matrix(test_class,predict) 
    print(conf_matrix)
    return f1_score(test_class, predict,labels = [0,1],average=None) 


traindf = read_data()
count_class(train_class)
decision_tree_org(traindf,train_class)
cross_validation(traindf, train_class)
