import re
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV,train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import tree
import csv
from sklearn.impute import SimpleImputer
from scipy.stats import randint
from sklearn.neighbors import KNeighborsClassifier
from imblearn.combine import SMOTETomek
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
train_path = './train_breast_cancer.csv'
TRAIN_SIZE = 286
train_data=[]
attributes = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps'\
,'deg-malig', 'breast', 'breast-quad', 'irradiat', 'class']
train_class = []
clf_num = 4
def read_data():
    global train_class
    with open(train_path) as f:
        data = f.readlines()
    for row in range(TRAIN_SIZE):
        data[row] = data[row].replace('\'','')
        train_data.append(data[row].rstrip().split(','))
    train_arr = np.asarray(train_data)
    traindf = pd.DataFrame(train_arr,columns=attributes)
    traindf['class'] = traindf['class'].map({'no-recurrence-events':0, 'recurrence-events':1})
    train_class = traindf['class']
    traindf = traindf[attributes[:-1]]
    return traindf
    # print (traindf)
def fill_missing_value(traindf):
    imp = SimpleImputer(missing_values='?',strategy="most_frequent")
    arr = imp.fit_transform(traindf)
    traindf = pd.DataFrame(arr,columns=traindf.columns.values)
    return traindf

def convert_dummy(traindf):
    traindf = pd.get_dummies(traindf)
    attributes_dum= list(traindf.columns.values)
    with open('./train_dummy.txt', 'w') as f:
        print(traindf,file=f)
    # print(attributes_dum)
    return traindf,attributes_dum
def split_data(traindf,train_class):
    train_split, test_split, train_split_cls, test_split_cls = train_test_split(traindf, train_class, test_size=1/4.0)
    print(len(train_split_cls))
    return train_split, test_split,train_split_cls, test_split_cls
# decision stump
def decision_tree_stump(train_split,train_split_cls,test_split,test_split_cls):
    f1 = 0.0
    # create decision stump by setting the max depth as 1
    clf = tree.DecisionTreeClassifier(max_depth=1,random_state=42)
    # print the decision tree (stump) 
    clf.fit(train_split,train_split_cls)
    tree.export_graphviz(clf,out_file='dt_stump.dot')
    # get the most important (or the only) feature in Stump tree
    # print(attributes_dum[np.argmax(clf.feature_importances_)]) 

    # cross validation
    # cross_val_predict return the whole result of each fold (size of 
    # examples will be the same as test size)
    y_predict = cross_val_predict(clf,train_split,train_split_cls,cv = 10)
    # y_predict = clf.predict(test_split)
    #  confusion matrix for the result 
    conf_matrix = confusion_matrix(train_split_cls,y_predict) 
    f1 = f1_score(train_split_cls, y_predict,labels = [0,1],average=None)
    print("----- Decision Tree Stump -----")
    print (conf_matrix)
    print(f1)
    return f1
    # print all arrtirbutes and corresponding feature importances
    # for name, importance in zip(attributes_dum, clf.feature_importances_):
    #     print(name, importance)

# decision tree unpruned
def decision_tree_unpruned(train_split,train_split_cls,test_split, test_split_cls):
    f1 = 0.0
    # create unpruned decision tree 
    dt_unpruned_clf = tree.DecisionTreeClassifier(random_state=42)
    # print the unpruned decision tree 
    dt_unpruned_clf.fit(train_split,train_split_cls)
    tree.export_graphviz(dt_unpruned_clf,out_file='dt_unpruned.dot')
    
    # cross validation
    # cross_val_predict returns the whole result of each fold (size of 
    # examples will be the same as test size)
    y_predict = cross_val_predict(dt_unpruned_clf,train_split,train_split_cls,cv = 10)
    # y_predict = dt_unpruned_clf.predict(test_split)
    #  confusion matrix for the result 
    conf_matrix = confusion_matrix(train_split_cls,y_predict) 
    f1 = f1_score(train_split_cls, y_predict,labels = [0,1],average=None)
    print("----- Decision Tree unpruned -----")
    print (conf_matrix)
    print(f1)
    return f1

    
    # print all arrtirbutes and corresponding feature importances
    # for name, importance in zip(attributes_dum, clf.feature_importances_):
    #     print(name, importance) 

def decision_tree_pruned(train_split,train_split_cls,test_split, test_split_cls):
    f1 = 0.0
    param_dist = {"criterion": ["gini", "entropy"],"min_samples_split": randint(2, 50),
                  "max_depth": randint(1, 10),"min_samples_leaf": randint(2, 10),"max_leaf_nodes": randint(2,10)}
    # param_dist = {"criterion": ["gini", "entropy"],"min_samples_split": [10,20,50],
    #               "max_depth": [None,2,5,10],"min_samples_leaf": [2,5,10],"max_leaf_nodes": [None,2,5,10]}
    clf = tree.DecisionTreeClassifier(max_depth=5,max_leaf_nodes=2,min_samples_leaf=5, min_samples_split=28)
    # dt_grid_search = GridSearchCV(clf,param_grid=param_dist, cv=10,scoring='f1')
    # random_search = RandomizedSearchCV(clf, param_distributions=param_dist, cv=10,scoring='f1',n_iter=50)
    # random_search.fit(train_split, train_split_cls)
    # dt_grid_search.fit(traindf,train_class)
    # print(random_search.best_score_)
    # best_clf = random_search.best_estimator_
    # print(random_search.best_estimator_)
    # print the decision tree 
    clf.fit(train_split, train_split_cls)
    tree.export_graphviz(clf,out_file='dt_pruned.dot')


    # cross validation
    # cross_val_predict return the whole result of each fold (size of 
    # examples will be the same as test size)
    y_predict = cross_val_predict(clf,train_split,train_split_cls,cv = 10)
    # y_predict = clf.predict(test_split)
    # y_predict = cross_val_predict(dt_grid_search,traindf,train_class,cv = 10)
    #  confusion matrix for the result 
    conf_matrix = confusion_matrix(train_split_cls,y_predict) 
    f1 = f1_score(train_split_cls, y_predict,labels = [0,1],average=None)
    print("----- Decision Tree pruned -----")
    print (conf_matrix)
    print(f1)
    # print(random_search.best_params_)
    return f1
    # print(dt_grid_search.best_params_)
    
    # print all arrtirbutes and corresponding feature importances
    # for name, importance in zip(attributes_dum, clf.feature_importances_):
    #     print(name, importance) 

def knn_claffifier(train_split,train_split_cls,test_split, test_split_cls):
    f1 = 0.0
    # param_dist = {"n_neighbors": randint(2, 50)}
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    # knn_random_search = RandomizedSearchCV(knn_clf, param_distributions=param_dist, cv=10,scoring='f1',n_iter=50)
    # knn_random_search.fit(train_split, train_split_cls)
    # y_predict = knn_random_search.predict(test_split)
    y_predict = cross_val_predict(knn_clf,train_split,train_split_cls,cv = 10)
    conf_matrix = confusion_matrix(train_split_cls,y_predict) 
    f1 = f1_score(train_split_cls, y_predict,labels = [0,1],average=None)
    print("----- knn classifier-----")
    print (conf_matrix)
    print(f1)
    # print(knn_random_search.best_params_)
    return f1


def decision_tree_post_pruned(traindf):
    f1 = 0.0
    # create decision tree 
    clf = tree.DecisionTreeClassifier(random_state=42) 
    clf.fit(traindf,train_class)
    post_prune(clf.tree_, 0, 5)
    tree.export_graphviz(clf,out_file='dt_post_pruned_.dot')
    clf.fit(traindf,train_class)
    tree.export_graphviz(clf,out_file='test.dot')
    # cross validation
    # cross_val_predict return the whole result of each fold (size of 
    # examples will be the same as test size)
    y_predict = cross_val_predict(clf,traindf,train_class,cv = 10)
    #  confusion matrix for the result 
    conf_matrix = confusion_matrix(train_class,y_predict) 
    f1 = f1_score(train_split_cls, y_predict,labels = [0,1],average=None)
    print (conf_matrix)
    print(f1)
    return f1


def post_prune(subtree, index, threshold):
    if subtree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        subtree.children_left[index] = TREE_LEAF
        subtree.children_right[index] = TREE_LEAF
    if subtree.children_left[index] != TREE_LEAF:
        post_prune(subtree, subtree.children_left[index],threshold)
        post_prune(subtree, subtree.children_right[index], threshold)


def cross_validation(traindf, train_class):
    cross_validation_res = np.zeros(shape=(clf_num,2))
    f1_score_arr = np.zeros(shape=(clf_num,5,2))
    
    # list for each clasifier function 
    classifier_functions=[decision_tree_stump,decision_tree_unpruned,decision_tree_pruned,knn_claffifier]
    
    # classifier_functions=[dt_stump_resample,dt_unpruned_resample,dt_pruned_resample, knn_clf_resample]
    # train_indices,test_indices = split_data(traindf, train_class)
    skf = StratifiedKFold(n_splits=5,random_state = 42,shuffle = True)
    i = 0
    for train_index, test_index in skf.split(traindf,train_class):
        X_train, X_test, y_train, y_test = traindf.iloc[train_index], traindf.iloc[test_index], train_class[train_index], train_class[test_index]
        # train_resampled, class_resampled = training(X_train,y_train)
        # fill in f1 score of each classifier in the array
        for j in range(len(classifier_functions)):
            score = classifier_functions[j](X_train,y_train,X_test,y_test)
            f1_score_arr[j][i]=score
        i+=1
    # get average f1 score of each classifier
    for i in range(clf_num):
        cross_validation_res[i] = f1_score_arr[i].mean(axis=0)
    print(cross_validation_res) 

def training(traindf,train_class):
    # Combine over- and under-sampling using SMOTE and Tomek links.
    smote_tomek = SMOTETomek(random_state=42)
    train_resampled, class_resampled = smote_tomek.fit_resample(traindf,train_class)
    return train_resampled, class_resampled

def dt_stump_resample(train_resampled,class_resampled,testdf,test_class):
    f1_score = 0.0
    # create decision stump by setting the max depth as 1
    dt_stump_clf = tree.DecisionTreeClassifier(max_depth=1,random_state=42)
    # print the decision tree (stump) 
    dt_stump_clf.fit(train_resampled,class_resampled)
    tree.export_graphviz(dt_stump_clf,out_file='dt_stump.dot')
    # get the most important (or the only) feature in Stump tree
    # print(attributes_dum[np.argmax(clf.feature_importances_)]) 

    # cross validation
    # cross_val_predict return the whole result of each fold (size of 
    # examples will be the same as test size)
    # y_predict = cross_val_predict(clf,traindf,train_class,cv = 10)
    predict = dt_stump_clf.predict(testdf)
    #  confusion matrix for the result 
    print("----- Decision Tree Stump (Resample)-----")
    f1_score = evaluation(test_class,predict)
   
    print(f1_score)
    return f1_score

def dt_unpruned_resample(train_resampled,class_resampled,testdf,test_class):
    dt_unpruned_clf = tree.DecisionTreeClassifier()
    f1_score =0.0
    # predict
    dt_unpruned_clf.fit(train_resampled,class_resampled)
    tree.export_graphviz(dt_unpruned_clf,out_file='dt_unpruned.dot')
    predict = dt_unpruned_clf.predict(testdf)
    # print(predict)
    # for i in range(len(predict)):
    #     print(predict[i])
    
    # print("pca lambda = " + str(pca_lambda))
    print("----- Decision Tree Unpruned (Resample)-----")
    f1_score = evaluation(test_class,predict)
    
    print(f1_score)
    return f1_score

def dt_pruned_resample(train_resampled,class_resampled,testdf,test_class):
    param_dist = {"criterion": ["gini", "entropy"],"min_samples_split": randint(2, 50),
                  "max_depth": randint(1, 10),"min_samples_leaf": randint(2, 10),"max_leaf_nodes": randint(2,10)}
    dt_pruned_clf = tree.DecisionTreeClassifier()
    dt_pruned_random_search = RandomizedSearchCV(dt_pruned_clf, param_distributions=param_dist, cv=10,scoring='f1',n_iter=50)
    dt_pruned_random_search.fit(train_resampled, class_resampled)
    # dt_grid_search.fit(traindf,train_class)
    
    
    # print the decision tree 
    best_clf = dt_pruned_random_search.best_estimator_
    tree.export_graphviz(best_clf,out_file='dt_pruned.dot')


    # cross validation
    # cross_val_predict return the whole result of each fold (size of 
    # examples will be the same as test size)
    # y_predict = cross_val_predict(random_search,traindf,train_class,cv = 10)
    predict = best_clf.predict(testdf)
    #  confusion matrix for the result 
    print("----- Decision Tree pruned (Resample)-----")
    f1_score = evaluation(test_class,predict)
    
    print(dt_pruned_random_search.best_params_)
    print(f1_score)
    return f1_score

def knn_clf_resample(train_resampled,class_resampled,testdf,test_class):
    param_dist = {"n_neighbors": randint(2, 50)}
    knn_clf = KNeighborsClassifier()
    knn_random_search = RandomizedSearchCV(knn_clf, param_distributions=param_dist, cv=10,scoring='f1',n_iter=50)
    knn_random_search.fit(train_resampled, class_resampled)
    predict = knn_random_search.predict(testdf)
    print("----- knn classifier (Resample)-----")
    f1_score = evaluation(test_class,predict)
    
    print(knn_random_search.best_params_)
    print(f1_score)
    return f1_score
    
   

def evaluation(test_class,predict):
    conf_matrix = confusion_matrix(test_class,predict) 
    print(conf_matrix)
    return f1_score(test_class, predict,labels = [0,1],average=None) 

traindf = read_data()
traindf = fill_missing_value(traindf)
traindf,attributes_dum = convert_dummy(traindf)
# cross_validation(traindf,train_class)
# i=0

train_split, test_split,train_split_cls, test_split_cls = split_data(traindf,train_class)
decision_tree_stump(train_split,train_split_cls,test_split, test_split_cls)
decision_tree_unpruned(train_split,train_split_cls,test_split, test_split_cls)
decision_tree_pruned(train_split,train_split_cls,test_split, test_split_cls)
knn_claffifier(train_split,train_split_cls,test_split, test_split_cls)
# decision_tree_post_pruned(traindf,attributes_dum)




