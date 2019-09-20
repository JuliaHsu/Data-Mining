import numpy as np
from info_gain import info_gain
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd 
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import sys
import scipy
from scipy.stats import randint
import random
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

path = "./libsvm/python"
sys.path.append(path)
path_tool = "./libsvm/tools"
sys.path.append(path_tool)
from svmutil import *
from grid import *
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from matplotlib.legend_handler import HandlerLine2D
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score


def read_transactions():
    data = pd.read_csv("./BlackFriday.csv") 
    data = data.drop(['User_ID', 'Product_ID','Product_Category_2', 'Product_Category_3','Purchase'], axis=1)
    return data
def encode_data(data):
    # Label encoding

    encode = LabelEncoder()
    encode.fit(['0-17','18-25','26-35','36-45','46-50','51-55', '55+'])
    data['Age'] = encode.transform(data['Age'])

    encode.fit(['M','F'])
    data['Gender'] = encode.transform(data['Gender'])
    encode.fit(['0','1','2', '3', '4+'])
    data['Stay_In_Current_City_Years'] = encode.transform(data['Stay_In_Current_City_Years'])

    encode.fit(['A','B','C'])
    data['City_Category'] = encode.transform(data['City_Category'])
    product_cat = data['Product_Category_1']
    train_data = data.drop(['Product_Category_1'],axis = 1)
    product_cat.to_csv("cat.csv")
    print (train_data)
    return train_data, product_cat
    
def split_data(train_data, product_cat):
    # scaler = StandardScaler()
    # scaler.fit(train_data)
    # train_data = scaler.transform(train_data)
    X_train, X_test, y_train, y_test = train_test_split(train_data, product_cat,test_size = 0.4,random_state = 42)
    
    

    # gnb = GaussianNB().fit(X_train, y_train) 
    # gnb_predictions = gnb.predict(X_test) 
    # # accuracy on X_test 
    # accuracy = gnb.score(X_test, y_test) 
    # print (accuracy) 
    return X_train, X_test, y_train, y_test

def svm_classifier(X_train, X_test, y_train, y_test): 
    
   
    # convert np array to scipy array
    train = scipy.asarray(X_train)  
    classes = scipy.asarray(y_train)
    # generate problem of svm
    prob  = svm_problem(classes, train)
    # conduct grid search to find the best parameters for c and g
    # rate, param_dict = find_parameters("train_libsvm.data", '-log2c -5,15,2 -log2g 3,-15,-2 -v 5 -gnuplot null')
    # convert the best parameters dictionary to list 
    # param_str = '-t 0 -b 1'+ ' -c '+ str(param_dict.get('c')) + ' -g ' + str(param_dict.get('g'))
    param_str = '-t 0 -b 1 -c 8.0 -g 0.001953125 -h 0'
    param = svm_parameter(param_str)
    # train
    m=svm_train(prob,param)
    # predict and evaluate the result with the test classes (cross validation)
    p_labs, p_acc, p_vals = svm_predict(y_test, X_test, m)
    # p_labs, p_acc, p_vals = svm_predict(test, m)
    print(p_labs)
    # print(param_dict)
    return p_acc

def knn(X_train, X_test, y_train, y_test):
    print("knn; k = 50")
    knn = KNeighborsClassifier(n_neighbors = 30).fit(X_train, y_train) 
# 0.31905632211169555
    # accuracy on X_test 
    accuracy = knn.score(X_test, y_test) 
    print (accuracy )
     
   

def random_forest_classifier(X_train, X_test, y_train, y_test):
    print("Random Forest: ")
    # smote = SMOTE(random_state=42)
    # smote = SMOTEENN(random_state=42)

    # X_train, y_train = smote.fit_resample(X_train,y_train)
    # {'criterion': 'gini', 'max_depth': 6, 'max_leaf_nodes': 47, 'min_samples_leaf': 31, 'min_samples_split': 12, 'n_estimators': 56}
    # 0.30175184043230974
    param_dist = {"criterion": ["gini", "entropy"],"min_samples_split": [10, 15, 20,25],"max_depth": [5,10, 20, 30, 40, 50],"min_samples_leaf": [10, 20, 30], 'n_estimators': [20,50,100]}
    rf_classifier = RandomForestClassifier(max_depth = 30, min_samples_leaf=30, min_samples_split=10,n_estimators=50)
    # rand_forest_rs = RandomizedSearchCV(rf_classifier, param_distributions=param_dist, cv=3,n_iter=20,return_train_score = True)
    rf_classifier.fit(X_train, y_train)
    predict = rf_classifier.predict(X_test)
    # print(rf_classifier.best_params_)


    print (rf_classifier.score(X_test, y_test))

def dt_classifier(X_train, X_test, y_train, y_test):
    print("Decision tree: ")
    
    # smote = SMOTE(random_state=42)
    # smote = SMOTEENN(random_state=42)
    # X_train, y_train = smote.fit_resample(X_train,y_train)
    # 0.3344959563969846
    # dtree_model = DecisionTreeClassifier().fit(X_train, y_train) 
    # dtree_predictions = dtree_model.predict(X_test)
    # 0.33451920885825764
    train_results = []
    test_results = []
    dtree_model = DecisionTreeClassifier(max_depth=30)
    
    # test data
    dtree_predictions = BaggingClassifier(base_estimator=dtree_model, n_estimators=100, random_state=42).fit(X_train, y_train).predict(X_test)

    print(f1_score(y_test, dtree_predictions, average=None))
    print(accuracy_score(y_test, dtree_predictions))

    

data = read_transactions()
train_data, product_cat = encode_data(data)
X_train, X_test, y_train, y_test = split_data(train_data, product_cat)
# svm_classifier(X_train, X_test, y_train, y_test)
knn(X_train, X_test, y_train, y_test)
# random_forest_classifier(X_train, X_test, y_train, y_test)
# dt_classifier(X_train, X_test, y_train, y_test)

    