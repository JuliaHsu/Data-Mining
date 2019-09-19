# python3
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,RandomizedSearchCV,GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from numpy import array
from scipy.stats import randint
import random
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold,StratifiedKFold
from scipy import interp

train_path = './train_drugs.data'
test_path = './test.data'
TRAIN_SIZE = 800
TEST_SIZE = 350
# number of classifiers for cross validation to choose the best model
# turned out that random forest might be the best model, set clf_num =1  
clf_num = 1
pca_lambda = 0.85
# number of estimaters for bagging and random forest classifiers
num_trees = 100

# class
cs = []
features = []
max_d=0
train_data = [[] for i in range(800)]
test_data = []
train_set = [[0]*(max_d+1) for i in range(800)]
test_set = [[0]*(max_d+1) for i in range(350)]


def main():
    global train_set, test_set
    read_data()
    get_max_dimension()
    train_set = revert_sparse(TRAIN_SIZE,train_data)
    test_set = revert_sparse(TEST_SIZE,test_data)
    
    traindf = pd.DataFrame(train_set, dtype = 'float')
    classdf = pd.DataFrame(cs)
    test_class=[]

    # split_data() only one time 
    # traindf, testdf,train_class,test_class = split_data()
    
    # perform K fold validation
    # K_fold_validation(traindf, classdf)

    # create data frame for real testing
    traindf,testdf,train_class = create_dataframe()

    # resample data for real training
    test_red, train_resampled, class_resampled = training_sampling(traindf,testdf,train_class)

    # classifiers
    random_forest_classifier(train_resampled,class_resampled,test_red,test_class)
    # SVM_classifier(train_resampled,class_resampled,test_red,test_class)
    # decision_tree_classifier(train_resampled,class_resampled,test_red,test_class)
    # naive_bayes_classifier(train_resampled,class_resampled,test_red)
    # adaboost_classifier(train_resampled,class_resampled,test_red,test_class)    
    # gradient_boosting_clasifier(train_resampled,class_resampled,test_red,test_class) 

def read_data():
    # handle the train data into array 
    with open(train_path) as f:
        data = f.readlines()
    # train_data[row][0] to get drug active or inactive
    # train_data[row][1][index] to get feature element
    train_buffer = [[] for i in range(800)]
    for row in range(TRAIN_SIZE):
        train_buffer[row] = data[row].split('\t')
        train_buffer[row][1] = train_buffer[row][1].split(' ')
        train_buffer[row][1].remove('\n')
        # size.append(len(train_data[row][1]))

    # put training data into a 2D list
    for row in range(TRAIN_SIZE):
        for col in range(6061):
            if col< len(train_buffer[row][1]):
                train_data[row].append(train_buffer[row][1][col])
    
    # cs[row] to show the class of row th
    for i in range(800):
        cs.append(int(train_buffer[i][0]))

    # handle the test data into array 
    with open(test_path) as f:
        data = f.readlines()
    for row in range(350):
        test_data.append(data[row].split()) 

# get maximum dimension
def get_max_dimension():
    global max_d
    for row in range(TRAIN_SIZE):
        for col in range(6061):
            if col < len(train_data[row]):
                x = int(train_data[row][col])
                if x > max_d:
                    max_d= int(train_data[row][col])
    # features name
    for i in range(max_d+1):
        features.append(str(i))

# convert sparse binary matrix to dense matrix
def revert_sparse(rows,data):
    dense_matrix = [[0]*(max_d+1) for i in range(rows)]
    for row in range(rows):
        for col in range(6061):
            if col < len(data[row]):
                    index = int(data[row][col])
                    dense_matrix[row][index] = 1
    return dense_matrix
# Perform K fold cross validation, cv = 10
def K_fold_validation(traindf, train_class):
    # Run classifier with cross-validation and plot ROC curves
    skf = StratifiedKFold(n_splits=10,shuffle = False)
    # balanced data
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    # imbalanced data
    tprs_org = []
    aucs_org = []
    mean_fpr_org = np.linspace(0, 1, 100)
    # arry for results
    cross_validation_res = np.zeros(shape=(clf_num,2))
    cross_validation_res_imb = np.zeros(shape=(clf_num,2))
    f1_score_arr = np.zeros(shape=(clf_num,10,2))
    f1_score_arr_imb = np.zeros(shape=(clf_num,10,2))

    # list for each clasifier function 
    # classifier_functions=[SVM_classifier,decision_tree_classifier,naive_bayes_classifier,random_forest_classifier,\
    #     adaboost_classifier]

    classifier_functions=[random_forest_classifier]
    i = 0
    k=0
    for train_index, test_index in skf.split(traindf,train_class):
        # get data from dataframe
        # data.iloc[[0,3,6,24], [0,5,6]] # 1st, 4th, 7th, 25th row + 1st 6th 7th columns.
        X_train, X_test, y_train, y_test = traindf.iloc[train_index], traindf.iloc[test_index], train_class.iloc[train_index], train_class.iloc[test_index]
        # Training each X_train (PCA and resample)
        test_red, train_resampled, class_resampled = training_sampling(X_train,X_test,y_train)
        
        # Loop for running different classifiers to select the best classifer for this dataset
        for j in range(len(classifier_functions)):
            score, probas_ = classifier_functions[j](train_resampled,class_resampled,test_red,y_test)
            f1_score_arr[j][i]=score
        
            # roc curve
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
        i+=1
    

        # use imbalance data to build the model
        test_red, train_red, classes = training(X_train,X_test,y_train)
        for j in range(len(classifier_functions)):
            score, probas_org = classifier_functions[j](train_red,classes,test_red,y_test)
            f1_score_arr_imb[j][k]=score

            # roc curve
           
            fpr, tpr, thresholds = roc_curve(y_test, probas_org[:, 1])
            tprs_org.append(interp(mean_fpr_org, fpr, tpr))
            tprs_org[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs_org.append(roc_auc)
        k+=1
        # get average f1 score of each classifier
        for i in range(clf_num):
            cross_validation_res[i] = f1_score_arr[i].mean(axis=0)
            cross_validation_res_imb[i] = f1_score_arr_imb[i].mean(axis=0)
    print("imbalanced: " + str(cross_validation_res_imb))
    print("balanced: " + str(cross_validation_res))
    show_roc_curve(tprs,mean_fpr,aucs,"roc_balanced.png")
    show_roc_curve(tprs_org,mean_fpr_org,aucs_org,"roc_org.png")

def show_roc_curve(tprs,mean_fpr,aucs,filename):
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename,bbox_inches='tight')
    plt.show()

# split training data into training and testing for validation  
def split_data():
    train_split, test_split, train_class, test_class = train_test_split(train_set, cs, test_size=1/4.0)

    # create dataframe for panda
    traindf = pd.DataFrame(train_split, columns = features, dtype = 'float')
    traindf.name='traindf'
    testdf = pd.DataFrame(test_split, columns = features, dtype = 'float')
    testdf.name = 'testdf'

    # print(testdf)
    # traindf['class'] = train_class
    return traindf, testdf,train_class,test_class


# create dataframe for panda without splitting training data
def create_dataframe():
    # create dataframe for panda
    traindf = pd.DataFrame(train_set, columns = features, dtype = 'float')
    traindf.name='traindf'
    testdf = pd.DataFrame(test_set, columns = features, dtype = 'float')
    testdf.name = 'testdf'

    # print(testdf)
    train_class = cs
    # traindf['class'] = train_class
    return traindf, testdf,train_class


# perform pca to reduce the dimension, and use SMOTE over sampling to balance data
def training_sampling(traindf, testdf,train_class):
    # standardize dataset before applying PCA 
    # scaler = StandardScaler()
    # train_std = scaler.fit_transform(traindf) 
    # test_std = scaler.transform(testdf)
 
    train_red, test_red = pca_reduction(traindf,testdf)

    # Combine over- and under-sampling using SMOTE and Tomek links.
    # smote_tomek = SMOTETomek(random_state=42)
    # train_resampled, class_resampled = smote_tomek.fit_resample(train_red,train_class)
    smote = SMOTE(random_state=42)
    train_resampled, class_resampled = smote.fit_resample(train_red,train_class)
    return test_red, train_resampled, class_resampled

# train the original data without balancing
def training(traindf, testdf,train_class):
    # standardize dataset before applying PCA 
 
    train_red, test_red = pca_reduction(traindf,testdf)
    return test_red, train_red, train_class

# pca, set the lambda = 0.85
def pca_reduction(train_std,test_std):

    #Plotting the Cumulative Summation of the Explained Variance
    # pca = PCA().fit(train_std)
    # plt.figure()
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of Components')
    # plt.ylabel('Variance (%)') #for each component
    # plt.title('Drug Dataset PCA Explained Variance')
    # plt.show()
    # choose the minimum number of principal components such that 99% of the variance is retained.

    # pca
    pca = PCA(pca_lambda)
    train_pca = pca.fit_transform(train_std)
    test_pca = pca.transform(test_std)
    print(train_pca.shape)

    # svd
    # svd = TruncatedSVD(n_components=400, n_iter=7, random_state=42)
    # train_pca = svd.fit_transform(train_std)
    # test_pca = svd.transform(test_std)
    # print(train_pca.shape)

    return train_pca,test_pca



def SVM_classifier(train_resampled,class_resampled,test_red,test_class):
    svm_cls = SVC(kernel='linear',probability=True)
    svm_cls.fit(train_resampled,class_resampled)
    # svm_adaboost = AdaBoostClassifier(svm_cls,n_estimators=100,random_state=42)
    # svm_adaboost.fit(train_resampled, class_resampled)

    # predict
    predict = svm_cls.predict(test_red)
    # predict = svm_adaboost.predict(test_red)
    # for i in range(len(predict)):
    #     print(predict[i])
    print("svm_adaboost:")
    print("pca lambda = " + str(pca_lambda))
    f1_score = evaluation(test_class,predict)
    print(f1_score)
    return f1_score

def random_forest_classifier(train_resampled,class_resampled,test_red,test_class):
    f1_score =0.0
    # grid search
    # param_dist = {"criterion": ["gini", "entropy"],"min_samples_split": [10,20,50],
                #   "max_depth": [5,6,7,8,9,10],"min_samples_leaf": [10,15,20],"max_leaf_nodes": [30,40,50]}
    
    # rf_grid_search = GridSearchCV(rand_forest,param_grid=param_dist, cv=10,scoring='f1')
    # rf_grid_search.fit(train_resampled,class_resampled)
    # predict = rf_grid_search.predict(test_red)

    # random cv
    # rand_forest = RandomForestClassifier(class_weight={0:1,1:2})
    param_dist = {"criterion": ["gini", "entropy"],"min_samples_split": randint(2, 50),
                  "max_depth": randint(1, 10),"min_samples_leaf": randint(10, 60),"max_leaf_nodes": randint(10,50), 'n_estimators': randint(50,200)}
    rand_forest = RandomForestClassifier(criterion='entropy',max_depth = 8,max_leaf_nodes = 31, min_samples_leaf =12,min_samples_split = 26, n_estimators=num_trees)
    # rand_forest_rs = RandomizedSearchCV(rand_forest, param_distributions=param_dist, cv=10,scoring='f1',n_iter=20)
    # rand_forest_rs.fit(train_resampled, class_resampled)
    # predict = rand_forest_rs.predict(test_red)
    rand_forest.fit(train_resampled, class_resampled)
    probas_ = rand_forest.predict_proba(test_red)
    predict = rand_forest.predict(test_red)
    # predict = RandomForestClassifier (n_estimators=num_trees, random_state=0,min_samples_leaf=50).fit(train_resampled, class_resampled).predict(test_red) 
    for i in range(len(predict)):
        print(predict[i])
    print("random forest classifier classifier:" + " number of tree = " + str(num_trees))
    print("pca lambda = " + str(pca_lambda))
    # f1_score = evaluation(test_class,predict)
    # print(f1_score)
    # print(rand_forest_rs.best_params_)
    # print(rand_forest_rs.best_score_)
    return f1_score, probas_
    
def decision_tree_classifier(train_resampled,class_resampled,test_red,test_class):
   
    classifier = tree.DecisionTreeClassifier()
    f1_score =0.0
    # predict
    predict = BaggingClassifier(base_estimator=classifier, n_estimators=num_trees, random_state=0).fit(train_resampled, class_resampled).predict(test_red) 
    # print(predict)
    # for i in range(len(predict)):
    #     print(predict[i])
    print("decision tree classifier:\n"+ "number of trees = "+ str(num_trees))
    print("pca lambda = " + str(pca_lambda))
    f1_score = evaluation(test_class,predict)
    return f1_score
    

def naive_bayes_classifier(train_resampled,class_resampled,test_red,test_class):
    gnb = GaussianNB()
    gnb.fit(train_resampled,class_resampled)
    predict = gnb.predict(test_red)
    # for i in range(len(predict)):
    #     print(predict[i])

    print("naive_bayes_classifier:\n")
    print("pca lambda = " + str(pca_lambda))
    f1_score = evaluation(test_class,predict)
    return f1_score
def adaboost_classifier(train_resampled,class_resampled,test_red,test_class):
    f1_score =0.0
    # predict
    param_dist = {"criterion": ["gini", "entropy"],"min_samples_split": randint(2, 50),
                  "max_depth": randint(1, 20),"min_samples_leaf": randint(2, 15),"max_leaf_nodes": randint(2,50)}
    # param_dist = {"criterion": ["gini", "entropy"],"min_samples_split": [10,20,50],
    #               "max_depth": [None,2,5,10],"min_samples_leaf": [2,5,10],"max_leaf_nodes": [None,2,5,10]}
    clf = tree.DecisionTreeClassifier()
    # dt_grid_search = GridSearchCV(clf,param_grid=param_dist, cv=10,scoring='f1')
    dt_randcv = RandomizedSearchCV(clf, param_distributions=param_dist, cv=10,scoring='f1',n_iter=20)
    dt_randcv.fit(train_resampled, class_resampled)
    best_clf = dt_randcv.best_estimator_
    adaboost = AdaBoostClassifier(best_clf,n_estimators=100,random_state=42)
    adaboost.fit(train_resampled, class_resampled)
    predict = adaboost.predict(test_red)
    # predict = AdaBoostClassifier(random_state=0).fit(train_resampled, class_resampled).predict(test_red) 
    # print(predict)
    # for i in range(len(predict)):
    #     print(predict[i])
    print("adaboost classifier:\n")
    print("pca lambda = " + str(pca_lambda))
    f1_score = evaluation(test_class,predict)
    
    print(f1_score)
    return f1_score

def gradient_boosting_clasifier(train_resampled,class_resampled,test_red,test_class):
    print("gradient boosting classifier classifier:\n")
    print("pca lambda = " + str(pca_lambda))
    f1_score =0.0
    param_dist = {"loss": ["deviance", "exponential"],"min_samples_split": randint(2, 50),
                  "max_depth": randint(1, 10),"min_samples_leaf": randint(10, 60),"max_leaf_nodes": randint(10,50), 'n_estimators': randint(50,200)}
    grad_boosting = GradientBoostingClassifier()
    grad_boosting_rs = RandomizedSearchCV(grad_boosting, param_distributions=param_dist, cv=10,scoring='f1',n_iter=50)
    grad_boosting_rs.fit(train_resampled, class_resampled)
    predict = grad_boosting_rs.predict(test_red)
    # predict = RandomForestClassifier (n_estimators=num_trees, random_state=0,min_samples_leaf=50).fit(train_resampled, class_resampled).predict(test_red) 
    for i in range(len(predict)):
        print(predict[i])
    
    f1_score = evaluation(test_class,predict)
    print(f1_score)
    print(grad_boosting_rs.best_params_)
    return f1_score

def evaluation(test_class,predict):
    # print(classification_report(test_class,predict))
    f1_scores = f1_score(test_class, predict,labels = [0,1],average=None)
    return  f1_scores
  
main()
