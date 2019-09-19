from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import numpy as np
import scipy
from heapq import nlargest
import re
trainingPath = glob.glob("HW1/data/Training/*/*/*.txt")
trainReviews=[]
trainReviews2=[]
testReviews=[]
testReviews2=[]
testReviewsCorrect=[]
numbers = re.compile(r'(\d+)')
accuracyAll=0.0
errors=0

# sorted the file by names and number
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# training function while applying cross validation
def training_c(trainingPath,folder):
	global trainReviews,trainReviews2
	
	for file in sorted(trainingPath, key=numericalSort):
		# the "folder" will be the testing data, so don't train data in the folder
		if folder not in file:
			f=open(file,"r")
			# if the txt file is in the truthful folder, the classfication is 0
			if "truthful" in file:
				reviews=f.read().lower()
				trainReviews.append([reviews,0])
				trainReviews2.append(reviews)
			# if the txt file is in the deceptive folder, the classfication is 1
			elif "deceptive" in file:
				reviews=f.read().lower()
				trainReviews.append([reviews,1])
				trainReviews2.append(reviews)
	f.close()

#training function without cross validation
def training(trainingPath):
	global trainReviews,trainReviews2
	for file in sorted(trainingPath, key=numericalSort):
		f=open(file,"r")
		# if the txt file is in the truthful folder, the classfication is 0
		if "truthful" in file:
			reviews=f.read().lower()
			trainReviews.append([reviews,0])
			trainReviews2.append(reviews)

		# if the txt file is in the deceptive folder, the classfication is 1
		elif "deceptive" in file:
			reviews=f.read().lower()
			trainReviews.append([reviews,1])
			trainReviews2.append(reviews)
	f.close()

# testing function
def testing(testingPath):
	global testReviews,testReviews2
	for file in sorted(testingPath, key=numericalSort):
		f=open(file,"r")
		reviews = f.read().lower()
		# put the reviews into list, and initialize their classfication as -1
		testReviews.append([reviews,-1])
		# put the reviews into list
		testReviews2.append(reviews)
	return testReviews

# Term Frequency Inverse Document Frequency
def tfidf():
	global trainTfidMatrix, testTfidMatrix
	# consider the top max_features =500 ordered by term frequency across the corpus
	# eliminate "stop words"
	vectorizer = TfidfVectorizer(max_features=500,stop_words= 'english',ngram_range=(1,1))
	# Learn vocabulary and idf, return term-document matrix.
	trainTfidMatrix = vectorizer.fit_transform(trainReviews2)
	# featuresName = vectorizer.get_feature_names()
	classList=[]
	# add the original classfication to classList from the trainreviews list
	for i in range(len(trainReviews)):
		classList.append([trainReviews[i][1]])
	classification= np.array(classList)
	# make classList as a vertical matrix
	np.vstack(classification)
	# add a column for the original classification 
	trainTfidMatrix = scipy.sparse.hstack([trainTfidMatrix, classification])
	# Transform documents to document-term matrix.
	testTfidMatrix = vectorizer.transform(testReviews2)
	# initialize classfication for testing data as random number
	randomClass = np.ones((testTfidMatrix.shape[0],1))
	testTfidMatrix = scipy.sparse.hstack([testTfidMatrix, randomClass])
	
	# print testMatrix
	# print testTfidMatrix.nonzero()[1]
	# for col in range(len(featuresName)):
	# 	if testMatrix[0,col]!=0.0:
	# 		print featuresName[col] + " - " + str(testMatrix[0,col])

	return trainTfidMatrix, testTfidMatrix

# compute the cosine distance between v1 and v2
def cosine_distance(v1, v2):
	return np.dot(v1,v2)/(np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))

# k nearest neighbor classfication
def knn_Classifier(K):
	similarity =[]
	mostSimilar=[]
	j=0
	# get cosine similarity of two vectors v1 and v2
	# get the  v1 vector from test matrix
	for v1 in testTfidMatrix.toarray():
		# calculate the similarities between v1 (test)vector and each vector from training data
		for v2 in trainTfidMatrix.toarray():
			# trainTfidMatrix.shape[1] = # columns
			# trainTfidMatrix.shape[0] = # rows
			# eliminate the classfication column
			v1=v1[0:trainTfidMatrix.shape[1]-2]
			v2=v2[0:trainTfidMatrix.shape[1]-2]
			# record the similarities between v1 and each v2
			similarity.append(cosine_distance(v1,v2))
		# get the K nearest point
		mostSimilar=map(similarity.index,nlargest(K,(similarity)))
		tCount=0
		fCount=0
		# reverse the list for following "weighted" purpose
		mostSimilar.reverse()
		for i in range(len(mostSimilar)):
			# mostSimilar[i]: the ith reviews
			if trainMatrix[mostSimilar[i],trainTfidMatrix.shape[1]-1]==0:
				# the more closer the point is, the bigger the tCount is
				# weighted 1*(i+1) 
				tCount=tCount+1*(i+1)
			else:
				fCount=fCount+1*(i+1)
		# compare truthful and deceptive points
		if tCount>fCount:
			testReviews[j][1] = 0
		else:
			testReviews[j][1]=1
		del mostSimilar[:]
		del similarity[:]
		j = j+1
	# print the predictied classfication (result)
	for k in range(len(testReviews)):
		print testReviews[k][1]

def get_accuracy(testingPath):
	i=0
	accuracy=0.0
	global errors, accuracyAll
	for file in sorted(testingPath, key=numericalSort):
		f=open(file,"r")
		reviews = f.read().lower()
		# list the actual classfication
		if "truthful" in file:
			testReviewsCorrect.append([reviews,0])
		elif "deceptive" in file:
			testReviewsCorrect.append([reviews,1])
		# check the accuracy 
		if testReviewsCorrect[i][1] == testReviews[i][1]:
			accuracy = accuracy +1
		else:
			errors = errors+1
		i=i+1 
	accuracyAll=(accuracy/len(testReviews))+accuracyAll
	print "accuracy: " + str(accuracy/len(testReviews))
	

# apply cross validation to find the best K
def cross_validation():
	global errors,accuracyAll
	for K in range(33,433,20):
		for i in range(10):
			folder="Fold"+str(i+1)
			training_c(trainingPath,folder)
			testingPath="HW1/data/Training/"+folder+"/*/*.txt"
			testingPath = glob.glob(testingPath)
			testReviews=testing(testingPath)
			main(K)
			get_accuracy(testingPath)
			del testReviews[:]
			del testReviews2[:]
			del testReviewsCorrect[:]
			del trainReviews[:]
			del trainReviews2[:] 
		# the mean of accuracy 
		print "K= "+ str(K)+ "	accuracy= "+ str(accuracyAll/10)
		print "errors= "+ str(errors)
		errors = 0
		accuracyAll=0.0
# main function
def main(K):
	trainTfidMatrix, testTfidMatrix = tfidf()
	# tocsr() access matrix element
	global trainMatrix, testMatrix
	trainMatrix = trainTfidMatrix.tocsr()
	testMatrix = testTfidMatrix.tocsr()
	# print trainTfidMatrix.shape[0]
	# print trainTfidMatrix.shape[1]
	knn_Classifier(K)

# cross_validation()
training(trainingPath)
testingPath = glob.glob("HW1/data/1547518033_1524422_CS584testHW1/*.txt")
testReviews=testing(testingPath)
main(413)




