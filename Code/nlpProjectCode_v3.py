import os
from collections import defaultdict
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
import pickle
import matplotlib.pyplot as plt

def saveToPickle(dict,name):
	pickle.dump(dict,open(name,"wb"))
	return

def loadFromPickle(filename):
	dictLoad=pickle.load(open(filename,"rb"))
	return copy.deepcopy(dictLoad)


if __name__ == '__main__':
	#Change the paths to match up with your corresponding path
	destPath="/Users/nugthug/PycharmProjects/Party-Affiliation/PickleDicts/"
	trainPath="/Users/nugthug/Documents/cmpsci/585/f2016/Project/convote/data_stage_one/training_set/"
	testPath="/Users/nugthug/Documents/cmpsci/585/f2016/Project/convote/data_stage_one/test_set/"
	devPath="/Users/nugthug/Documents/cmpsci/585/f2016/Project/convote/data_stage_one/development_set/"
	trainFileNames=os.listdir(trainPath)
	testFileNames=os.listdir(testPath)
	devFileNames=os.listdir(devPath)
	train_labels=np.array([])
	test_labels=np.array([])
	vectorizer=CountVectorizer()
	trainCorpus=[]
	for fn in trainFileNames:
		train_labels=np.append(train_labels,fn[19])
		f=open(os.path.join(trainPath,fn))
		trainCorpus.append(f.read())
		f.close()
	# for fn in devFileNames:
	# 	train_labels=np.append(train_labels,fn[19])
	# 	f=open(os.path.join(devPath,fn))
	# 	trainCorpus.append(f.read())
	# 	f.close()
	testCorpus=[]
	print "Number of Documents in Training Set: "+str(len(trainFileNames))
	for fn in testFileNames:
		test_labels=np.append(test_labels,fn[19])
		f=open(os.path.join(testPath,fn))
		testCorpus.append(f.read())
		f.close()
	text_pipeline=Pipeline([('vect', CountVectorizer()),
							('tfidf', TfidfTransformer(use_idf=False)),
							('clf',  MultinomialNB(alpha=0))])
	text_pipeline=text_pipeline.fit(trainCorpus,train_labels)
	predicted=text_pipeline.predict(testCorpus)
	print "NB Accuracy: "+str(np.mean(predicted==test_labels))

	wordFreqTrain=text_pipeline.named_steps['vect'].fit_transform(trainCorpus).toarray()
	wordNameDict=text_pipeline.named_steps['vect'].get_feature_names()
	wordFreqArray=np.sum(wordFreqTrain, axis=0)

	print "Number of training Tokens: "+str(sum(wordFreqArray))
	print "Size of training Vocabulary: "+str(len(wordNameDict))
	#print "Top 20 words below: "
	wordCountDict=dict(zip(wordNameDict,wordFreqArray))
	wordCountList=sorted(wordCountDict,key=wordCountDict.get, reverse=True)
	print "Top 100 words below: "
	for i in range(100):
		print str(wordCountList[i]) +" : "+ str(wordCountDict[str(wordCountList[i])])
	for i in range(200):
		print str(wordCountList[-i])
	#print wordCountList
	length=len(wordCountList)
	rank=range(1,length+1)
	log_rank=[math.log(x) for x in rank]
	log_freq=[math.log(wordCountDict[word]) for word in wordCountList]
	fig = plt.figure()
	ax = plt.gca()
	ax.scatter( log_rank, log_freq, linewidth=2)
	plt.xlabel("log(rank)")
	plt.ylabel("log(frequency)")
	plt.title("Zipfs Law on Training Corpus")
	#plt.show()
	plt.savefig("zipfslaw.pdf")

	params={"vect__ngram_range":[(1,1),(1,2)],
			"tfidf__use_idf":(True,False),
			"clf__alpha":[0,.1,.5,1,1.5,2,10,100]}
	crossValidationClf=GridSearchCV(text_pipeline,params,n_jobs=-1)
	crossValidationClf=crossValidationClf.fit(trainCorpus,train_labels)
	best_parameters, score, _ = max(crossValidationClf.grid_scores_ , key=lambda x: x[1])
	for param_name in sorted(params.keys()):
		print("%s: %r" % (param_name, best_parameters[param_name]))
	predicted=crossValidationClf.predict(testCorpus)
	print "NB CV Accuracy: "+str(score)
	print "NB CV Test Accuracy: "+str(np.mean(predicted==test_labels))
	saveToPickle(crossValidationClf.grid_scores_,destPath+"NBGS.p")

	text_pipeline=Pipeline([('vect', CountVectorizer()),
							('tfidf', TfidfTransformer()),
							('clf',  SGDClassifier())])
	text_pipeline=text_pipeline.fit(trainCorpus,train_labels)
	predicted=text_pipeline.predict(testCorpus)
	print "SVM Accuracy: "+str(np.mean(predicted==test_labels))

	print(metrics.classification_report(test_labels,predicted))
	print(metrics.confusion_matrix(test_labels,predicted))
	params={"vect__ngram_range":[(1,1),(1,2)],
			"tfidf__use_idf":(True,False),
			"clf__alpha":[.0001,.01,.1,1,10,100],
			"clf__loss":["hinge","log","perceptron"],
			"clf__penalty":["none","l1","l2"]}
	crossValidationClf=GridSearchCV(text_pipeline,params,n_jobs=-1)
	crossValidationClf=crossValidationClf.fit(trainCorpus,train_labels)
	best_parameters, score, _ = max(crossValidationClf.grid_scores_ , key=lambda x: x[1])
	for param_name in sorted(params.keys()):
		print("%s: %r" % (param_name, best_parameters[param_name]))
	predicted=crossValidationClf.predict(testCorpus)
	print "SVM CV Accuracy: "+str(score)
	print "SVM CV Test Accuracy: "+str(np.mean(predicted==test_labels))
	saveToPickle(crossValidationClf.grid_scores_,destPath+"sdgClassifierGS.p")