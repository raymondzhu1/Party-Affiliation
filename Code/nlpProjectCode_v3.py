import os
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

if __name__ == '__main__':
	#Change the paths to match up with your corresponding path
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
	for fn in devFileNames:
		train_labels=np.append(train_labels,fn[19])
		f=open(os.path.join(devPath,fn))
		trainCorpus.append(f.read())
		f.close()
	testCorpus=[]
	for fn in testFileNames:
		test_labels=np.append(test_labels,fn[19])
		f=open(os.path.join(testPath,fn))
		testCorpus.append(f.read())
		f.close()
	text_pipeline=Pipeline([('vect', CountVectorizer()),
							('tfidf', TfidfTransformer()),
							('clf',  MultinomialNB(alpha=0))])
	text_pipeline=text_pipeline.fit(trainCorpus,train_labels)
	predicted=text_pipeline.predict(testCorpus)
	print "NB Accuracy: "+str(np.mean(predicted==test_labels))


	params={"vect__ngram_range":[(1,1),(1,2)],
			"tfidf__use_idf":(True,False),
			"clf__alpha":[0,.1,.5,1,1.5,2]}
	crossValidationClf=GridSearchCV(text_pipeline,params,n_jobs=-1)
	crossValidationClf=crossValidationClf.fit(trainCorpus,train_labels)
	best_parameters, score, _ = max(crossValidationClf.grid_scores_ , key=lambda x: x[1])
	for param_name in sorted(params.keys()):
		print("%s: %r" % (param_name, best_parameters[param_name]))
	predicted=crossValidationClf.predict(testCorpus)
	print "NB CV Accuracy: "+str(score)
	print "NB CV Test Accuracy: "+str(np.mean(predicted==test_labels))


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
			"clf__alpha":(1e-2,1e-3)}
	crossValidationClf=GridSearchCV(text_pipeline,params,n_jobs=-1)
	crossValidationClf=crossValidationClf.fit(trainCorpus,train_labels)
	best_parameters, score, _ = max(crossValidationClf.grid_scores_ , key=lambda x: x[1])
	for param_name in sorted(params.keys()):
		print("%s: %r" % (param_name, best_parameters[param_name]))
	predicted=crossValidationClf.predict(testCorpus)
	print "SVM CV Accuracy: "+str(score)
	print "SVM CV Test Accuracy: "+str(np.mean(predicted==test_labels))