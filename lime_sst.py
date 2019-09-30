import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
from lime import lime_text
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups
from lime.lime_text import LimeTextExplainer



"""Get dataset and split, choose target classes"""
categories = ['alt.atheism', 'soc.religion.christian']  # only choosing 2 categories of 20 total categories
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories) #size: 1079
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories) #size: 717
class_names = ['atheism', 'christian']



"""init vectorizer and fit onto training data, vectorize test data"""
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)



"""init RF classifier and fit onto train data"""
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)



"""Get predictions for test data and calculate metrics"""
pred = rf.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')


"""create end to end pipeline to map IP to OP for lime"""
c = make_pipeline(vectorizer, rf)

"""test model pipeline"""
print(c.predict_proba([newsgroups_test.data[0]]))


"""init lime explainer and init lime class names for neatness"""
explainer = LimeTextExplainer(class_names=class_names)


"""chose sample testdata instance to fit lime and display table, fit lime and show explanation"""
sample = 83
exp = explainer.explain_instance(newsgroups_test.data[sample], c.predict_proba, num_features=6)
print('\nDocument id: %d\n' % sample)
print('\nDocument is : {}\n'.format(newsgroups_test.data[sample]))
print('\nProbability(christian) =', c.predict_proba([newsgroups_test.data[sample]])[0,1])
print('\nTrue class: %s\n\n' % class_names[newsgroups_test.target[sample]])
exp.as_list()



"""Evaluate lime and display expected difference on input removal on lime"""
print('Original prediction:', rf.predict_proba(test_vectors[sample])[0,1])
tmp = test_vectors[sample].copy()
tmp[0,vectorizer.vocabulary_['Posting']] = 0
tmp[0,vectorizer.vocabulary_['Host']] = 0
print('Prediction removing some features:', rf.predict_proba(tmp)[0,1])
print('Difference:', rf.predict_proba(tmp)[0,1] - rf.predict_proba(test_vectors[sample])[0,1])





"""Lime visualizations"""
fig = exp.as_pyplot_figure()
exp.show_in_notebook(text=False)
exp.save_to_file('/tmp/oi.html')
exp.show_in_notebook(text=True)


