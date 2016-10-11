#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import math
import  numpy as np


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features
outliers_keys = ['TOTAL']  # data point with key of "TOTAL" should be removed

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
##################################################################################
### Task 2: Remove outliers
for i in outliers_keys:
    for k in data_dict.keys():
        if k == i :
            del data_dict[i]
###################################################################################
### Task 3: Create new feature(s)
#####  calaculating the new features and add them to the data_dic
for k, v in data_dict.iteritems():
    # adding from_poi_to_this_person_perc feature
    from_poi_to_this_person_perc = float(v['from_poi_to_this_person'])/float(v['to_messages'])
    ### Replacing nan values with zero
    if not math.isnan(from_poi_to_this_person_perc) :
        v['from_poi_to_this_person_perc']  = from_poi_to_this_person_perc
    else :
         v['from_poi_to_this_person_perc'] =0.

     # adding from_poi_to_this_person_perc feature
    from_this_person_to_poi_perc = float(v['from_this_person_to_poi'])/float(v['from_messages'])
     ### Replacing nan values with zero
    if not math.isnan (from_this_person_to_poi_perc ):
        v['from_this_person_to_poi_perc'] = from_this_person_to_poi_perc
    else:
         v['from_this_person_to_poi_perc'] = 0.
##### updating featuresList
##### using SelectKBest(f_classif, k=7)
##### =>  Highest 7 features are exercised_stock_options','total_stock_value','bonus','salary','deferred_income',
##### => 'long_term_incentive','restricted_stock'
features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus', 'deferred_income']

##### Adding the new created features to the list
features_list.append('from_poi_to_this_person_perc')
features_list.append('from_this_person_to_poi_perc')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#### Scaling features in range from 0 to 1
from  sklearn.preprocessing import  MinMaxScaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.cross_validation import train_test_split

def classifier_eval(clf,input_features, input_labels, n_component=2, pca_applied=False, fold = 10):
    '''
    This function use kfold cross validation and calculate the average precision, recall & f-score for the given data
    pca_applied--> is a boolean flag that indicates PCA were applied to the training set or not
    :return:
    '''
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.decomposition import PCA

    accuracy_scores=[]
    precision_scores=[]
    recall_scores =[]
    f1_scores = []

    skf = StratifiedKFold(input_labels, fold,random_state = 42, shuffle=True)
    for train_idx, test_idx in skf:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        if pca_applied:
            # apply pca on training_features & testing features before evaluating the classifier
            pca = PCA(n_components=n_component)
            features_train = pca.fit_transform(features_train)
            features_test = pca.fit_transform(features_test)

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        precision_scores.append(precision_score(labels_test,predictions))
        recall_scores.append(recall_score(labels_test,predictions))
        f1_scores.append(f1_score(labels_test,predictions))
        accuracy_scores.append(accuracy_score(labels_test,predictions))

    accuracy_score_avg = np.mean(accuracy_scores)
    precision_score_avg = np.mean(precision_scores)
    recall_score_avg = np.mean(recall_scores)
    f1_score_avg = np.mean(f1_scores)
    print "Accuracy: ", accuracy_score_avg
    print "Precision: ", precision_score_avg
    print "Recall: ", recall_score_avg
    print "F1: ", f1_score_avg
    return  accuracy_score_avg, precision_score_avg, recall_score_avg, f1_score_avg

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# Provided to give you a starting point. Try a variety of classifiers.
print "Features list:exercised_stock_options, total_stock_value, bonus, deferred_income , from_poi_to_this_person_perc, " \
      "from_this_person_to_poi_perc"
print ""

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_scaled,labels)

classifier_eval(clf,features_scaled, labels)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

