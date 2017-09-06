import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot
import pandas as pd
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".



financial_features = [#'salary',
                      #'deferral_payments', 
                      #'total_payments',
                      #'loan_advances',
                      'bonus',
                      #'restricted_stock_deferred',
                      #'deferred_income',
                      'total_stock_value', 
                      'expenses', 
                      #'exercised_stock_options',
                      #'other',
                      #'long_term_incentive',
                      'restricted_stock']
                      #'director_fees']

email_features = ['to_messages',
                  #'email_address',
                  'from_poi_to_this_person',
                  'from_messages',
                  'from_this_person_to_poi',
                  'shared_receipt_with_poi']

poi_label = ['poi'] 


new_features = [#'fraction_from_poi', 
                'fraction_emails_with_poi',
                'fraction_to_poi'

               ]


#We decided to use only new_features after getting a good score while testing.

features_list = poi_label + new_features + financial_features   # You will need to use more features



### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
print len(data_dict)

### Task 2: Remove outliers

data_test = data_dict

def plot_data(data_test):
    
    features = ["salary", "bonus"]
    data = featureFormat(data_test, features)
    for point in data:
        salary = point[0]
        bonus = point[1]
        matplotlib.pyplot.scatter( salary, bonus )
        
    matplotlib.pyplot.xlabel("salary")
    matplotlib.pyplot.ylabel("bonus")
    matplotlib.pyplot.show()
    
    return None
    

#plot_data(data_test)
    
data_dict.pop("TOTAL", 0 )

#plot_data(data_dict)



df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.replace('NaN', np.nan)

df[financial_features] = df[financial_features].fillna(0)
df[email_features] = df[email_features].fillna(df[email_features].median())


data_dict = df.to_dict('index')
    

    


### Task 3: Create new feature(s)

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    
    try:
        fraction =  1.*poi_messages/all_messages
        return fraction
    
    except:
        return "NaN"




def create_features(data_dict):
    
    ### if you are creating any new features, you might want to do that here
    # have some ideas for new features
    for name in data_dict:
        poi_msg_to = data_dict[name]['from_poi_to_this_person']
        all_msg_to = data_dict[name]['to_messages']
        data_dict[name]['fraction_from_poi'] = computeFraction(poi_msg_to, all_msg_to)
        poi_msg_from = data_dict[name]['from_this_person_to_poi']
        all_msg_from = data_dict[name]['from_messages']
        data_dict[name]['fraction_to_poi'] = computeFraction(poi_msg_from, all_msg_from)
        poi_msg_all = poi_msg_to + poi_msg_from
        all_msg_all = all_msg_to + all_msg_from
        data_dict[name]['fraction_emails_with_poi'] = computeFraction(poi_msg_all, all_msg_all)
    return data_dict



### Store to my_dataset for easy export below.

my_dataset = create_features(data_dict)


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

print len(labels)

### Task 4: Try a varity of classifiers


#PCA is tested with all the features. 

#estimators = [('reduce_dim', PCA()), ('nb', GaussianNB())]
#estimators = [('reduce_dim', PCA()), ('lr', LogisticRegression())]
#estimators = [('reduce_dim', PCA()), ('dtc', DecisionTreeClassifier())]

#pipe = Pipeline(estimators)




#pipe = GaussianNB()
#pipe = LogisticRegression()
pipe = DecisionTreeClassifier()



# print all the parameters available
print pipe.get_params().keys()




### Task 5: Tune your classifier to achieve better than .3 precision and recall using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


#params = dict(reduce_dim__n_components=[1,2,3], dtc__min_samples_split=[10,50,100] )
#params = dict(reduce_dim__n_components=[2,3,4], dtc__min_samples_split=[2,3,4,5] )
#params = dict(reduce_dim__n_components=[2,3,4], rc__max_leaf_nodes=[2,3]  ) 





test_classifier(pipe, my_dataset, features_list)


for name, importance in zip(features_list[1:], pipe.feature_importances_):
    print(name, importance)

    
#test different  parameters.    

param_grid = [
  {'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 10, 100]} ]

clf_test = GridSearchCV(pipe, param_grid)
test_classifier(clf_test, my_dataset, features_list)


clf = GaussianNB()
test_classifier(clf, my_dataset, features_list)


### Dump your classifier, dataset, and features_list so
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)

