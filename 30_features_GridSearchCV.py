"""
Code Explanation

1. Import class and 30 numerical features.
2. Split the data into train and test data sets
3. Use GridSearchCV to find optimal parameters to apply for SVC
4. Evaluate the result
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm



col_names = ['ID', 'class']
for j in range(1, 31):
	col_names.append('feat' + str(j))

df = pd.read_csv('data.txt', names = col_names)

df.drop('ID', axis = 1, inplace = True)


df['class'] = df['class'].replace('M', 0)
df['class'] = df['class'].replace('B', 1)

df_X = df.drop(['class'], axis = 1)

df_y = df['class']

#Converting pandas object to numpy object
X = df_X.values
y = df_y.values

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Set the parameters by cross-validation
parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}]


print '# Tuning hyper-parameters'
print '\n'

clf = GridSearchCV(svm.SVC(decision_function_shape = 'ovr'), parameters, cv = 8)
clf.fit(X_train, y_train)

print 'Best parameters set found on development set:'
print '\n'
print clf.best_params_
print '\n'
print 'Grid scores on training set:'
print '\n'
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print '\n'

print 'Detailed classification report:'
print '\n'
print 'The model is trained on the full development set."'
print 'The scores are computed on the full evaluation set.'
print '\n'
y_true, y_pred = y_test, clf.predict(X_test)
print classification_report(y_true, y_pred)
print '\n'
