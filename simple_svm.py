"""
This code throws training data into SVC without tuning any parameters. 
kernel is rbf by default.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.metrics import confusion_matrix

col_names = ['ID', 'class']
for j in range(1, 31):
	col_names.append('feat' + str(j))

df = pd.read_csv('data.txt', names = col_names)

df.drop('ID', axis = 1, inplace = True)


df['class'] = df['class'].replace('M', 0)
df['class'] = df['class'].replace('B', 1)

df_X = df.drop(['class'], axis = 1)
df_y = df['class']

X = df_X.values
y = df_y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

clf = svm.SVC(kernel='linear', C=1,gamma=1)
clf.fit(X_train, y_train)  

y_true, y_pred = y_test, clf.predict(X_test)

print(classification_report(y_true, y_pred))

print confusion_matrix(y_true, y_pred)