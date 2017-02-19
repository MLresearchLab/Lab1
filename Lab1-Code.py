# Created by Dickson Kwong & Kevin Figueroa
# MLresearchLab.com
# Lab 1 Random Forest
# Python 2.7 run inside of Jupyter Notebook
# [Question to ask ML]
# Determine based on traffic from OpenDNS.com if a user has been primarily visting work related sites or not

#MIT License

#Copyright (c) 2017 Dickson Kwong & Kevin Figueroa

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

# Pythons scientific libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
# You need this import division for python 2.7 to display percentages correctly
from __future__ import division

# The line below will only work in Jupyter Notebook -> purpose: it shows plots inside Jupyter Notebook
%matplotlib inline

# Create a dataframe from panda and call it 'df' and make it read in the sanatized / prepared data CSV file 'Mod2.csv' 
df = pd.read_csv(r'Mod2.csv',encoding='latin-1')

# Ask the dataframe to show information about the imported 'Mod2.csv' file
df.info()

# Ask the dataframe to return the results of the first 5 rows but with all columns
df.head()

# Create a new row called 'is_train' inside the dataframe 'df' and randomize a number between 0 and 1 and if its less than or equal to .75 make it 0. 
# 0 = training data
# 1 = test data
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

# Ask the dataframe to return the results of the first 5 rows but with all columns 
# Notice that the dataframe now has 'is_train' as the last column
df.head()

# Set the variable train to be a dataframe with all the rows that have the dataframe 'df' field 'is_train' true while
# Set the variable test to be a dataframe with all the rows that have the dataframe field 'is_train' false
train, test = df[df['is_train']==True], df[df['is_train']==False]

# Print the number of training data & test data you have based on the column 'is_train'
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

# Create a dataframe called 'features' that has the first 58 columns which are the "Categories of website classifications" ending with "Webmail". This dataframe purposely ignores the 'is_train' and 'WorkRelated' because those are not values the ML should be using in the prediction.
features = df.columns[:58]

# Prints out the values inside features
features

# Selecting RandomForestClassifier
clf = RandomForestClassifier(n_jobs=2)

#Only call factorize when the value in the column 'WorkRelated' is a string not a numeral
#y = pd.factorize(train['WorkRelated'])[0]
#y is not neccessary because 'WorkRelated' is not a string
#clf.fit(train[features], y)

# Using the classifier we fit the data using dataframe 'train' that will only use the 'features' columns
# against the dataframe 'train' 'WorkRelated' column in order to train the classifier how they relate.
clf.fit(train[features], train['WorkRelated'])

# Apply the classifier we trained to the test data
clf.predict(test[features])

# Show the predicated probablity of the first 10 results
clf.predict_proba(test[features])[0:10]

# Set the variable preds to the values of the prediction
preds = clf.predict(test[features])

# Show the first 5 predictions 
preds[0:5]

# Show the first 5 actual 'WorkRelated' values to compare against the 5 predictions
test['WorkRelated'].head()

# Show all the actual 'WorkRelated' values in all of the test data
test.ix[:200,[x for x in df.columns if 'WorkRelated' in x]]

# Create a dataframe 'rf' that has the listing of all predictions actual 'WorkRelated' values 
rf = pd.DataFrame(list(zip(preds[0:200],test['WorkRelated'] )), columns=['predicted','actual'])
rf

# Create a row in dataframe 'rf' to mark '1' for true if both the predicted and actual were correct or a '0' for false
rf['correct'] = rf.apply(lambda r: 1 if r['predicted']==r['actual'] else 0, axis=1)
rf

# Get a percentage of accuracy of the ML by dividing the sum of the correctly guessed predictions against the count of all the predictions
rf['correct'].sum() / rf['correct'].count()

# Confusion matrix is: 
#  anything on the diagonal was classified correctly
#  anything off the diagonal was classified incorrectly.
pd.crosstab(test['WorkRelated'], preds, rownames=['Actual Work Related'], colnames=['Predicted Work Related'])

# List how important each feature was
list(zip(train[features], clf.feature_importances_))