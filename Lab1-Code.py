import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

%matplotlib inline

df = pd.read_csv(r'Mod2.csv',encoding='latin-1')

df.info()

df.head()

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

df.head()

train, test = df[df['is_train']==True], df[df['is_train']==False]

print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

features = df.columns[:58]

features

clf = RandomForestClassifier(n_jobs=2)

#Only call factorize when the value in the column 'WorkRelated' is a string not a numeral
#y = pd.factorize(train['WorkRelated'])[0]
#y is not neccessary because 'WorkRelated' is not a string
#clf.fit(train[features], y)

clf.fit(train[features], train['WorkRelated'])

clf.predict(test[features])

clf.predict_proba(test[features])[0:10]
preds = clf.predict(test[features])
preds[0:5]

test['WorkRelated'].head()

test.ix[:50,[x for x in df.columns if 'WorkRelated' in x]]

rf = pd.DataFrame(list(zip(preds[0:50],test['WorkRelated'] )), columns=['predicted','actual'])
rf

rf['correct'] = rf.apply(lambda r: 1 if r['predicted']==r['actual'] else 0, axis=1)
rf

from __future__ import division
rf['correct'].sum() / rf['correct'].count()

#confusion matrix is: 
# anything on the diagonal was classified correctly
# anything off the diagonal was classified incorrectly.
pd.crosstab(test['WorkRelated'], preds, rownames=['Actual Work Related'], colnames=['Predicted Work Related'])

#List how important each feature was
list(zip(train[features], clf.feature_importances_))