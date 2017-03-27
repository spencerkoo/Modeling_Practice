# General use imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, auc
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score


# load in the feature data
train = pd.read_csv('features.csv')
test = pd.read_csv('features_test.csv')


train['Type'] = 'Train'
test['Type'] = 'Test'
df = pd.concat([train, test], axis = 0)


df.drop(['email_open', 'form_submit', 'email_click_thru', 'cust_sup', 'page_view', 'web_view'], axis = 1, inplace = True)


ID_col = ['id']
target_col = ['has_purchased']
category_cols = ['has_email_open', 'has_form_submit', 'has_email_click_thru', 'has_cust_sup', 'has_page_view', 'has_web_view']
num_cols = ['total_actions', 'total_form_submit', 'total_email_click_thru', 'total_cust_sup', 'total_page_view',
            'total_web_view', 'days_as_user']
other_col = ['Type']


num_category_cols = num_cols + category_cols


for var in category_cols:
    number =LabelEncoder()
    df[var] = number.fit_transform(df[var].astype('str'))

df['has_purchased'] = number.fit_transform(df['has_purchased'].astype('str'))

train = df[df['Type'] == 'Train']
test = df[df['Type'] == 'Test']

train['is_train'] = np.random.uniform(0, 1, len(train)) <= 0.75
Train, Validate = train[train['is_train'] == True], train[train['is_train'] == False]



features = list(set(list(df.columns)) - set(ID_col) - set(target_col) - set(other_col))



x_train = Train[list(features)].values
y_train = Train['has_purchased'].values
x_validate = Validate[list(features)].values
y_validate = Validate['has_purchased'].values
x_test = test[list(features)].values



random.seed(100)
rf = RandomForestClassifier(n_estimators = 1000)
forest = rf.fit(x_train, y_train)



scores = cross_val_score(rf, x_train, y_train, verbose = 5)
print(scores.mean())



status = rf.predict_proba(x_validate)
fpr, tpr, _ = roc_curve(y_validate, status[:, 1])
roc_auc = auc(fpr, tpr)
print roc_auc



final_status = rf2.predict_proba(x_test)
test['has_purchased'] = final_status[:, 1]
test.to_csv('test_results.csv', columns = 'id')
print final_status