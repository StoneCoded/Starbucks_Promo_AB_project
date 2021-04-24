from itertools import combinations
from test_results import test_results, score
import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sb


# # Data Wrangling

# load in the data and have a quick gander at it
train_data = pd.read_csv('./training.csv')
print(train_data.dtypes)
train_data.describe()


purchase_data = train_data[train_data['purchase'] == 1].copy()
non_purchase_data = train_data[train_data['purchase'] == 0].copy()

print(f'Total Purchases: {purchase_data.shape[0]}')
print(f'Total Non-Purchases: {non_purchase_data.shape[0]}')

# Random selection of non purchase data to create a balanced training set
random_npd = non_purchase_data.sample(purchase_data.shape[0],random_state = 42).copy()
new_train_data = pd.concat([purchase_data, random_npd]).reset_index(drop=True)

#set X variable and get dummy variables for each feature within
X = new_train_data.iloc[:,3:].copy()
X = pd.get_dummies(data=X, columns=['V1','V4', 'V5','V6','V7'], drop_first = True)
y = new_train_data['purchase']
#Split new_training_data into train and test values for model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('Accuracy: {:.2f}))'.format(model.score(X_test, y_test)))


def promotion_strategy(df):
    '''
    INPUT
    df - a dataframe with *only* the columns V1 - V7 (same as train_data)

    OUTPUT
    promotion_df - np.array with the values
                   'Yes' or 'No' related to whether or not an
                   individual should recieve a promotion
                   should be the length of df.shape[0]

    Ex:
    INPUT: df

    V1	V2	  V3	V4	V5	V6	V7
    2	30	-1.1	1	1	3	2
    3	32	-0.6	2	3	2	2
    2	30	0.13	1	1	4	2

    OUTPUT: promotion

    array(['Yes', 'Yes', 'No'])
    indicating the first two users would recieve the promotion and
    the last should not.
    '''
    df = pd.get_dummies(df, columns=['V1','V4', 'V5','V6','V7'], drop_first=True)
    predicts = model.predict(df)
    pred_map = {0: "No", 1: "Yes"}
    promotion = np.vectorize(pred_map.get)(predicts)

    return promotion

test_results(promotion_strategy)
#test results are very promising.
#With IRR of 0.0183 and NIR of 290.50, this a much more optimistic model

# ## Model 2
#
# Attempt at a large upsample model based on purchases but trained on whether
# or not there is a promotion and purchase.

train_data = pd.read_csv('./training.csv')
train_data.columns = ['ID', 'promotion', 'purchase', 'V1', 'V2', 'V3', 'V4',
        'V5', 'V6','V7']
train_data['pro_pur'] = train_data.apply(lambda x: 1 if
                                         (x['promotion'] == 'Yes') &
                                         (x['purchase'] == 1) else 0, axis=1)
train_data['promotion'] = train_data['promotion'].apply(lambda x: 1 if x
                                                        == 'Yes' else 0)
train_data.head()


def upsample(df):
    '''
    INPUT:
    dataframe

    OUTPUT:
    Dataframe with resampled purchase values for bigger
    and hopefully more balanced and precise training/prediction
    '''
    purchases = df[df.purchase == 1]
    non_purchases = df[df.purchase == 0]

    purchases_upsampled = resample(purchases, replace=True,
                        # sample with replacement
                        n_samples = len(non_purchases),
                        # match number in majority class
                        random_state = 42)
                        # reproducible results

    df = pd.concat([purchases, non_purchases])
    return df

new_train_data = upsample(train_data)

#Split new_training_data into train and test values for model
X = new_train_data.iloc[:,3:10].copy()
y = new_train_data['pro_pur']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)
#Initiate and fit classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

#predict and score model
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('Accuracy: {:.4f}))'.format(model.score(X_test, y_test)))

#[[16755 0]
#[152   0]] suggests this didn't go according to plan

def promotion_strategy(df):
    '''
    INPUT
    df - a dataframe with *only* the columns V1 - V7 (same as train_data)

    OUTPUT
    promotion_df - np.array with the values
                   'Yes' or 'No' related to whether or not an
                   individual should recieve a promotion
                   should be the length of df.shape[0]

    Ex:
    INPUT: df

    V1	V2	  V3	V4	V5	V6	V7
    2	30	-1.1	1	1	3	2
    3	32	-0.6	2	3	2	2
    2	30	0.13	1	1	4	2

    OUTPUT: promotion

    array(['Yes', 'Yes', 'No'])
    indicating the first two users would recieve the promotion and
    the last should not.
    '''
    y_pred = model.predict(df)
    pred_map = {0: "No", 1: "Yes"}
    promotion = np.vectorize(pred_map.get)(y_pred)

    return promotion


# In[116]:


# This will test your results, and provide you back some information
# on how well your promotion_strategy will work in practice

test_results(promotion_strategy)
#With an IRR of 0 and NIR of -0.75 my initial suspicions that this approach was
#flawed was justified.
