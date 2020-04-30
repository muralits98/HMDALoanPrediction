from functions import get_data,build_model
import pandas as pd
import numpy as np

#First column

ColName = 'action_taken_name'
data,mapping = get_data(ColName,nr = 500000)
s1 = data[data[ColName] == 0].sample(n = 20000)
s2 = data[data[ColName] == 1].sample(n = 20000)
d = pd.concat([s1,s2])
y = d[ColName]
X = d.drop(columns = [ColName])
build_model(X,y,cross = 10,models = ['RandomForest','SVM','xgb','Logistic'])

ColName = 'denial_reason_name_1'
data,mapping = get_data(ColName,nr = 500000)
s1 = data[data[ColName] == 0].sample(n = np.minimum(data[data[ColName] == 0].shape[0],data[data[ColName] == 1].shape[0]))
s2 = data[data[ColName] == 1].sample(n = np.minimum(data[data[ColName] == 0].shape[0],data[data[ColName] == 1].shape[0]))
d = pd.concat([s1,s2])
y = d[ColName]  
X = d.drop(columns = [ColName])
build_model(X,y,cross = 10,models = ['RandomForest','SVM','xgb','Logistic'])

ColName = 'denial_reason_name_2'
data,mapping = get_data(ColName,nr = 500000)
s1 = data[data[ColName] == 0].sample(n = np.minimum(data[data[ColName] == 0].shape[0],data[data[ColName] == 2].shape[0]))
s2 = data[data[ColName] == 2].sample(n = np.minimum(data[data[ColName] == 0].shape[0],data[data[ColName] == 2].shape[0]))
d = pd.concat([s1,s2])
y = d[ColName]
X = d.drop(columns = [ColName])
build_model(X,y,cross = 10,models = ['RandomForest','SVM','xgb','Logistic'])

ColName = 'denial_reason_name_3'
data,mapping = get_data(ColName,nr = 100000)
s1 = data[data[ColName] == 0].sample(n = np.minimum(data[data[ColName] == 0].shape[0],data[data[ColName] == 2].shape[0]))
s2 = data[data[ColName] == 2].sample(n = np.minimum(data[data[ColName] == 0].shape[0],data[data[ColName] == 2].shape[0]))
d = pd.concat([s1,s2])
y = d[ColName]
X = d.drop(columns = [ColName])
build_model(X,y,cross = 10,models = ['RandomForest','SVM','xgb','Logistic'])

#Pipeline to predict loan/no-loan and then predict reason

