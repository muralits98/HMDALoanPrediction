from __future__ import division
from functions import get_data,build_model
import pandas as pd
import numpy as np
from HyperParameterTune import tune_model#,GuidedTuneModel
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm

"""

Building model to predict the acceptance/denial

"""
# ColName = 'action_taken'
# data = get_data(ColName,nr = 100000)
# s1 = data[data[ColName] == 1].sample(n = np.minimum(data[data[ColName] == 1].shape[0],data[data[ColName] == 3].shape[0]))
# s2 = data[data[ColName] == 3].sample(n = np.minimum(data[data[ColName] == 1].shape[0],data[data[ColName] == 3].shape[0]))
# d = pd.concat([s1,s2])
# y = d[ColName]
# X = d.drop(columns = [ColName])
# # build_model(X,y,name = 'acceptance_denial_model',cross = 10,models = ['nvb','RandomForest','xgb','Logistic'])

# # GuidedTuneModel(X,y)
# tune_model(X,y,name = 'acceptance_denial_tuned',n_it = 50, models = ['RandomForest','xgb','Logistic'])

#########################################################################################################


"""

Building model to predict the denial reason

"""
# ColName = 'denial_reason_1'
# data = get_data(ColName,nr = 100000,res = 1)
# s1 = data[data[ColName] == 1].sample(n = np.minimum(data[data[ColName] == 1].shape[0],data[data[ColName] == 3].shape[0]))
# s2 = data[data[ColName] == 3].sample(n = np.minimum(data[data[ColName] == 1].shape[0],data[data[ColName] == 3].shape[0]))
# d = pd.concat([s1,s2])
# y = d[ColName]  
# X = d.drop(columns = [ColName])
# # build_model(X,y,name = 'denial_reason_model',cross = 10,models = ['nvb','RandomForest','xgb','Logistic'])

# # GuidedTuneModel(X,y)
# tune_model(X,y,name = 'denial_reason_tuned',n_it = 50, models = ['RandomForest','xgb','Logistic'])

#################################################################################################

"""
The above lines of code have been commented out because the models built have been saved.

Predict if acceptance/denial and filter that to predict denial reason

"""

# filename = 'acceptance_denial_tuned.sav'
# accden = pickle.load(open(filename, 'rb'))
# ColName = 'action_taken'
# data = get_data(ColName,nr = 100000)
# X = data.drop(columns = [ColName])
# need = accden.predict(X)
# X['accden'] = need
# X  = X[X['accden'] == 3]
# X = X.drop(columns = ['accden'])
# filename = 'denial_reason_tuned.sav'
# denreason = pickle.load(open(filename, 'rb'))
# reason = denreason.predict(X)

# plt.hist(need)
# plt.show()
# plt.hist(reason)
# plt.show()

#####################################################################################################

"""

Studying Bias and Fairness in the data with the next 100000 datapoints in 2017

"""

filename = 'acceptance_denial_tuned.sav'
accden = pickle.load(open(filename, 'rb'))
ColName = 'action_taken'
data = get_data(ColName,nr = 100000,year = 2017,sk = 100000)
original = np.array(data[ColName])
X = data.drop(columns = [ColName])
need = accden.predict(X)
X['accden'] = need
X1  = X[X['accden'] == 3]
X1 = X1.drop(columns = ['accden'])
filename = 'denial_reason_tuned.sav'
denreason = pickle.load(open(filename, 'rb'))
reason = denreason.predict(X1)
# X['accden'] = need
X = X.drop(columns = ['accden'])
# print(np.sum(original == need))
# print(np.sum(original == need) * (1/len(original)) ) #Accuracy is Roughly 52%
ori = X['applicant_sex']
old_pred = need

#########################

filename = 'acceptance_denial_tuned.sav'
accden = pickle.load(open(filename, 'rb'))
ColName = 'action_taken'
data = get_data(ColName,nr = 100000,year = 2017,sk = 100000)
X = data
male_acceptance_rate = (X[(X['applicant_sex'] == 1) & (X['action_taken'] == 1)].shape[0])/(X[(X['applicant_sex'] == 1)].shape[0])
female_acceptance_rate = (X[(X['applicant_sex'] == 2) & (X['action_taken'] == 1)].shape[0])/(X[(X['applicant_sex'] == 2)].shape[0])

print("Male acceptance rate is ",male_acceptance_rate)
print("Female acceptance rate is ",female_acceptance_rate)
##########################

# random.seed(1)
filename = 'acceptance_denial_tuned.sav'
accden = pickle.load(open(filename, 'rb'))
ColName = 'action_taken'
data = get_data(ColName,nr = 100000,year = 2017,sk = 100000)
original = np.array(data[ColName])
X = data.drop(columns = [ColName])
X = X[X['applicant_sex'] == 1]
need = accden.predict(X)
X['applicant_sex'] = [2 for j in range(X.shape[0])]
print("Ratio of males in the data =",X.shape[0]/83323)
monte = accden.predict(X)
X['new_pred'] = monte
X['original_sex'] = ori
X['old_pred'] = need
female_to_male_accept = X[(X['applicant_sex'] == 1) & (X['original_sex'] == 2) & (X['new_pred'] == 1)].shape[0]
male_to_female_accept = X[(X['applicant_sex'] == 2) & (X['original_sex'] == 1) & (X['new_pred'] == 1)].shape[0]
male_to_female_reject = X[(X['applicant_sex'] == 2) & (X['original_sex'] == 1) & (X['new_pred'] == 3)].shape[0]
female_to_male_reject = X[(X['applicant_sex'] == 1) & (X['original_sex'] == 2) & (X['new_pred'] == 3)].shape[0]

total_female_to_male = X[(X['applicant_sex'] == 1) & (X['original_sex'] == 2)].shape[0]
total_male_to_female = X[(X['applicant_sex'] == 2) & (X['original_sex'] == 1)].shape[0]

no_change = X[(X['applicant_sex'] ==  X['original_sex'])].shape[0]

print(female_to_male_accept)
print(male_to_female_accept)
print(male_to_female_reject)
print(female_to_male_reject)

male_to_female_accept_prob =(male_to_female_accept/(male_to_female_accept + male_to_female_reject))
male_to_female_reject_prob =(male_to_female_reject/(male_to_female_accept + male_to_female_reject))
no_change_prob = (no_change/X.shape[0])

print("The probability of male_to_female_accept_prob = ",male_to_female_accept_prob)
print("The probability of male_to_female_reject_prob = ",male_to_female_reject_prob)

##########################

filename = 'acceptance_denial_tuned.sav'
accden = pickle.load(open(filename, 'rb'))
ColName = 'action_taken'
data = get_data(ColName,nr = 100000,year = 2017,sk = 100000)
original = np.array(data[ColName])
X = data.drop(columns = [ColName])
X = X[X['applicant_sex'] == 2]
need = accden.predict(X)
X['applicant_sex'] = [1 for j in range(X.shape[0])]
print("Ratio of females in the data =",X.shape[0]/83323)

monte = accden.predict(X)
X['new_pred'] = monte
X['original_sex'] = ori
X['old_pred'] = need
female_to_male_accept = X[(X['applicant_sex'] == 1) & (X['original_sex'] == 2) & (X['new_pred'] == 1)].shape[0]
male_to_female_accept = X[(X['applicant_sex'] == 2) & (X['original_sex'] == 1) & (X['new_pred'] == 1)].shape[0]
male_to_female_reject = X[(X['applicant_sex'] == 2) & (X['original_sex'] == 1) & (X['new_pred'] == 3)].shape[0]
female_to_male_reject = X[(X['applicant_sex'] == 1) & (X['original_sex'] == 2) & (X['new_pred'] == 3)].shape[0]

total_female_to_male = X[(X['applicant_sex'] == 1) & (X['original_sex'] == 2)].shape[0]
total_male_to_female = X[(X['applicant_sex'] == 2) & (X['original_sex'] == 1)].shape[0]

no_change = X[(X['applicant_sex'] ==  X['original_sex'])].shape[0]
print(female_to_male_accept)
print(male_to_female_accept)
print(male_to_female_reject)
print(female_to_male_reject)

female_to_male_accept_prob =(female_to_male_accept/(female_to_male_accept + female_to_male_reject))
female_to_male_reject_prob =(female_to_male_reject/(female_to_male_accept + female_to_male_reject))
no_change_prob = (no_change/X.shape[0])

print("The probability of female_to_male_accept_prob = ",female_to_male_accept_prob)
print("The probability of female_to_male_reject_prob = ",female_to_male_reject_prob)

##########################

female_to_male_accept_prob = []
male_to_female_accept_prob = []
male_to_female_reject_prob = []
female_to_male_reject_prob = []
no_change_prob = []
for i in tqdm(range(1,100)): #10257
    random.seed(i)
    X['applicant_sex'] = [random.choice([1,2,3,4]) for j in range(len(original))]
    monte = accden.predict(X)
    X['new_pred'] = monte
    X['original_sex'] = ori
    X['old_pred'] = need
    female_to_male_accept = X[(X['applicant_sex'] == 1) & (X['original_sex'] == 2) & (X['new_pred'] == 1)].shape[0]
    male_to_female_accept = X[(X['applicant_sex'] == 2) & (X['original_sex'] == 1) & (X['new_pred'] == 1)].shape[0]
    male_to_female_reject = X[(X['applicant_sex'] == 2) & (X['original_sex'] == 1) & (X['new_pred'] == 3)].shape[0]
    female_to_male_reject = X[(X['applicant_sex'] == 1) & (X['original_sex'] == 2) & (X['new_pred'] == 3)].shape[0]

    total_female_to_male = X[(X['applicant_sex'] == 1) & (X['original_sex'] == 2)].shape[0]
    total_male_to_female = X[(X['applicant_sex'] == 2) & (X['original_sex'] == 1)].shape[0]

    no_change = X[(X['applicant_sex'] ==  X['original_sex'])].shape[0]
    try:
        female_to_male_accept_prob.append(female_to_male_accept/(female_to_male_accept + female_to_male_reject))
        male_to_female_accept_prob.append(male_to_female_accept/(male_to_female_accept + male_to_female_reject))
        male_to_female_reject_prob.append(male_to_female_reject/(male_to_female_accept + male_to_female_reject))
        female_to_male_reject_prob.append(female_to_male_reject/(female_to_male_accept + female_to_male_reject))
        no_change_prob.append(no_change/X.shape[0])
    except:
        col = ['old_pred','new_pred','original_sex']
        X = X.drop(columns = col)
        continue
    col = ['old_pred','new_pred','original_sex']
    X = X.drop(columns = col)  
    # print(i)
print(no_change_prob,female_to_male_accept_prob,female_to_male_reject_prob,male_to_female_accept_prob,male_to_female_reject_prob)
print(np.mean(no_change_prob),np.mean(female_to_male_accept_prob),np.std(female_to_male_accept_prob),np.mean(female_to_male_reject_prob),np.std(female_to_male_reject_prob),np.mean(male_to_female_accept_prob),np.std(male_to_female_accept_prob),np.mean(male_to_female_reject_prob),np.std(male_to_female_reject_prob))

plt.plot(female_to_male_accept_prob, label = 'female_to_male_accept_prob')
plt.plot(male_to_female_accept_prob, label = 'male_to_female_accept_prob')
plt.plot(male_to_female_reject_prob, label = 'male_to_female_reject_prob')
plt.plot(female_to_male_accept_prob, label = 'female_to_male_accept_prob')
plt.plot(female_to_male_reject_prob, label = 'female_to_male_reject_prob')
plt.title('probabilities of said events')
plt.legend()
# plt.show()
plt.savefig('probabilitiesofsaidevents.png')
plt.show(block=True)

plt.hist(female_to_male_accept_prob)
plt.title('female_to_male_accept_prob')
# plt.show()
plt.savefig('female_to_male_accept_prob.png')
plt.show(block=True)
plt.hist(male_to_female_accept_prob)
plt.title('male_to_female_accept_prob')

# plt.show()
plt.savefig('male_to_female_accept_prob.png')
plt.show(block=True)

plt.hist(male_to_female_reject_prob)
plt.title('male_to_female_reject_prob')

# plt.show()
plt.savefig('male_to_female_reject_prob.png')
plt.show(block=True)

plt.hist(female_to_male_reject_prob)
plt.title('female_to_male_reject_prob')

# plt.show()
plt.savefig('female_to_male_reject_prob.png')
plt.show(block=True)


# plt.hist(X['applicant_sex'])
# sns.catplot(x=[1,2,3,4], y=X.groupby("applicant_sex").count().accden, data=X)
# plt.show()


# plt.hist(need)
# plt.show()
# plt.hist(reason)
# plt.show()

# tune_model(X,y,n_it = 50,models = ['RandomForest','xgb','Logistic'])

# ColName = 'denial_reason_2'
# data = get_data(ColName,nr = 100000)
# s1 = data[data[ColName] == 1].sample(n = np.minimum(data[data[ColName] == 1].shape[0],data[data[ColName] == 3].shape[0]))
# s2 = data[data[ColName] == 3].sample(n = np.minimum(data[data[ColName] == 1].shape[0],data[data[ColName] == 3].shape[0]))
# d = pd.concat([s1,s2])
# y = d[ColName]
# X = d.drop(columns = [ColName])
# X['prediction'] = model.predict(X)
# build_model(X,y,cross = 10,models = ['nvb','RandomForest','xgb','Logistic'])
# tune_model(X,y,n_it = 50,models = ['RandomForest','xgb','Logistic'])

# ColName = 'denial_reason_3'
# data = get_data(ColName,nr = 100000)
# s1 = data[data[ColName] == 1].sample(n = np.minimum(data[data[ColName] == 1].shape[0],data[data[ColName] == 9].shape[0]))
# s2 = data[data[ColName] == 9].sample(n = np.minimum(data[data[ColName] == 1].shape[0],data[data[ColName] == 9].shape[0]))
# d = pd.concat([s1,s2])
# y = d[ColName]
# X = d.drop(columns = [ColName])
# X['prediction'] = model.predict(X)
# build_model(X,y,cross = 10,models = ['nvb','RandomForest','xgb','Logistic'])
# tune_model(X,y,n_it = 50,models = ['RandomForest','xgb','Logistic'])

# #Pipeline to predict loan/no-loan and then predict reason

