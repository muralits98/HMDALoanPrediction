"""
Necessary imports

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
# from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
# from IPython.display import Image
from xgboost import plot_tree
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import export_graphviz
from tpot import TPOTClassifier
# from pygam import LogisticGAM, s, f

from subprocess import call
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as skm
import pickle
# from interpret import show
from sklearn import svm
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB

def Exploratory_plots(data):
    correlation = data.corr()
    print(correlation)

    for i in data.columns():
        plt.hist(data[i])
        plt.savefig(str(i)+'historgram.png')

        plt.boxplot(data[i])
        plt.savefig(str(i)+'boxplot.png')

def get_data(drop1,nr = 10000,nona = 1,res = 0,year = 2017,sk = 0):
    if year == 2017:
        data = pd.read_csv(r"C:\Users\tsmur\Desktop\Spring2020\StatDataMining\Project\hmda_"+str(year)+"_nationwide_all-records_codes.csv", nrows = nr,skiprows=[i for i in range(1,sk)])
    elif year <= 2016:
        data = pd.read_csv(r"C:\Users\tsmur\Desktop\Spring2020\StatDataMining\Project\hmda_"+str(year)+"_nationwide_all-records_labels.csv", nrows = nr,skiprows=[i for i in range(1,sk)])
    else:
        print("Year not found")
    data = data.replace(' ', None)
    data = data.replace('', None)
    data = data.replace('NA', None)
    if res == 1:
        c = ['rate_spread','sequence_number','co_applicant_race_5','agency_code',
       'co_applicant_race_4', 'co_applicant_race_3','applicant_race_4','census_tract_number',
       'applicant_race_3', 'applicant_race_2','denial_reason_3', 'denial_reason_2',
         'denial_reason_1','county_code', 'purchaser_type',
       'applicant_race_1','applicant_race_5','as_of_year','preapproval',
       'co_applicant_race_2', 'co_applicant_race_1','application_date_indicator',
         'respondent_id','state_code','edit_status','msamd']
        c.remove(drop1)
        data = data.drop(columns = c)
        print(np.shape(data))
        if nona == 1:
            data = data.dropna()
        data = data[data['action_taken'] == 3]
        data = data.drop(columns = ['action_taken'])
        print(np.shape(data))
        return data
    c = ['rate_spread','sequence_number','co_applicant_race_5','agency_code',
       'co_applicant_race_4', 'co_applicant_race_3','applicant_race_4','census_tract_number',
       'applicant_race_3', 'applicant_race_2','denial_reason_3', 'denial_reason_2',
         'denial_reason_1','action_taken','county_code', 'purchaser_type',
       'applicant_race_1','applicant_race_5','as_of_year','preapproval',
       'co_applicant_race_2', 'co_applicant_race_1','application_date_indicator',
         'respondent_id','state_code','edit_status','msamd']
    c.remove(drop1)
    data = data.drop(columns = c)
    print(np.shape(data))
    if nona == 1:
        data = data.dropna()
    # to_cat = ['owner_occupancy', 'co_applicant_ethnicity', 'loan_type', 'hoepa_status', 'applicant_ethnicity', 
    #             'loan_purpose', 'lien_status', 'applicant_sex', 'co_applicant_sex', 'property_type', 'number_of_owner_occupied_units']
    # for i in to_cat:
    #     data = pd.concat([data,pd.get_dummies(data[i])],axis=1)
    #     data.drop([i],axis=1, inplace=True)
    print("The columns in the data are", data.columns)
    print(np.shape(data))
    return data

def post_proc(X,model):
    """
    Need Post processing support like
    1. What to do with the output of feature importance
    """
    for ind,val in zip(X.columns,model.feature_importances_*100):
        if val>1:
            print(ind,val)


def build_model(X,y,name,cross = 5,models = ['xgb']):
    Acc_zero = 0
    """
    Need support for more models, along with cross validation and feature importances which can be easily taken out
    something like
    build_model(X,y,cross = 5,model)
        if model == 'xgb':
            ...
        if model == 'logistic'
            ...
    """
    seed = 7
    test_size = 0.30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    for model1 in tqdm(models):
        if model1 == 'xgb':
            print("\n XGBoost Classifier: \n")
            model = XGBClassifier()
            model.fit(X_train,y_train)
            pred = model.predict(X_test)
            print("Training Balanced Accuracy is ",balanced_accuracy_score(y_train,model.predict(X_train))*100)
            print("Testing Balanced Accuracy is ",balanced_accuracy_score(y_test,pred)*100)
            results = cross_val_score(model, X_train, y_train, cv=cross,scoring = 'balanced_accuracy')
            print("Cross Validation Balanced Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
            post_proc(X_train,model)
            Acc = balanced_accuracy_score(y_test,pred)*100
            # plot_tree(model)
            # plt.savefig('xgb_model_untuned_tree.png')
        elif model1 == 'Logistic':
            print("\n Logistic Classifier: \n")
            model = LogisticRegression(solver = 'liblinear')
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            print("Training Balanced Accuracy is ",balanced_accuracy_score(y_train,model.predict(X_train))*100)
            print("Testing Balanced Accuracy is ",balanced_accuracy_score(y_test,pred)*100)
            results = cross_val_score(model, X_train, y_train, cv=cross,scoring = 'balanced_accuracy')
            print("Cross Validation Balanced Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
            cm = confusion_matrix(y_test, pred)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(cm)
            ax.grid(False)
            ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
            ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
            ax.set_ylim(1.5, -0.5)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
            # plt.show()
            plt.savefig(str(name)+'logistic_model_untuned.png')
            Acc = balanced_accuracy_score(y_test,pred)*100
        elif model1 == 'auto':
            print("\n Automatic ML: \n")
            model = TPOTClassifier(generations=10, population_size=30,verbosity=3, scoring = 'balanced_accuracy')
            model.fit(X_train, y_train)
            print(model.score(X_test, y_test))
            print("Training Balanced Accuracy is ",model.score(X_train,y_train)*100)
            print("Testing Balanced Accuracy is ",model.score(X_test,y_test)*100)
            Acc = model.score(X_test, y_test) * 100
            model.export(str(name) + '_auto.py')
        elif model1 == 'SVM':
            print("\n SVM: \n")
            model = svm.NuSVC(gamma='auto')
            model.fit(X_train,y_train)
            pred = model.predict(X_test)
            print("Training Balanced Accuracy is ",balanced_accuracy_score(y_train,model.predict(X_train))*100)
            print("Testing Balanced Accuracy is ",balanced_accuracy_score(y_test,pred)*100)
            results = cross_val_score(model, X_train, y_train, cv=cross,scoring = 'balanced_accuracy')
            print("Cross Validation Balanced Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
            Acc = balanced_accuracy_score(y_test,pred)*100
        # elif model1 == 'gam':
        #     lams = np.random.rand(X_train.shape[0], X_train.shape[1]) # random points on [0, 1], with shape (100, 3)
        #     lams = lams * 8 - 15 # shift values to -3, 3
        #     lams = np.exp(lams) # transforms values to 1e-3, 1e3
        #     X_train.to_numpy()
        #     y_train.to_numpy()
        #     X_test.to_numpy()
        #     y_test.to_numpy()
        #     random_gam =  LogisticGAM(f(0) + f(1) + f(2) + f(3) + s(4) + f(5) + f(6) + f(7) + f(8) + f(9) + f(10) + s(11) + f(12) + f(13) + f(14) + s(15)+ s(16)+ s(17))
        #     # model = LogisticGAM(n_splines=20)
        #     # model.gridsearch(X_train.values, y_train.values)
        #     random_gam.gridsearch(X_train.values, y_train.values, lam=lams)
        #     print(model.summary())
        #     pred = model.predict(X_test)
        #     print("Balanced Accuracy is ",balanced_accuracy_score(y_test,pred)*100)
        #     Acc = balanced_accuracy_score(y_test,pred)*100
        elif model1 == 'RandomForest':
            print("\n Random Forest: \n")
            model = RandomForestClassifier()
            model.fit(X_train,y_train)
            pred = model.predict(X_test)
            print("Training Balanced Accuracy is ",balanced_accuracy_score(y_train,model.predict(X_train))*100)
            print("Testing Balanced Accuracy is ",balanced_accuracy_score(y_test,pred)*100)
            results = cross_val_score(model, X_train, y_train, cv=cross,scoring = 'balanced_accuracy')
            print("Cross Validation Balanced Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
            Acc = balanced_accuracy_score(y_test,pred)*100
            # export_graphviz(model, 
            #     out_file='random_forest_untuned_tree.dot', 
            #     feature_names = X.columns,
            #     class_names = 'action_taken',
            #     rounded = True, proportion = False, 
            #     precision = 2, filled = True)
            # # call(['dot', '-Tpng', 'random_forest_untuned_tree.dot', '-o', 'random_forest_untuned_tree.png', '-Gdpi=600']
            # Image(filename = 'random_forest_untuned_tree.png')
        elif model1 == 'nvb':
            print("\n Naive Bayes Classifier: \n")
            model = GaussianNB()
            model.fit(X_train,y_train)
            pred = model.predict(X_test)
            print("Training Balanced Accuracy is ",balanced_accuracy_score(y_train,model.predict(X_train))*100)
            print("Testing Balanced Accuracy is ",balanced_accuracy_score(y_test,pred)*100)
            results = cross_val_score(model, X_train, y_train, cv=cross,scoring = 'balanced_accuracy')
            print("Cross Validation Balanced Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
            Acc = balanced_accuracy_score(y_test,pred)*100
        # elif model1 == 'explainable':
        #     ebm = ExplainableBoostingClassifier()
        #     ebm.fit(X_train, y_train)
        #     ebm_global = ebm.explain_global()
        #     show(ebm_global)
        #     ebm_local = ebm.explain_local(X_test, y_test)
        #     show(ebm_local)
        #     Acc = balanced_accuracy_score(y_test,pred)*100
        else:
            print(model1, "- Name not detected. Try using one of the models that are defined")
        try:
            if Acc > Acc_zero:
                filename = str(name) + '.sav'
                pickle.dump(model, open(filename, 'wb'))
                Acc_zero = Acc
        except:
            continue
    print("Best picked model is", filename)
if __name__ == "__main__":
    ColName = 'action_taken_name'
    data,mapping = get_data(ColName,nr = 1000)
    y = data[ColName]
    X = data.drop(columns = [ColName])
    build_model(X,y,name='test', cross = 10,models = ['xgb','Logistic'])
    