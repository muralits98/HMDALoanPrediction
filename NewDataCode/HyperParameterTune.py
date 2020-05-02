import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as skm
import pickle
from sklearn import svm
import joblib
from tqdm import tqdm

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

def tune_model(X,y,name,n_it = 30, models = ['xgb']):
    Acc_zero = 0
    seed = 7
    test_size = 0.30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    for model1 in tqdm(models):
        if model1 == 'Logistic':
            print("logistic Regression \n")
            logistic = LogisticRegression()
            distributions = {
                'C' : [1,2,3,4], 
                'penalty': ['l1', 'l2']                                
            }
            clf = GridSearchCV(logistic, distributions,cv = 5)
            clf.fit(X_train, y_train)
            print(clf.best_params_)
            print(clf.cv_results_)
            pred = clf.predict(X_test)
            print("The best Logistic Balanced Accuracy is ",balanced_accuracy_score(y_test,pred)*100)
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
            plt.show()
            plt.savefig(str(name)+'logistic_model_tuned.png')
        elif model1 == 'xgb':
            print("\n XGBoost \n")
            model = XGBClassifier()
            distributions = {
                'booster' : ['gbtree','gblinear','dart'],
                'eta' : [0,0.2,0.4,0.6,0.8,1],
                'max_depth' : [50,100,150,200,250,300],
                'lambda' : [0,0.2,0.4,0.6,0.8,1],
                'alpha' : [0,0.2,0.4,0.6,0.8,1],
                'grow_policy' : ['depthwise','lossguide']
            }
            clf = RandomizedSearchCV(model, distributions, random_state=0,cv = 5)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            print(clf.best_params_)
            print(clf.cv_results_)
            print("The best XGBoost Balanced Accuracy is ",balanced_accuracy_score(y_test,pred)*100)
            Acc = balanced_accuracy_score(y_test,pred)*100
        elif model1 == 'SVM':
            print("\n SVM \n")
            model = svm.NuSVC(gamma='auto')
            distributions = {
                'kernel' : ['linear','rbf','poly','sigmoid'],
                'degree' : [4,5,6,7,8,9,10]
            }
            clf = RandomizedSearchCV(model, distributions, random_state=0,n_iter = n_it,cv = 5)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            print(clf.best_params_)
            print(clf.cv_results_)
            print("The best SVM Balanced Accuracy is ",balanced_accuracy_score(y_test,pred)*100)
            Acc = balanced_accuracy_score(y_test,pred)*100
        elif model1 == 'RandomForest':
            print("\n Random Forest \n")
            model = RandomForestClassifier()
            distributions = {
                'n_estimators' : [50,100,150,200,250,300],
                'criterion' : ['gini','entropy'],
                'min_samples_split' : [2,3,4,5],
                'min_samples_leaf' : [2,3,4,5],
                'max_features' : ['auto','sqrt','log2']
            }
            clf = RandomizedSearchCV(model, distributions, random_state=0,n_iter = n_it,cv = 5)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            print(clf.best_params_)
            print(clf.cv_results_)
            print("The best Random Forest Balanced Accuracy is ",balanced_accuracy_score(y_test,pred)*100)
            Acc = balanced_accuracy_score(y_test,pred)*100
        else:
            print(model1, "- Name not detected. Try using one of the models that are defined")
        if Acc > Acc_zero:
            model1 = clf
            filename = str(name) + '.sav'
            pickle.dump(model1, open(filename, 'wb'))
            Acc_zero = Acc
    print("Best picked model is", model1)


# import sklearn.ensemble
# import sklearn.model_selection
# import sklearn.svm

# import optuna


# # FYI: Objective functions can take additional arguments
# # (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).



# def GuidedTuneModel(X,y):
#     def objective(trial):

#         classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
#         if classifier_name == "SVC":
#             svc_c = trial.suggest_loguniform("svc_c", 1e-10, 1e10)
#             classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
#         else:
#             rf_max_depth = int(trial.suggest_loguniform("rf_max_depth", 2, 32))
#             classifier_obj = sklearn.ensemble.RandomForestClassifier(
#                 max_depth=rf_max_depth, n_estimators=10
#             )

#         score = sklearn.model_selection.cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=3)
#         accuracy = score.mean()
#         return accuracy
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=100)
#     print(study.best_trial)