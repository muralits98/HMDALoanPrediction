from __future__ import division
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tqdm as tqdm
from functions import get_data

def MonteCarlo():
    filename = 'acceptance_denial_tuned.sav'
    accden = pickle.load(open(filename, 'rb'))
    ColName = 'action_taken'
    data = get_data(ColName,nr = 100000,year = 2017,sk = 100000)
    original = np.array(data[ColName])
    X = data.drop(columns = [ColName])
    need = accden.predict(X)

    ori = X['applicant_sex']
    old_pred = need
    female_to_male_accept_prob = []
    male_to_female_accept_prob = []
    male_to_female_reject_prob = []
    female_to_male_reject_prob = []
    no_change_prob = []
    for i in tqdm(range(1,500)): #10257
        random.seed(i)
        X['applicant_sex'] = [random.choice([1,2,3,4]) for j in range(len(original))]
        monte = accden.predict(X)
        X['new_pred'] = monte
        X['original_sex'] = ori
        X['old_pred'] = need
        female_to_male_accept = X[(X['applicant_sex'] == 1) & (X['original_sex'] == 2) & (X['new_pred'] == 1)& (X['old_pred'] == 3)].shape[0]
        male_to_female_accept = X[(X['applicant_sex'] == 2) & (X['original_sex'] == 1) & (X['new_pred'] == 1)& (X['old_pred'] == 3)].shape[0]
        male_to_female_reject = X[(X['applicant_sex'] == 2) & (X['original_sex'] == 1) & (X['new_pred'] == 3)& (X['old_pred'] == 1)].shape[0]
        female_to_male_reject = X[(X['applicant_sex'] == 1) & (X['original_sex'] == 2) & (X['new_pred'] == 3)& (X['old_pred'] == 1)].shape[0]

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