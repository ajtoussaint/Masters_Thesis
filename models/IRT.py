#!/usr/bin/env python
# coding: utf-8

# Implementation of IRT based on Edu CDM: @misc{bigdata2021educdm, title={EduCDM}, author={bigdata-ustc}, publisher = {GitHub}, journal = {GitHub repository}, year = {2021}, howpublished = {\url{https://github.com/bigdata-ustc/EduCDM}}, }
# 
# Original model based on Reckase, Mark D. "18 Multidimensional Item Response Theory." Handbook of statistics 26 (2006): 607-642.

# In[1]:


import logging
import pandas as pd
import numpy as np
import pickle #used to save and load parameter settings
import os #manage saving user data
from tqdm import tqdm
from scipy import stats
#added to generate more metrics
from sklearn.metrics import roc_auc_score, accuracy_score


# In[2]:


def irt3pl(theta, a, b, c, D=1.702, *, F=np): #compute probability of correct response
    return c + (1 - c) / (1 + F.exp(-D * a * (theta - b))) #This equation dose NOT match the one in Reckase 06, not sure why
    #theta: ability parameter of student
    # a, b, c are as defined for init_parameters
    # D is a scaling constant applied which causes the model to produce similar item characteristic curves to the normal ogive model
    #NOTE: whenever this function is called theta is given as "a * (theta - b)" and a is given as 1 and b as 0 which is the same as actually plugging in the values. Not sure why this was done
    #also the variable stu_prof is theta 
    
#from EduCDM import CDM
class CDM(object):
    def __init__(self, *args, **kwargs) -> ...:
        pass

    def train(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def eval(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def save(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def load(self, *args, **kwargs) -> ...:
        raise NotImplementedError

def init_parameters(prob_num, dim): # Initializes distributions to be refined
    alpha = stats.norm.rvs(loc=0.75, scale=0.01, size=(prob_num, dim)) #alpha represents the "differentiability" of a given problem
    beta = stats.norm.rvs(size=(prob_num, dim)) #beta represents the "difficulty" of a given problem
    gamma = stats.uniform.rvs(size=prob_num) #gamma is the "guess" parameter associated with a problem
    return alpha, beta, gamma


def init_prior_prof_distribution(dim):
    prof = stats.uniform.rvs(loc=-4, scale=8, size=(100, dim))  # shape = (100,dim) #generates a random 100xdim matrix with possible values -4 to 4
    dis = stats.multivariate_normal.pdf(prof, mean=np.zeros(dim), cov=np.identity(dim)) # calculates probability density function
    norm_dis = dis / np.sum(dis)  # shape = (100,) # normalizes density function
    return prof, norm_dis


def get_Likelihood(a, b, c, prof, R):
    stu_num, prob_num = R.shape[0], R.shape[1]
    prof_prob = irt3pl(np.sum(a * (np.expand_dims(prof, axis=1) - b), axis=-1), 1, 0, c)  # shape = (100, prob_num) #probability of correct response for each student item pair
    tmp1, tmp2 = np.zeros(shape=(prob_num, stu_num), dtype=np.int8), np.zeros(shape=(prob_num, stu_num), dtype=np.int8)
    tmp1[np.where(R == 1)[1], np.where(R == 1)[0]] = 1
    tmp2[np.where(R == 0)[1], np.where(R == 0)[0]] = 1
    prob_stu = np.exp(np.dot(np.log(prof_prob + 1e-9), tmp1) + np.dot(np.log(1 - prof_prob + 1e-9), tmp2)) #likelihood of the actual observed responses
    return prof_prob, prob_stu


def update_prior(prior_dis, prof_stu_like): #update the given prior distribution based on the given likelihoods
    dis_like = prof_stu_like * np.expand_dims(prior_dis, axis=1)
    norm_dis_like = dis_like / (np.sum(dis_like, axis=0) + 1e-9) #added 1e-9 to prevent div0 error
    update_prior_dis = np.sum(norm_dis_like, axis=1) / (np.sum(norm_dis_like) + 1e-9) #added 1e-9 to prevent div0 error
    return update_prior_dis, norm_dis_like


def update_irt(a, b, c, D, prof, R, r_ek, s_ek, lr, epoch=10, epsilon=1e-3): #updates the a, b, and c parameters of the model
    for iteration in range(epoch):
        a_tmp, b_tmp, c_tmp = np.copy(a), np.copy(b), np.copy(c)
        prof_prob, _ = get_Likelihood(a, b, c, prof, R) #returns the probability of a correct response for each student/item pair
        common_term = (r_ek - s_ek * prof_prob) / prof_prob / (1 - c + 1e-9)  # shape = (100, prob_num)
        a_1 = np.transpose(
            D * common_term * (prof_prob - c) * np.transpose(np.expand_dims(prof, axis=1) - b, (2, 0, 1)), (1, 2, 0))
        b_1 = D * common_term * (c - prof_prob)
        a_grad = np.sum(a_1, axis=0)
        b_grad = a * np.expand_dims(np.sum(b_1, axis=0), axis=1)
        c_grad = np.sum(common_term, axis=0)
        a = a + lr * a_grad #increase each parameter along their gradient according to the learning rate
        b = b + lr * b_grad
        c = np.clip(c + lr * c_grad, 0, 1)
        change = max(np.max(np.abs(a - a_tmp)), np.max(np.abs(b - b_tmp)), np.max(np.abs(c - c_tmp)))
        if iteration > 5 and change < epsilon: #stop if 5 epochs have passed with no significant change
            break
    return a, b, c


class IRT(CDM):
    """
    IRT model, training (EM) and testing methods
    Parameters
    ----------
    R: numpy.array
        response matrix, shape = (stu_num, prob_num)
    stu_num: int
        number of students
    prob_num: int
        number of problems
    dim: int
        dimension of student/problem embedding, MIRT for dim > 1
    skip_value: int
        skip value in response matrix
    ----------
    """
    def __init__(self, R, stu_num, prob_num, dim=1, skip_value=-1):
        super(IRT, self).__init__()
        self.R, self.skip_value = R, skip_value
        self.stu_num, self.prob_num, self.dim = stu_num, prob_num, dim
        self.a, self.b, self.c = init_parameters(prob_num, dim)  # IRT parameters
        self.D = 1.702
        self.prof, self.prior_dis = init_prior_prof_distribution(dim) #start with random proabaility of correct response and normalized density
        self.stu_prof = np.zeros(shape=(stu_num, dim), dtype=np.int8)

    def train(self, lr, epoch, epoch_m=10, epsilon=1e-3):
        a, b, c = np.copy(self.a), np.copy(self.b), np.copy(self.c)
        prior_dis = np.copy(self.prior_dis)
        for iteration in range(epoch):
            a_tmp, b_tmp, c_tmp, prior_dis_tmp = np.copy(a), np.copy(b), np.copy(c), np.copy(prior_dis)
            prof_prob_like, prof_stu_like = get_Likelihood(a, b, c, self.prof, self.R) # returns the probability of a correct response and the likelihood of the observed response
            prior_dis, norm_dis_like = update_prior(prior_dis, prof_stu_like) # based on the liklihood of the observed responses the prior distribution is updated

            r_1 = np.zeros(shape=(self.stu_num, self.prob_num), dtype=np.int8)
            r_1[np.where(self.R == 1)[0], np.where(self.R == 1)[1]] = 1
            r_ek = np.dot(norm_dis_like, r_1)  # shape = (100, prob_num)
            r_1[np.where(self.R != self.skip_value)[0], np.where(self.R != self.skip_value)[1]] = 1
            s_ek = np.dot(norm_dis_like, r_1)  # shape = (100, prob_num)
            #the irt function parameters are updated based on information from the new normalized likelihood distribution
            a, b, c = update_irt(a, b, c, self.D, self.prof, self.R, r_ek, s_ek, lr, epoch_m, epsilon) 
            change = max(np.max(np.abs(a - a_tmp)), np.max(np.abs(b - b_tmp)), np.max(np.abs(c - c_tmp)),
                         np.max(np.abs(prior_dis_tmp - prior_dis_tmp)))
            if iteration > 20 and change < epsilon: #stop iterating if the updated parameters have converged
                break
        self.a, self.b, self.c, self.prior_dis = a, b, c, prior_dis
        self.stu_prof = self.transform(self.R) #applies MLE to update the student profiles

    def eval(self, test_data) -> tuple:
        pred_score = irt3pl(np.sum(self.a * (np.expand_dims(self.stu_prof, axis=1) - self.b), axis=-1), 1, 0, self.c)
        test_rmse, test_mae, y_true, y_pred = [], [], [], [] # y matricies used to calculate AUC and ACC
        for i in test_data:
            stu, test_id, true_score = i['user_id'], i['item_id'], i['score']
            test_rmse.append((pred_score[stu, test_id] - true_score) ** 2)
            test_mae.append(abs(pred_score[stu, test_id] - true_score))

            #for ACC and AUC
            predicted = pred_score[stu, test_id]
            y_true.append(true_score)
            y_pred.append(predicted)
            

        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9) #clip to prevent underflow/overflow, may reduce auc
        
        rmse = np.sqrt(np.average(test_rmse))
        mae = np.average(test_mae)
        acc = accuracy_score(y_true, [1 if p >= 0.5 else 0 for p in y_pred])
        if len(set(y_true)) < 2:
            auc = None 
        else:
            auc = roc_auc_score(y_true, y_pred) 
        
        return acc, auc, mae, rmse

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump({"a": self.a, "b": self.b, "c": self.c, "prof": self.stu_prof}, file)
            logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        with open(filepath, 'rb') as file:
            self.a, self.b, self.c, self.stu_prof = pickle.load(file).values()
            logging.info("load parameters from %s" % filepath)

    def inc_train(self, inc_train_data, lr=1e-3, epoch=10, epsilon=1e-3):  # incremental training, can be applied in real time as students generate responses
        for i in inc_train_data:
            stu, test_id, true_score = i['user_id'], i['item_id'], i['score']
            self.R[stu, test_id] = true_score
        self.train(lr, epoch, epsilon=epsilon)

    def transform(self, records, lr=1e-3, epoch=10, epsilon=1e-3):  # MLE for evaluating students' state
        # can evaluate multiple students' states simultaneously, thus output shape = (stu_num, dim)
        # initialization stu_prof, shape = (stu_num, dim)
        if len(records.shape) == 1:  # one student
            records = np.expand_dims(records, axis=0)
        _, prof_stu_like = get_Likelihood(self.a, self.b, self.c, self.prof, records)
        stu_prof = self.prof[np.argmax(prof_stu_like, axis=0)]

        for iteration in range(epoch):
            prof_tmp = np.copy(stu_prof)
            ans_prob = irt3pl(np.sum(self.a * (np.expand_dims(stu_prof, axis=1) - self.b), axis=-1), 1, 0, self.c) #expand dims too big 7/31
            ans_1 = self.D * (records - ans_prob) / ans_prob * (ans_prob - self.c) / (1 - self.c + 1e-9)
            ans_1[np.where(records == self.skip_value)[0], np.where(records == self.skip_value)[1]] = 0
            prof_grad = np.dot(ans_1, self.a)
            stu_prof = stu_prof - lr * prof_grad
            change = np.max(np.abs(stu_prof - prof_tmp))
            if iteration > 5 and change < epsilon:
                break
        return stu_prof  # shape = (stu_num, dim)

def run_IRT(Q, train_data, test_data, valid_data, fold_n, dataset, runType, user_data=False, overwrite=False):
    print("Prepparing to train on IRT...")
    model = "IRT"
    #even when loading existing data, need to prepare the cdm class to be able to load in weights
    stu_num = max(max(train_data['user_id']), max(test_data['user_id']))
    prob_num = max(max(train_data['item_id']), max(test_data['item_id']))

    R = -1 * np.ones(shape=(stu_num, prob_num), dtype=np.int8)
    R[train_data['user_id']-1, train_data['item_id']-1] = train_data['score'] #R matrix shows each question a student answered and whether they got it correct or incorrect

    cdm = IRT(R, stu_num, prob_num, dim=1, skip_value=-1)

    filename = "models/snapshots/" + model + "/" + model + "_" + dataset + "_" + runType + "_" + str(fold_n) + ".params"
    
    if((not os.path.exists(filename)) or overwrite):
        print("Training IRT with new data")
        cdm.train(lr=1e-3, epoch=2)
        cdm.save(filename)
        
    cdm.load(filename)
        
    valid_set = []
    for i in range(len(valid_data)):
        row = valid_data.iloc[i]
        valid_set.append({'user_id':int(row['user_id'])-1, 'item_id':int(row['item_id'])-1, 'score':row['score']})

    acc, auc, mae, rmse = cdm.eval(valid_set)

    print("Trained on IRT for ACC: %.6f, AUC: %.6f, MAE: %.6f, RMSE: %.6f" % (acc, auc, mae, rmse))

    #get detailed info on how the model performs on each user
    if(user_data):
        print("Gathering data on individual users")
        ud_path = 'user_results.csv'
        if os.path.exists(ud_path):
            df = pd.read_csv(ud_path)
        else:
            df = pd.DataFrame(columns = ["model", "runType", "dataset","fold_n","user_id",
                                         "n_responses_train", "portion_correct_train", "problem_set_train", "concept_set_train",
                                         "n_responses_valid", "portion_correct_valid", "problem_set_valid", "concept_set_valid",
                                         "percent_tested_problems_trained", "percent_tested_concepts_trained", "ACC", "AUC", "MAE", "RMSE"])
            
        for user in tqdm(train_data["user_id"].unique(), "Evaluating individual users"):
            #check that df does not already contain this data
            if not (((df['model'] == model) & (df['runType'] == runType) & (df['dataset'] == dataset) & (df['fold_n'] == fold_n) & (df['user_id'] == user)).any()):
                user_train = train_data[train_data["user_id"] == user]
                user_valid = valid_data[valid_data["user_id"] == user]
        
                train_probs = user_train['item_id'].unique()
                subset_Q = Q[Q['item_id'].isin(train_probs)]
                train_concepts = subset_Q.columns[(subset_Q == 1).any()].tolist()
        
                valid_probs = user_valid['item_id'].unique()
                subset_Q = Q[Q['item_id'].isin(train_probs)]
                valid_concepts = subset_Q.columns[(subset_Q == 1).any()].tolist()
        
                def findSetRatio(train, valid):
                    count = 0
                    for i in valid:
                        if i in train:
                            count += 1
                    return count/len(valid)
        
                #format the user_valid data for evaluation
                user_valid_set = []
                for i in range(len(user_valid)):
                    row = user_valid.iloc[i]
                    user_valid_set.append({'user_id':int(row['user_id'])-1, 'item_id':int(row['item_id'])-1, 'score':row['score']})
        
                acc, auc, mae, rmse = cdm.eval(user_valid_set)
                
                result_object = {
                    "model":model,
                    "dataset":dataset,
                    "runType":runType,
                    "fold_n":fold_n,
                    "user_id":user,
                    "n_responses_train":len(user_train),
                    "portion_correct_train":len(user_train[user_train['score'] == 1]) / len(user_train),
                    "problem_set_train":train_probs,
                    "concept_set_train":train_concepts,
                    "n_responses_valid":len(user_valid),
                    "portion_correct_valid":len(user_valid[user_valid['score'] == 1]) / len(user_valid),
                    "problem_set_valid":valid_probs,
                    "concept_set_valid":valid_concepts,
                    "percent_tested_problems_trained": findSetRatio(train_probs, valid_probs),
                    "percent_tested_concepts_trained": findSetRatio(train_concepts, valid_concepts),
                    "ACC": acc, "AUC": auc, "MAE":mae, "RMSE":rmse
                }
                
                df = pd.concat([df, pd.DataFrame([result_object])], ignore_index=True)
                #update file each iteration
                df.to_csv(ud_path, index=False)
            
    return acc, auc, mae, rmse