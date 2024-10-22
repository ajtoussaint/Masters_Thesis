#!/usr/bin/env python
# coding: utf-8

# This code is adapted from Edu CDM:</br>
# @misc{bigdata2021educdm,
#   title={EduCDM},
#   author={bigdata-ustc},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   year = {2021},
#   howpublished = {\url{https://github.com/bigdata-ustc/EduCDM}},
# }<br></br>
# Specifically the presentation of the NCDM model and example as originally proposed in: </br>
# @article{wang2022neuralcd,
#   title={NeuralCD: A General Framework for Cognitive Diagnosis},
#   author={Wang, Fei and Liu, Qi and Chen, Enhong and Huang, Zhenya and Yin, Yu and Wang, Shijin and Su, Yu},
#   journal={IEEE Transactions on Knowledge and Data Engineering},
#   year={2022},
#   publisher={IEEE}
# }

# In[ ]:


#NCDM required imports
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error
import os


# In[ ]:


# showing the NCDM class for understanding definitions

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

# Uses the gradient descent model
class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

class NCDM(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n):
        super(NCDM, self).__init__()
        self.ncdm_net = Net(knowledge_n, exer_n, student_n)

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            if test_data is not None:
                auc, accuracy, rmse, mae = self.eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, mae: %.6f" % (epoch_i, auc, accuracy, rmse, mae))

    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred = [], []
        for batch_data in test_data:
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        acc = accuracy_score(y_true, np.array(y_pred) >= 0.5)
        if(len(set(y_true))<2):
            auc = None
        else:
            auc = roc_auc_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        #calculate rmse
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        return acc, auc, mae, rmse

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath))  
        logging.info("load parameters from %s" % filepath)


def run_NCDM(Q, train_data, test_data, valid_data, fold_n, dataset, runType, user_data=False, overwrite=False):
    print("Preparing to train on NCDM...")
    model = "NCDM"
    #even when loading existing data, need to prepare the cdm class to be able to load in weights
    item2knowledge = {}
    knowledge_set = set()
    #format problem IDs accordingly
    for _, row in Q.iterrows():
        item_id = int(row['item_id'])
        columns_with_one = row[row == 1].index.tolist()
        #prevent item id=1 from causing "item_id" to be a skill
        if "item_id" in columns_with_one:
            columns_with_one.remove("item_id")
        knowledge_codes = list(map(int, columns_with_one))
        item2knowledge[item_id] = knowledge_codes
        knowledge_set.update(knowledge_codes)

    #find the number of users, items, and knowledge concepts
    user_n = np.max(train_data['user_id']) + 1
    item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])]) + 1
    #print("k_set max, min, len", max(list(map(int, knowledge_set))), min(list(map(int, knowledge_set))), len(list(map(int, knowledge_set))))
    knowledge_n = np.max(list(map(int, knowledge_set))) + 1
    print(f'NCDM detects {user_n} users, {item_n} items, and {knowledge_n} knowledge concepts')
    
    cdm = NCDM(knowledge_n, item_n, user_n)

    filename = "models/snapshots/" + model + "/" + model + "_" + dataset + "_" + runType + "_" + str(fold_n) + ".snapshot"

    #utility to transform and batch data
    batch_size = 32
    def transform(user, item, item2knowledge, score, batch_size):
        knowledge_emb = torch.zeros((len(item), knowledge_n))
        item=item.reset_index(drop=True)
        user=user.reset_index(drop=True)
        score=score.reset_index(drop=True)
        for idx in range(len(item)):
            knowledge_emb[idx][np.array(item2knowledge[item[idx]])] = 1.0
        
        data_set = TensorDataset( 
            torch.tensor(user, dtype=torch.int64),  
            torch.tensor(item, dtype=torch.int64),  
            knowledge_emb,
            torch.tensor(score, dtype=torch.float32)
        )
        return DataLoader(data_set, batch_size=batch_size, shuffle=True)

    if((not os.path.exists(filename)) or overwrite):
        print("Training NCDM with new data")       
        
        train_set, valid_set, test_set = [
            transform(data["user_id"], data["item_id"], item2knowledge, data["score"], batch_size)
            for data in [train_data, valid_data, test_data]
        ] #data iterates through train, test, valid to transform all of them
        
        #check for GPU and select or run on cpu
        device_str = "cpu"
        if torch.cuda.is_available():
            device_str = "cuda"
        cdm.train(train_set, test_set, epoch=3, device=device_str)
        cdm.save(filename)
    else:
        #still transform valid set for evaluation
        valid_set = transform(valid_data["user_id"], valid_data["item_id"], item2knowledge, valid_data["score"], batch_size)

    cdm.load(filename)
    acc, auc, mae, rmse = cdm.eval(valid_set)
    print("accuracy: %.6f, auc: %.6f, mae: %.6f, rmae:%.6f" % (acc, auc, mae, rmse))
    if(user_data):
        print("Gathering data on individual users")
        ud_path='user_results.csv'
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
                user_valid_set = transform(user_valid["user_id"], user_valid["item_id"], item2knowledge, user_valid["score"], batch_size)
        
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