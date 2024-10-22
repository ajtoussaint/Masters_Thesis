import logging
logging.getLogger().setLevel(logging.INFO)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error
import os

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

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, exer_n, student_n, knowledge_n, mf_type, dim):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim #sets embedding size, higher trades train time for better representation
        self.mf_type = mf_type #matrix factorization method
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        if mf_type == 'gmf':
            self.k_diff_full = nn.Linear(self.emb_dim, 1)
            self.stat_full = nn.Linear(self.emb_dim, 1)
        elif mf_type == 'ncf1':
            self.k_diff_full = nn.Linear(2 * self.emb_dim, 1)
            self.stat_full = nn.Linear(2 * self.emb_dim, 1)
        elif mf_type == 'ncf2':
            self.k_diff_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.k_diff_full2 = nn.Linear(self.emb_dim, 1)
            self.stat_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.stat_full2 = nn.Linear(self.emb_dim, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        exer_emb = self.exercise_emb(input_exercise)
        # get knowledge proficiency
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf': # general matrix factorization
            stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1': # neural collaborative filtering methods
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2': # neural collaborative filtering methods
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)
        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        if self.mf_type == 'mf':
            k_difficulty = torch.sigmoid((exer_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            k_difficulty = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
        # get exercise discrimination
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        # prednet
        input_x = e_discrimination * (stat_emb - k_difficulty) * input_knowledge_point
        # f = input_x[input_knowledge_point == 1]
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


class KaNCD(CDM):
    def __init__(self, **kwargs):
        super(KaNCD, self).__init__()
        mf_type = kwargs['mf_type'] if 'mf_type' in kwargs else 'gmf' #general matrix factorization
        self.net = Net(kwargs['exer_n'], kwargs['student_n'], kwargs['knowledge_n'], mf_type, kwargs['dim'])

    def train(self, train_set, valid_set, lr=0.002, device='cpu', epoch_n=15):
        logging.info("traing... (lr={})".format(lr))
        self.net = self.net.to(device)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        for epoch_i in range(epoch_n):
            self.net.train()
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_set, "Epoch %s" % epoch_i):
                batch_count += 1
                user_info, item_info, knowledge_emb, y = batch_data
                user_info: torch.Tensor = user_info.to(device)
                item_info: torch.Tensor = item_info.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred = self.net(user_info, item_info, knowledge_emb)
                loss = loss_function(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            logging.info("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            auc, acc, rmse, mae = self.eval(valid_set, device)
            print("[Epoch %d] auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))
            logging.info("[Epoch %d] auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))

        return auc, acc

    def eval(self, test_data, device="cpu"):
        #logging.info('eval ... ')
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []
        for batch_data in test_data:
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred = self.net(user_id, item_id, knowledge_emb)
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
        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))
        logging.info("load parameters from %s" % filepath)

def run_KaNCD(Q, train_data, test_data, valid_data, fold_n, dataset, runType, user_data=False, overwrite=False):
    print("Preparing to train on KaNCD...")
    model = "KaNCD"
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
    
    user_n = np.max(train_data['user_id']) + 1
    item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])]) + 1
    knowledge_n = np.max(list(map(int, knowledge_set))) + 1
    print(f'KaNCD detects {user_n} users, {item_n} items, and {knowledge_n} knowledge concepts')
    
    cdm = KaNCD(exer_n=item_n, student_n=user_n, knowledge_n=knowledge_n, mf_type='gmf', dim=20)

    filename = "models/snapshots/" + model + "/" + model + "_" + dataset + "_" + runType + "_" + str(fold_n) + ".snapshot"
    
    #transform and batch data
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
        print("Training KaNCD with new data")       
        
        train_set, valid_set, test_set = [
            transform(data["user_id"], data["item_id"], item2knowledge, data["score"], batch_size)
            for data in [train_data, valid_data, test_data]
        ] #data iterates through train, test, valid to transform all of them
        
        #check for GPU and select or run on cpu
        device_str = "cpu"
        if torch.cuda.is_available():
            device_str = "cuda"
        cdm.train(train_set, test_set, epoch_n=3, device=device_str, lr=0.002)
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


def old_run_KaNCD(Q, train_data, test_data, valid_data, param_file):
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

  user_n = np.max(train_data['user_id']) + 1
  item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])]) + 1
  knowledge_n = np.max(list(map(int, knowledge_set))) + 1
  print(f'KaNCD detects {user_n} users, {item_n} items, and {knowledge_n} knowledge concepts')

  #transform and batch data
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
  
  
  train_set, valid_set, test_set = [
      transform(data["user_id"], data["item_id"], item2knowledge, data["score"], batch_size)
      for data in [train_data, valid_data, test_data]
  ] #data iterates through train, test, valid to transform all of them
  device_str = "cpu"
  if torch.cuda.is_available():
    device_str = "cuda"

  cdm = KaNCD(exer_n=item_n, student_n=user_n, knowledge_n=knowledge_n, mf_type='gmf', dim=20)
  cdm.train(train_set, test_set, epoch_n=3, device=device_str, lr=0.002)
  cdm.save("kancd.snapshot")
  cdm.load("kancd.snapshot")
  auc, acc, rmse, mae = cdm.eval(valid_set)
  print("auc: %.6f, accuracy: %.6f, rmse: %.6f, mae:%.6f" % (auc, accuracy, rmse, mae))

  return acc, auc, rmse, mae