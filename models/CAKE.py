import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error
import os

# A neural network layer adapted to enforce the monotonicity assumption as proposed in NCDM model
class MonotonicLinear(nn.Linear):
  def forward(self, input: torch.Tensor) -> torch.Tensor:
    weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
    return F.linear(input, weight, self.bias)

class CakeNet(nn.Module):

  def __init__(self, user_n, item_n, knowledge_dim):
    #initialize class variables
    self.user_n = user_n
    self.item_n = item_n
    self.knowledge_dim = self.input_size = knowledge_dim
    self.nodes_layer1, self.nodes_layer2 = 512, 256 #arbitrary number of nodes based on EDUCDM defaults

    super(CakeNet, self).__init__()
    
    #Initialize the neural network input embeddings
    #embedding describing each students proficiency on each latent knowledge concept
    self.user_embedding = nn.Embedding(self.user_n, self.knowledge_dim)
    nn.init.xavier_normal_(self.user_embedding.weight) #initialize embedding weight according to xavier normal distribution
    #embedding describing how each item is related to each latent knowledge concept
    self.item_embedding = nn.Embedding(self.item_n, self.knowledge_dim)
    nn.init.xavier_normal_(self.user_embedding.weight)
    #embedding describing how well a given exercise discriminates between students with high vs. low proficiency
    self.discrimination_embedding = nn.Embedding(self.item_n, 1)
    nn.init.xavier_normal_(self.user_embedding.weight)

    #Map neural network layers and nodes
    self.input_layer = MonotonicLinear(self.input_size, self.nodes_layer1)
    self.dropout_layer1 = nn.Dropout(p=0.5)
    self.layer1 = MonotonicLinear(self.nodes_layer1, self.nodes_layer2)
    self.dropout_layer2 = nn.Dropout(p=0.5)
    self.layer2 = MonotonicLinear(self.nodes_layer2, 1) #single node output layer

    #initilaize layer weightss with xavier normal distribution
    for layer in [self.input_layer, self.layer1, self.layer2]:
      nn.init.xavier_normal_(layer.weight)
    
  #Define the forward function to dictate how data will be passed through the layers
  def forward(self, user_ids, item_ids):
    #given user ids, select the embeddings for those users and apply sigmoid
    specified_user_emb = torch.sigmoid(self.user_embedding(user_ids))
    #same for items and item discriminations
    specified_item_emb = torch.sigmoid(self.item_embedding(item_ids))
    specified_item_disc_emb = torch.sigmoid(self.discrimination_embedding(item_ids))

    #calculate the input value
    #likelyhood of answer is described by difference between student proficiency and question difficulty, multiplied by the questions ability to discriminate
    input = specified_item_disc_emb * (specified_user_emb - specified_item_emb)
    #pass the input through the defined neural network layers with sigmoid activation
    input = self.input_layer(input)
    input = torch.sigmoid(input)
    input = self.dropout_layer1(input)

    input = self.layer1(input)
    input = torch.sigmoid(input)
    input = self.dropout_layer2(input)

    input = self.layer2(input)
    output = torch.sigmoid(input)
    #no dropout layer for output

    return output.view(-1) #removes unecessary nesting to produce clean single output value

class CAKE(): #concept agnostic knowledge evaluation
  def __init__(self, user_n, item_n, knowledge_dim):
    self.cake_net = CakeNet(user_n, item_n, knowledge_dim)

  #training function is not significantly different from referenced NCDM implementation
  def train(self, train_data, test_data, epoch_n=10, device="cpu", lr=0.002):
    self.cake_net = self.cake_net.to(device)
    self.cake_net.train()
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(self.cake_net.parameters(), lr=lr)
    for epoch in range(epoch_n):
      epoch_losses = []
      batch_count = 0
      for batch_data in tqdm(train_data, f"Training on epoch{epoch}"):
        batch_count += 1
        #unpack batch data into tensors
        user_ids, item_ids, target = batch_data
        user_ids: torch.Tensor = user_ids.to(device)
        item_ids: torch.Tensor = item_ids.to(device)
        target: torch.Tensor = target.to(device)
        
        #use cake net to predict student performance
        predicted_values: torch.Tensor = self.cake_net(user_ids, item_ids)
        loss = loss_function(predicted_values, target)

        #use the information from the loss to train model weights/biases
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.mean().item())

      print(f"Epoch{epoch} average loss: {float(np.mean(epoch_losses))}")

      acc, auc, mae, rmse = self.eval(test_data, device=device)
      print("Epoch%d ACC:%.6f, AUC:%.6f, MAE:%.6f, RMSE:%.6f" % (epoch, acc, auc, mae, rmse))
    
  #evaluate function is not significantly different from referenced NCDM implementation 
  def eval(self, valid_data, device="cpu"):
    self.cake_net = self.cake_net.to(device)
    self.cake_net.eval()
    y_true, y_pred = [], []
    for batch_data in valid_data:
      #unpack batch data into tensors
      user_ids, item_ids, target = batch_data
      user_ids: torch.Tensor = user_ids.to(device)
      item_ids: torch.Tensor = item_ids.to(device)
      target: torch.Tensor = target.to(device)
      
      #use cake net to predict student performance
      predicted_values: torch.Tensor = self.cake_net(user_ids, item_ids)
      y_pred.extend(predicted_values.detach().cpu().tolist())
      y_true.extend(target.tolist())

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
    torch.save(self.cake_net.state_dict(), filepath)

  def load(self, filepath):
    self.cake_net.load_state_dict(torch.load(filepath))

  def getEmbeddings(self):
    item = self.cake_net.item_embedding.weight.data.cpu().numpy();
    user = self.cake_net.user_embedding.weight.data.cpu().numpy();
    disc = self.cake_net.discrimination_embedding.weight.data.cpu().numpy();

    return user, item, disc;

def run_CAKE(Q, train_data, test_data, valid_data, fold_n, dataset, runType, user_data=False, overwrite=False):
    print("Preparing to train on CAKE...")
    model = "CAKE"
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

    knowledge_n = np.max(list(map(int, knowledge_set))) + 1
    print(f'CAKE detects {user_n} users, {item_n} items, and {knowledge_n} knowledge concepts')
    
    cdm = CAKE(user_n, item_n, knowledge_n)

    filename = "models/snapshots/" + model + "/" + model + "_" + dataset + "_" + runType + "_" + str(fold_n) + ".snapshot"

    #transform data into a batch data loader
    batch_size = 32
    def transform(user, item, score, batch_size):
        item = item.reset_index(drop=True)
        user = user.reset_index(drop=True)
        score = score.reset_index(drop=True)
    
        data_set = TensorDataset(
            torch.tensor(user, dtype=torch.int64),
            torch.tensor(item, dtype=torch.int64),
            torch.tensor(score, dtype=torch.float32) #target must be float to satisfy BCELoss function
        )
        return DataLoader(data_set, batch_size=batch_size, shuffle=True)

    if((not os.path.exists(filename)) or overwrite):
        print("Training CAKE with new data")       
        
        train_set, valid_set, test_set = [
            transform(data["user_id"], data["item_id"], data["score"], batch_size)
            for data in [train_data, valid_data, test_data]
        ] #data iterates through train, test, valid to transform all of them
        
        #check for GPU and select or run on cpu
        device_str = "cpu"
        if torch.cuda.is_available():
            device_str = "cuda"
        cdm.train(train_set, test_set, epoch_n=3, device=device_str)
        cdm.save(filename)
    else:
        #still transform valid set for evaluation
        valid_set = transform(valid_data["user_id"], valid_data["item_id"], valid_data["score"], batch_size)

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
                user_valid_set = transform(user_valid["user_id"], user_valid["item_id"], user_valid["score"], batch_size)
        
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


def old_run_CAKE(Q, train_data, test_data, valid_data, filename): #Q is not used, input given to conform with other models
  user_n = np.max(train_data['user_id']) + 1
  item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])]) + 1
  knowledge_dim = 128 #can be changed, selected as power of 2 close to average number of kcs defined per dataset
  print(f'CAKE detects {user_n} users, {item_n} items, and assumes {knowledge_dim} knowledge concepts')

  #transform data into a batch data loader
  batch_size = 32
  def transform(user, item, score, batch_size):
    item = item.reset_index(drop=True)
    user = user.reset_index(drop=True)
    score = score.reset_index(drop=True)

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64),
        torch.tensor(item, dtype=torch.int64),
        torch.tensor(score, dtype=torch.float32) #target must be float to satisfy BCELoss function
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)
  
  train_set, test_set, valid_set = [
      transform(d["user_id"], d["item_id"], d["score"], batch_size)
      for d in [train_data, test_data, valid_data]
  ]

  #check for GPU and select or run on cpu
  device_str = "cpu"
  if torch.cuda.is_available():
      device_str = "cuda"
      print("Using GPU!")
      
  cake = CAKE(user_n, item_n, knowledge_dim)
  cake.train(train_set, test_set, epoch_n=3, device=device_str)
  
  cake.save("cake.snapshot")
  cake.load("cake.snapshot")
  acc, auc, mae, rmse = cake.eval(valid_set)
  print("ACC:%.6f, AUC:%.6f, MAE:%.6f, RMSE:%.6f" % (acc, auc, mae, rmse))
  user_emb, item_emb, disc_emb = cake.getEmbeddings();
  #save the embeddings as np arrays for further use
  arr_file_count = 0
  arr_path = f'embeddings/{len(Q.columns)}_CAKE_embeddings_'
  while(os.path.exists(arr_path + str(arr_file_count) + ".npz")):
    arr_file_count += 1
  np.savez(arr_path + str(arr_file_count), user=user_emb, item=item_emb, disc=disc_emb)
  return acc, auc, mae, rmse

def user_evaluation(Q, train_data, test_data, valid_data, model, fold_n, dataset, runType):
    return run_CAKE(Q, train_data, test_data, valid_data, "cake.snapshot")
