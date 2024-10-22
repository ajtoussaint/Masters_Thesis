import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

def basicSplit(df, target='score'):
    users = df['user_id'].unique()
    
    n = 5
    
    kf = KFold(n_splits=n, shuffle=True, random_state=217)
    
    dfs = [{"train": [], "test": [], "valid": []} for _ in range(n)]

    
    for uid in tqdm(users, "Performing 5 fold split on users"):
        
        #perform a KFold split on each individual users data
        udf = df[df['user_id'] == uid]
    
        for fold_index, (train_index, test_index) in enumerate(kf.split(udf)):
            train, test = udf.iloc[train_index], udf.iloc[test_index]
            #append this users training/testing data to the total collection
            dfs[fold_index]["train"].append(train)
            #if only 1 value in test add it to the test set and do nothing to valid
            if len(test) < 2:
                dfs[fold_index]["test"].append(test)
            else:
                #if more than 1 value split the dataframe randomly between train and test
                shuffle = test.sample(frac=1, random_state=217).reset_index(drop=True)
                half_size = len(shuffle) // 2
                test_test = shuffle.iloc[:half_size]
                test_valid = shuffle.iloc[half_size:]

                dfs[fold_index]["test"].append(test_test)
                dfs[fold_index]["valid"].append(test_valid)

    #merge all dataframes in train/test/valid respectively into a single dataframe including all users
    for fold in dfs:
        fold["train"] = pd.concat(fold["train"], ignore_index=True)
        fold["test"] = pd.concat(fold["test"], ignore_index=True)
        fold["valid"] = pd.concat(fold["valid"], ignore_index=True)      
        
    return dfs

def sampledSplit(df, target='score'):
    print("Performing sampled split...")
    users = df['user_id'].unique()
    
    n = 5
    
    kf = KFold(n_splits=n, shuffle=True, random_state=217)
    
    dfs = [{"train":[], 'test':[], 'valid':[]} for _ in range(n)]
        
    for uid in tqdm(users, "Performing 5 fold sampled split on users"):
        #perform a KFold split on each individual users data
        udf = df[df['user_id'] == uid]
        udf = udf.sample(frac=0.4) #Lowest split can go to guarantee 5-fold is possible with a minimum of 15 examples per student
        
        for fold_index, (train_index, test_index) in enumerate(kf.split(udf)):
            train, test = udf.iloc[train_index], udf.iloc[test_index]
            #append this users training/testing data to the total collection
            dfs[fold_index]["train"].append(train)
            if len(test) < 2:
                dfs[fold_index]["test"].append(test)
            else:
                #if more than 1 value split the dataframe randomly between train and test
                shuffle = test.sample(frac=1, random_state=217).reset_index(drop=True)
                half_size = len(shuffle) // 2
                test_test = shuffle.iloc[:half_size]
                test_valid = shuffle.iloc[half_size:]

                dfs[fold_index]["test"].append(test_test)
                dfs[fold_index]["valid"].append(test_valid)
                
    #merge all dataframes in train/test/valid respectively into a single dataframe including all users            
    for fold in dfs:
        fold["train"] = pd.concat(fold["train"], ignore_index=True)
        fold["test"] = pd.concat(fold["test"], ignore_index=True)
        fold["valid"] = pd.concat(fold["valid"], ignore_index=True) 

    return dfs

def correctSaturatedSplit(df, target='score'):
    users = df['user_id'].unique()
    
    n = 5
    
    kf = KFold(n_splits=n, shuffle=True, random_state=217)
    
    dfs = [{"train": [], "test": [], "valid": []} for _ in range(n)]
    
    #not going to be true k-fold
    for uid in tqdm(users, "Creating 5 datasets with correct answers in training"):
        for fold_index in range(n):
            udf = df[df['user_id'] == uid]
            udf = udf.sample(frac=1).reset_index(drop=True)#shuffle random
            #for this user create a random distribution of train and test 80/20 with train containing a greater ratio of correct answers
            udf_ones = udf[udf[target] == 1]
            udf_zeros = udf[udf[target] == 0]

            train = pd.concat([udf_ones.sample(frac=0.9), udf_zeros.sample(frac=0.65)], axis=0, ignore_index=True)
            test = udf.merge(train, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)

            dfs[fold_index]["train"].append(train)
            if len(test) < 2:
                dfs[fold_index]["test"].append(test)
            else:
                #if more than 1 value split the dataframe randomly between train and test
                shuffle = test.sample(frac=1, random_state=217).reset_index(drop=True)
                half_size = len(shuffle) // 2
                test_test = shuffle.iloc[:half_size]
                test_valid = shuffle.iloc[half_size:]

                dfs[fold_index]["test"].append(test_test)
                dfs[fold_index]["valid"].append(test_valid)

    #merge all dataframes in train/test/valid respectively into a single dataframe including all users
    for fold in dfs:
        fold["train"] = pd.concat(fold["train"], ignore_index=True)
        fold["test"] = pd.concat(fold["test"], ignore_index=True)
        fold["valid"] = pd.concat(fold["valid"], ignore_index=True) 

    return dfs

def incorrectSaturatedSplit(df, target='score'):
    print("Performing split with extra incorrect responses in training...")
    users = df['user_id'].unique()
    
    n = 5
    
    kf = KFold(n_splits=n, shuffle=True, random_state=217)
    
    dfs = [{"train": [], "test": [], "valid": []} for _ in range(n)]
    
    #not going to be true k-fold
    for uid in tqdm(users, "Creating 5 datasets with wrong answers in training"):
        for fold_index in range(n):
            udf = df[df['user_id'] == uid]
            udf = udf.sample(frac=1).reset_index(drop=True)#shuffle random
            #for this user create a random distribution of train and test 80/20 with train containing a greater ratio of correct answers
            udf_ones = udf[udf[target] == 1]
            udf_zeros = udf[udf[target] == 0]

            #alternatively, could assing one correct and one incorrect to each set train/test
            #then compute based on the remaining length...

            #fractions are determined assuming 60/40 correct/incorrect, may be a problem outside of assistments...
            train = pd.concat([udf_ones.sample(frac=(44/60)), udf_zeros.sample(frac=0.9)], axis=0, ignore_index=True)
            test = udf.merge(train, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)

            dfs[fold_index]["train"].append(train)
            if len(test) < 2:
                dfs[fold_index]["test"].append(test)
            else:
                #if more than 1 value split the dataframe randomly between train and test
                shuffle = test.sample(frac=1, random_state=217).reset_index(drop=True)
                half_size = len(shuffle) // 2
                test_test = shuffle.iloc[:half_size]
                test_valid = shuffle.iloc[half_size:]

                dfs[fold_index]["test"].append(test_test)
                dfs[fold_index]["valid"].append(test_valid)
                
    for fold in dfs:
        fold["train"] = pd.concat(fold["train"], ignore_index=True)
        fold["test"] = pd.concat(fold["test"], ignore_index=True)
        fold["valid"] = pd.concat(fold["valid"], ignore_index=True) 
    
    return dfs

def undersample(df, train_correct):
    #used to return no values in some scenarios
    empty = (pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns))
    
    train_ratio = 0.8
    test_ratio = 1 - train_ratio
    #assume that test correct is inverse of train for the purposes of this experiment
    test_correct = 1 - train_correct

    #determine the ideal tuple ratios that would produce desired train/test ratios
    tuples_train_correct = train_ratio * train_correct
    tuples_train_incorrect = train_ratio * (1 - train_correct)
    tuples_test_correct = test_ratio * test_correct
    tuples_test_incorrect = test_ratio * (1- test_correct)

    #once the dataset is undersampled, determine which % of eahc type of answer is given to the train set
    correct_tuples_filtered_train = tuples_train_correct / (tuples_train_correct + tuples_test_correct)
    incorrect_tuples_filtered_train = tuples_train_incorrect / (tuples_train_incorrect + tuples_test_incorrect)

    n_tuples = len(df)
    
    df_correct = df[df['score'] == 1]
    df_incorrect = df[df['score'] == 0]
    
    n_correct = len(df_correct)
    n_incorrect = len(df_incorrect)

    #avoid divide by 0
    if(n_correct == 0 or n_incorrect == 0):
        return empty

    actual_correct_ratio = n_correct / n_tuples

    desired_correct_ratio = tuples_train_correct + tuples_test_correct

    corr_sample_factor = desired_correct_ratio * n_incorrect / ((1- desired_correct_ratio) * n_correct)
    incorr_sample_factor = (1- desired_correct_ratio)*n_correct / (desired_correct_ratio * n_incorrect)

    if corr_sample_factor > 1:
        df_incorrect = df_incorrect.sample(frac = incorr_sample_factor)
    else:
        df_correct = df_correct.sample(frac = corr_sample_factor)

    train_df = pd.concat([df_correct.sample(frac = correct_tuples_filtered_train), df_incorrect.sample(frac=incorrect_tuples_filtered_train)])
    testing_df = pd.concat([df_correct, df_incorrect]).merge(train_df, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)

    test_df = testing_df
    valid_df = pd.DataFrame(columns = train_df.columns)

    #divide responses between test and valid dfs
    if len(testing_df) > 1:
        testing_correct = testing_df[testing_df['score'] == 1]
        testing_incorrect = testing_df[testing_df['score'] == 0]

        test_df = pd.concat([testing_correct.sample(frac=0.5), testing_incorrect.sample(frac=0.5)])
        valid_df = pd.concat([testing_correct, testing_incorrect]).merge(test_df, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
    return train_df, test_df, valid_df

def undersampleEven(df):
    print("Performing split with undersampling to balance correct/incorrect responses")

    ratio = 0.5
    
    users = df['user_id'].unique()
    n = 5
    dfs = [{"train": [], "test": [], "valid": []} for _ in range(n)]

    for uid in tqdm(users, "Creating 5 datasets with even answer types"):
        for fold_index in range(n):
            udf = df[df['user_id'] == uid]
            train_udf, test_udf, valid_udf = undersample(udf,ratio)

            dfs[fold_index]["train"].append(train_udf)
            dfs[fold_index]["test"].append(test_udf)
            dfs[fold_index]["valid"].append(valid_udf)
    
    for fold in dfs:        
        fold["train"] = pd.concat(fold["train"], ignore_index=True)
        fold["test"] = pd.concat(fold["test"], ignore_index=True)
        fold["valid"] = pd.concat(fold["valid"], ignore_index=True) 
    
    return dfs

def undersampleCorrect(df):
    print("Performing split with undersampling to balance correct/incorrect responses")

    ratio = 0.6
    
    users = df['user_id'].unique()
    n = 5
    dfs = [{"train": [], "test": [], "valid": []} for _ in range(n)]

    for uid in tqdm(users, "Creating 5 datasets with even answer types"):
        for fold_index in range(n):
            udf = df[df['user_id'] == uid]
            train_udf, test_udf, valid_udf = undersample(udf,ratio)

            dfs[fold_index]["train"].append(train_udf)
            dfs[fold_index]["test"].append(test_udf)
            dfs[fold_index]["valid"].append(valid_udf)
    
    for fold in dfs:        
        fold["train"] = pd.concat(fold["train"], ignore_index=True)
        fold["test"] = pd.concat(fold["test"], ignore_index=True)
        fold["valid"] = pd.concat(fold["valid"], ignore_index=True) 
    
    return dfs

def undersampleIncorrect(df):
    print("Performing split with undersampling to balance correct/incorrect responses")

    ratio = 0.4
    
    users = df['user_id'].unique()
    n = 5
    dfs = [{"train": [], "test": [], "valid": []} for _ in range(n)]

    for uid in tqdm(users, "Creating 5 datasets with even answer types"):
        for fold_index in range(n):
            udf = df[df['user_id'] == uid]
            train_udf, test_udf, valid_udf = undersample(udf,ratio)

            dfs[fold_index]["train"].append(train_udf)
            dfs[fold_index]["test"].append(test_udf)
            dfs[fold_index]["valid"].append(valid_udf)
    
    for fold in dfs:        
        fold["train"] = pd.concat(fold["train"], ignore_index=True)
        fold["test"] = pd.concat(fold["test"], ignore_index=True)
        fold["valid"] = pd.concat(fold["valid"], ignore_index=True) 
    
    return dfs
    

from data.ASSISTments2009 import assist09_wrangler 
from data.NeurIPS2020 import neur20_wrangler 
from data.junyi2015 import junyi15_wrangler 
from data.ASSISTments2012 import assist12_wrangler
#definitions of various file names and paths
q_matrix_file = "q_matrix.csv"
response_matrix_file = "response_matrix.csv"

dir_path = os.path.dirname(__file__)

paths = {
    "ASSIST09" : "ASSISTments2009",
    "NEUR20" : "NeurIPS2020",
    "JUNYI15" : "junyi2015",
    "ASSIST12" : "ASSISTments2012"
}

wranglers = {
    "ASSIST09" : assist09_wrangler,
    "NEUR20" : neur20_wrangler,
    "JUNYI15" : junyi15_wrangler,
    "ASSIST12": assist12_wrangler
}

splitters = {
    "basic" : basicSplit,
    "sampled" : sampledSplit,
    "correctSaturated":correctSaturatedSplit,
    "incorrectSaturated":incorrectSaturatedSplit,
    "undersampleEven":undersampleEven,
    "undersampleCorrect":undersampleCorrect,
    "undersampleIncorrect":undersampleIncorrect,
}

#recieves input on dataset, they type of run to be executed
#returns the Q matrix as a dataframe, and an array containing objects with train/test datasets
def prepareData(dataset = "ASSIST09", runType = "basic", overwrite=False):
    data_path = os.path.join(dir_path, paths[dataset])
    q_matrix_path = os.path.join(data_path, q_matrix_file)

    if(not os.path.exists(q_matrix_path)):
        print("Could not find Q matrix, calling wrangler")
        #have wrangler run ot make the Q matrix
        wranglers[dataset].wrangle(overwrite=overwrite)

    Q = pd.read_csv(q_matrix_path)
    
    existing_prepared_path = os.path.join(data_path, "splits/" + runType + "_" + "split_data.csv")
    if(os.path.exists(existing_prepared_path) and not overwrite):
        print(f'Using existing split from {dataset} for {runType} format')
        existing_df = pd.read_csv(existing_prepared_path)
        #unpack the saved dataframe into the object format
        res = []
        for i in tqdm(range(existing_df['fold_n'].max() + 1), "Loading existing data:"):
            obj = {}
            for k in existing_df['dataset'].unique():
                obj[k] = existing_df[(existing_df['dataset'] == k) & (existing_df['fold_n'] == i)]
                obj[k] = obj[k].drop(['fold_n', 'dataset'], axis=1)
            res.append(obj)
            
        Q = pd.read_csv(q_matrix_path)

        printStats(Q, res)
        
        return Q, res
        
    else:
        print(f'preparing data from {dataset} for {runType} format')
    
    
        response_matrix_path = os.path.join(data_path, response_matrix_file)
    
        if(not os.path.exists(response_matrix_path)):
            print("Could not find Response matrix, calling wrangler")
            #have wrangler run ot make the Response matrix - facit
            wranglers[dataset].wrangle(overwrite=overwrite)
    
        Y = pd.read_csv(response_matrix_path)

        res = splitters[runType](Y)

        #Save res as an existing data split
        master_data = []
        for i in tqdm(range(len(res)), "saving data for reuse..."):
            for k in res[i].keys():
                res[i][k]['fold_n'] = i
                res[i][k]['dataset'] = k
                master_data.append(res[i][k])
        
        master_df = pd.concat(master_data, ignore_index=True)
        master_df.to_csv(existing_prepared_path)

        #allows user to check correctness
        printStats(Q, res)
        
        return Q, res
    
def printStats(Q, res):

    Y = pd.concat([res[0]['train'], res[0]['test'], res[0]['valid']])

    n_user = len(Y['user_id'].unique())
    min_user = min(Y['user_id'].unique())
    max_user = max(Y['user_id'].unique())

    n_items = len(Y['item_id'].unique())
    min_item = min(Y['item_id'].unique())
    max_item = max(Y['item_id'].unique())

    min_prob = min(Q['item_id'].unique())
    max_prob = max(Q['item_id'].unique())

    kc_columns = [int(col) for col in Q.columns if col != 'item_id']
    min_kc = min(kc_columns)
    max_kc = max(kc_columns)
    
    print(f'Using a response matrix with {n_user} users ({min_user} - {max_user}), {n_items} items ({min_item} - {max_item}), and {len(Y)} responses')
    print(f'Using a Q matrix with {len(Q)} items ({min_prob} - {max_prob}) and {len(kc_columns)} knowledge concepts ({min_kc} - {max_kc})')


