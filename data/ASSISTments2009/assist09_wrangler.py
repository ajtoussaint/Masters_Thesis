import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

path = os.path.dirname(__file__) #path to the wrangler from the file it is imported to
input_file = os.path.join(path, "data/assistments_2009_2010.csv") #finds the data file relative to this file
q_matrix_output_file = os.path.join(path, "q_matrix.csv") #generates q and response within the same directory as the wrangler
response_matrix_output_file = os.path.join(path, "response_matrix.csv")

MAX_USERS = 5000 #problem and user caps are set to prevent attempted creation of a 25Gb array for some models
MAX_PROBLEMS = 10000

def wrangle(overwrite=False):
    # if overwrite is true exisitng data is replaced
    # if overwrite is false you can use the existing files if they are there and don't need to actually run the code
    if(overwrite or not os.path.exists(q_matrix_output_file) or not os.path.exists(response_matrix_output_file)):
        print("Wrangling ASSIST09 data...")
        
        #read in the data
        df = pd.read_csv(input_file)
    
        #select relevant columns and remove any skilless tasks
        df = df[["user_id", "problem_id", "correct", "list_skill_ids"]].dropna(subset=["list_skill_ids"])
        df = df.drop_duplicates(subset=["user_id", "problem_id"])

        min_problem_count = 15 #only include users that answer at least 15 problems
        min_user_count = 15 #only include problems that are answered by at least 15 users

        #to achive a minimum of 15 exampes for users and problems trim the dataset recursively until both conditions are met
        #could result in no remaining data in theory but practically impossible
        while True:
            problem_counts = df["problem_id"].value_counts()
            df = df[df["problem_id"].isin(problem_counts.index[problem_counts >= min_problem_count])]

            user_counts = df["user_id"].value_counts()
            df = df[df["user_id"].isin(user_counts.index[user_counts >= min_user_count])]

            # Check the conditions to stop
            if (df["problem_id"].value_counts().min() >= min_problem_count and 
                df["user_id"].value_counts().min() >= min_user_count):
                break
    
        #Truncate to top 5000 users and top 10000 problems
        top_user = df['user_id'].value_counts().nlargest(MAX_USERS).index
        df = df[df['user_id'].isin(top_user)]

        top_probs = df['problem_id'].value_counts().nlargest(MAX_PROBLEMS).index
        df = df[df['problem_id'].isin(top_probs)]
    
        #re-index problem_ids and user ids to go from 0 - (n-1)
        df['score'] = df['correct'].astype(int)

        df['item_id'], item_key = pd.factorize(df['problem_id'], sort=True)
         #save the translation key to potentialy recover original ids
        with open(os.path.join(path,'keys/item_key.json'), 'w') as json_file:
            json.dump(dict(enumerate(item_key.tolist())), json_file) 
        
        df['user_id'], user_key = pd.factorize(df['user_id'], sort=True)
        #save the translation key to potentialy recover original ids
        with open(os.path.join(path,'keys/user_key.json'), 'w') as json_file:
            json.dump(dict(enumerate(user_key.tolist())), json_file)  
    
        Q = createQ(df)
        Y = df[["user_id", "item_id", "score"]]

        #gather statistics for display
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
        
        print(f'Created a response matrix with {n_user} users ({min_user} - {max_user}), {n_items} items ({min_item} - {max_item}), and {len(Y)} responses')
        print(f'Created a Q matrix with {len(Q)} items ({min_prob} - {max_prob}) and {len(Q.columns) - 1} knowledge concepts ({min_kc} - {max_kc})') 
        
        Q.to_csv(q_matrix_output_file, index=False)
        Y.to_csv(response_matrix_output_file, index=False)
        
    else:
        print("Using pre-existing wrangled data for ASSIST09")

def createQ(df):
    #creates a "Q" matrix relating items to knowledge concepts as one hot vectors
    problems = df["item_id"].unique()
    
    qdata = {}
    qdata["item_id"] = problems

    
    all_skills = df['list_skill_ids'].str.split(';').explode().unique()
    skill_list = list(all_skills)
    
    skill2new = {}

    #translate skills from sparse to dense number distribution
    for i in range(len(skill_list)):
        skill2new[skill_list[i]] = i

    #save the translation key to potentialy recover original ids
    skill_dict = {int(k): int(v) for k, v in skill2new.items()}
    with open(os.path.join(path,'keys/skill_key.json'), 'w') as json_file:
        json.dump(skill_dict, json_file)  

    for skill in tqdm(skill_list, "parsing skills to build Q matrix: "):
      skill_problems = np.zeros(len(problems))
      for problem in problems:
        skills = df.loc[df["item_id"] == problem]["list_skill_ids"].tolist()[0].split(";")
        if(skill in skills):
          skill_problems[np.where(problems == problem)[0]] = 1
      qdata[skill2new[skill]] = skill_problems
    
    Q = pd.DataFrame(qdata)
    
    return Q