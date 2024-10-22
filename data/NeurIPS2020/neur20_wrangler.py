import os
import pandas as pd
import numpy as np
import time
import json
from tqdm import tqdm

path = os.path.dirname(__file__) #path to the wrangler from the file it is imported to
q_data = os.path.join(path, "data/metadata/question_metadata_task_1_2.csv")
train_data = os.path.join(path, "data/train_data/train_task_1_2.csv")
test_public_data = os.path.join(path, "data/test_data/test_public_answers_task_1.csv")
test_private_data = os.path.join(path, "data/test_data/test_private_answers_task_1.csv")
q_matrix_output_file = os.path.join(path, "q_matrix.csv")
response_matrix_output_file = os.path.join(path, "response_matrix.csv")

MAX_USERS = 5000 #problem and user caps are set to prevent attempted creation of a 25Gb array for some models
MAX_PROBLEMS = 10000

def wrangle(overwrite=False):
    if(overwrite or not os.path.exists(q_matrix_output_file) or not os.path.exists(response_matrix_output_file)):
        print("Wrangling neurIPS Data")

        #read the data
        train = pd.read_csv(train_data)
        test_public = pd.read_csv(test_public_data)
        test_private = pd.read_csv(test_private_data)
        df = pd.concat([train, test_public, test_private], axis=0, ignore_index=True)
        df = df.rename(columns={'UserId':'user_id', 'QuestionId':'item_id', 'IsCorrect':'score'})
        df['score'] = df['score'].astype(int)
        
        #drop NA
        df = df.dropna(how='any')

        #drop any questions that aren't in the Q matrix
        Q = pd.read_csv(q_matrix_output_file) if os.path.exists(q_matrix_output_file) else createQ()
        labeled = Q['item_id'].unique()
        df = df[df['item_id'].isin(labeled)]

        #drop duplicates
        df = df.drop_duplicates(subset=["user_id", "item_id"])

        #Truncate to top 5000 users and top 10000 problems
        top_user = df['user_id'].value_counts().nlargest(MAX_USERS).index
        df = df[df['user_id'].isin(top_user)]

        top_probs = df['item_id'].value_counts().nlargest(MAX_PROBLEMS).index
        df = df[df['item_id'].isin(top_probs)]

        min_problem_count = 15
        min_user_count = 15

        #to achive a minimum of 15 exampes for users and problems trim the dataset recursively until both conditions are met
        #could result in no remaining data in theory but practically impossible
        while True:
            problem_counts = df["item_id"].value_counts()
            df = df[df["item_id"].isin(problem_counts.index[problem_counts >= min_problem_count])]

            user_counts = df["user_id"].value_counts()
            df = df[df["user_id"].isin(user_counts.index[user_counts >= min_user_count])]

            # Check the conditions to stop
            if (df["item_id"].value_counts().min() >= min_problem_count and 
                df["user_id"].value_counts().min() >= min_user_count):
                break
        
        #re-index problem and user ids
        df['item_id'], item_key = pd.factorize(df['item_id'], sort=True)
        #save the translation key to potentialy recover original ids
        with open(os.path.join(path,'keys/item_key.json'), 'w') as json_file:
            json.dump(dict(enumerate(item_key.tolist())), json_file)

        #swap to replace existing item_ids in dataframe with factorized versions
        key_item = {value: key for key, value in dict(enumerate(item_key.tolist())).items()}
        
        #remove unused items from Q and re index 0 -9999
        Q = Q[Q['item_id'].isin(item_key.tolist())]
        Q['item_id'].replace(key_item, inplace=True)
        
        df['user_id'], user_key = pd.factorize(df['user_id'], sort=True)
        #save the translation key to potentialy recover original ids
        with open(os.path.join(path,'keys/user_key.json'), 'w') as json_file:
            json.dump(dict(enumerate(user_key.tolist())), json_file) 
        
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
        print("Using pre-existing wrangled data for neurIPS")

def createQ(verbose=False):
    df = pd.read_csv(q_data)

    problems = df["QuestionId"].unique()
    
    qdata = {}
    qdata["item_id"] = problems
    
    skill_list = []
    #create a list of all skills
    for problem in problems: #change to no subscript for real life
        skills = eval(df[df["QuestionId"] == problem]['SubjectId'].iloc[0])
        for skill in skills:
            if skill not in skill_list:
                skill_list.append(skill)
                
    skill2new = {}

    #translate skills from sparse to dense number distribution
    for i in range(len(skill_list)):
        skill2new[skill_list[i]] = i

    with open(os.path.join(path,'keys/skill_key.json'), 'w') as json_file:
            json.dump(skill2new, json_file) 
    
    for skill in tqdm(skill_list, "parsing skills to build Q matrix: "):
      skill_problems = np.zeros(len(problems))
      for problem in problems:
        skills = eval(df[df["QuestionId"] == problem]['SubjectId'].iloc[0])
        if(skill in skills):
          skill_problems[np.where(problems == problem)[0]] = 1
      qdata[skill2new[skill]] = skill_problems

    Q = pd.DataFrame(qdata)
    return Q