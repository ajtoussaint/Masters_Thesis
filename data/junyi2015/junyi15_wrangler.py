import os
import pandas as pd
import numpy as np
import time
import json
from tqdm import tqdm

path = os.path.dirname(__file__) #path to the wrangler from the file it is imported to
q_data=os.path.join(path, 'data/junyi_Exercise_table.csv')
response_data = os.path.join(path, 'data/junyi_ProblemLog_original.csv')
q_matrix_output_file = os.path.join(path, "q_matrix.csv")
response_matrix_output_file = os.path.join(path, "response_matrix.csv")

MAX_USERS = 5000 #problem and user caps are set to prevent attempted creation of a 25Gb array for some models
MAX_PROBLEMS = 10000

def wrangle(overwrite=False):
    print("Wrangling Junyi data...")
    for f in [q_data, response_data]:
        if not os.path.exists(f):
            print(f'missing {f}')
            return None
    if overwrite or (not os.path.exists(q_matrix_output_file)) or (not os.path.exists(response_matrix_output_file)):
        createMatricies()
    else:
        print("Using existing Q and response matrix for junyi")

def createMatricies():
    #read in the csv files
    qdf = pd.read_csv(q_data)
    ydf = pd.read_csv(response_data)
    qdf = qdf[['name', 'topic']] #prerequisites column is also interesting...next time
    ydf = ydf[['user_id', 'correct', 'exercise']]
    
    #clean qdf
    qdf = qdf.dropna(subset = ['topic'])
    qdf = qdf.drop_duplicates()
    problems = qdf['name'].unique()

    #clean ydf
    ydf = ydf.rename(columns = {'exercise':'item_id', 'correct':'score'})
    ydf = ydf.dropna(how='any')
    ydf['score'] = ydf['score'].astype(int)
    #only include itesm featured in Q
    ydf = ydf[ydf['item_id'].isin(problems)]

    #drop duplicates
    ydf = ydf.drop_duplicates(subset=["user_id", "item_id"])

    min_problem_count = 15
    min_user_count = 15

    #Truncate to top 5000 users and top 10000 problems
    top_user = ydf['user_id'].value_counts().nlargest(MAX_USERS).index
    ydf = ydf[ydf['user_id'].isin(top_user)]

    top_probs = ydf['item_id'].value_counts().nlargest(MAX_PROBLEMS).index
    ydf = ydf[ydf['item_id'].isin(top_probs)]

    #to achive a minimum of 15 exampes for users and problems trim the dataset recursively until both conditions are met
    #could result in no remaining data in theory but practically impossible
    while True:
        problem_counts = ydf["item_id"].value_counts()
        ydf = ydf[ydf["item_id"].isin(problem_counts.index[problem_counts >= min_problem_count])]

        user_counts = ydf["user_id"].value_counts()
        ydf = ydf[ydf["user_id"].isin(user_counts.index[user_counts >= min_user_count])]

        # Check the conditions to stop
        if (ydf["item_id"].value_counts().min() >= min_problem_count and 
            ydf["user_id"].value_counts().min() >= min_user_count):
            break

    #Build Q matrix
    problems = qdf['name'].unique()
    name2id = {}
    skill2id = {}

    for pid in range(len(problems)):
        name2id[problems[pid]] = pid

    all_skills = qdf['topic'].unique()
    for sid in range(len(all_skills)):
        skill2id[all_skills[sid]] = sid
    
    q_new = qdf.replace(name2id)
    q_new.replace(skill2id, inplace=True)
    
    qdata={'item_id':list(name2id.values())}
    
    for skill in qdf['topic'].unique():
        skill_problems = np.zeros(len(qdata['item_id']))
        skill_problems[np.where(q_new['topic'] == skill2id[skill])[0]] = 1
        qdata[skill2id[skill]] = skill_problems#0s for all problems, 1s for with skill

    with open(os.path.join(path,'keys/skill_key.json'), 'w') as json_file:
            json.dump(skill2id, json_file)
        
    with open(os.path.join(path,'keys/item-name_key.json'), 'w') as json_file:
            json.dump(name2id, json_file)
    
    Q = pd.DataFrame(qdata)
    #end build Q matrix

    ydf['item_id'] = ydf['item_id'].map(name2id)

    ydf['item_id'], item_key = pd.factorize(ydf['item_id'], sort=True)
    #save the translation key to potentialy recover original ids
    with open(os.path.join(path,'keys/item_key.json'), 'w') as json_file:
        json.dump(dict(enumerate(item_key.tolist())), json_file)

    #swap to replace existing item_ids in dataframe with factorized versions
    key_item = {value: key for key, value in dict(enumerate(item_key.tolist())).items()}
    
    #remove unused items from Q and re index 0 -9999
    Q = Q[Q['item_id'].isin(item_key.tolist())]
    Q['item_id'].replace(key_item, inplace=True)
    
    ydf['user_id'], user_key = pd.factorize(ydf['user_id'], sort=True)
    #save the translation key to potentialy recover original ids
    with open(os.path.join(path,'keys/user_key.json'), 'w') as json_file:
        json.dump(dict(enumerate(user_key.tolist())), json_file) 

    Y = ydf[["user_id", "item_id", "score"]]

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
