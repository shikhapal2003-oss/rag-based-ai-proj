import requests
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    r=requests.post("http://localhost:11434/api/embed",json={
                    "model":"bge-m3",
                    "input":text_list})
    embedding=r.json()['embeddings']
    return embedding

jsons=os.listdir("jsons")     #list all the json
my_dicts=[]
chunk_id=0

for json_file in jsons:
    with open(f"jsons/{json_file}")as f:  
        content=json.load(f)
    embeddings=create_embedding([c['text'] for c in content['chunks']])

    for i,chunk in enumerate(content['chunks']):
        chunk['chunk_id']=chunk_id
        chunk_id+=1
        chunk['embedding']=embeddings[i]
        my_dicts.append(chunk)
        
    
# print(my_dicts)
df=pd.DataFrame.from_records(my_dicts)
joblib.dump(df,'embeddings.joblib')
# incoming_query=input('Ask a question:')
# query_embeddigs=create_embedding([incoming_query])[0]
# print(query_embeddigs)

# #find similarities of question_embeddingwith other embedding
# # print(np.vstack(df['embedding'].values))
# # print(np.vstack (df['embedding']).shape)

# similarities=cosine_similarity(np.vstack (df['embedding']),[query_embeddigs]).flatten()
# print(similarities)
# max_index=similarities.argsort()[::-1]
# print(max_index)
# new_df=df.loc[max_index]
# print(new_df[["title","numbers","text"]])
# print(new_df.columns)

