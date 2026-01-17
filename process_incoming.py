import joblib
import requests
import pandas as pf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from read_chunks import create_embedding

def create_embedding(text_list):
    r=requests.post("http://localhost:11434/api/embed",json={
                    "model":"bge-m3",
                    "input":text_list})
    embedding=r.json()['embeddings']
    return embedding


def inference(prompt):
    r=requests.post("http://localhost:11434/api/generate",json={
                    "model":"llama3.2",
                    "prompt": prompt,
                    "stream":False
                    })
    
    response=r.json()
    print(response)   
    return(response)
    


df=joblib.load('embeddings.joblib')


incoming_query=input('Ask a question:')
query_embeddigs=create_embedding([incoming_query])[0]
print(query_embeddigs)

#find similarities of question_embeddingwith other embedding
# print(np.vstack(df['embedding'].values))
# print(np.vstack (df['embedding']).shape)

similarities=cosine_similarity(np.vstack (df['embedding']),[query_embeddigs]).flatten()
print(similarities)
max_index=similarities.argsort()[::-1]
print(max_index)
new_df=df.loc[max_index]
# print(new_df[["title","numbers","text"]])
prompt=f'''i am teaching web development using sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["numbers","title","start","end","text"]].to_json(orient="records")}
----------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way(dont answer the above format,it is just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course.
'''
with open('prompt.txt',"w")as f:
    f.write(prompt)

response=inference(prompt)["response"]
print(response)

with open("response.txt","w")as f:
    f.write(response)