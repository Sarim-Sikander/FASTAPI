from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
from utils import *
import warnings
import pickle
warnings.filterwarnings('ignore')

class Entities(BaseModel):
    text: list

class EntitesOut(BaseModel):
    Answer: dict
    
X = pickle.load(open('tfidfFeaturesX.pkl','rb'))
v = pickle.load(open('tfidfVectorsV.pkl','rb'))

data = pd.read_csv('data_of_ahadees.csv')
data = data[~data['Text'].isnull()].reset_index(drop=True)
data['Text'] = data['Text'].apply(lambda x: str(x).strip())

app = FastAPI()

@app.post('/similarity', response_model=EntitesOut)
def prep_data(text:Entities):
    text = text.text
    sim_vecs, cosine_similarities = calculate_similarity(X, v, text,top_k=1)
    a,b,c = show_similar_documents(data,cosine_similarities, sim_vecs)
    return {'Answer': {
        'Similarity':a,
        'Hadees Number': b,
        'Hadees': c,
    }}