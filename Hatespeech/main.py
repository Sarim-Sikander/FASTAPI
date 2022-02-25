from numpy.core.arrayprint import str_format
from fastapi import FastAPI
from tensorflow.keras.models import load_model
from pydantic import BaseModel
# import tensorflow_text as text
import tensorflow as tf
import pickle
import numpy as np
from starlette.responses import RedirectResponse
import os
from starlette.status import HTTP_302_FOUND,HTTP_303_SEE_OTHER
import warnings
warnings.filterwarnings('ignore')

class Entities(BaseModel):
    text: str

class EntitesOut(BaseModel):
    headings: str
    Probability: str
    Prediction: str

tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
# model_load = pickle.load(open("model_load.pkl", "rb"))
localhost_save_option = tf.saved_model.LoadOptions(experimental_io_device="/job:localhost")
model_load = load_model('BERT_HATESPEECH',options=localhost_save_option)
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

app = FastAPI()

@app.post('/predict', response_model=EntitesOut)
def prep_data(text:Entities):
    text = text.text
    tokens = tokenizer(text, max_length=150, truncation=True, 
                       padding='max_length', 
                       add_special_tokens=True, 
                       return_tensors='tf')
    tokens = {'input_ids': tf.cast(tokens['input_ids'], tf.float64), 'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)}
    headings = '''Non-offensive', 'identity_hate', 'neither', 'obscene','offensive', 'sexism'''
    probs = model_load.predict(tokens)[0]
    pred = np.argmax(probs)
    pred = label_encoder.inverse_transform([pred])
    return {"headings":headings,
            "Probability":str(np.round(probs,3)),
            "Prediction":str(pred)}

