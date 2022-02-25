from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
from keras import backend as K
from keras.models import load_model
import cv2
from Hadees.utils import *
from google.cloud import vision
import torch
import io
import warnings
from utils import *
import time
import spacy
warnings.filterwarnings('ignore')


nlp = spacy.load('en_core_web_lg')
K.clear_session()
model_load = load_model('Hatespeech/BERT_HATESPEECH', compile=False)
K.clear_session()
model_load_2 = load_model('Islamophobia/BERT_ISLAMOPHOBIA', compile=False)


#Application
app = FastAPI()

#Hatespeech
class Entities_1(BaseModel):
    text: str
class EntitesOut_1(BaseModel):
    headings: str
    Probability: str
    Prediction: str
hate = hatespeech

#Hatespeech
@app.post('/hatespeech', response_model=EntitesOut_1)
async def hatespeechs(text:Entities_1):
    text = text.text
    if text == '' or text == None:
        raise HTTPException(status_code=404, detail='Enter the Text')
    # model_load = await hatespeech()
    tokens = hate.tokenizer(text, max_length=150, truncation=True, 
                       padding='max_length', 
                       add_special_tokens=True, 
                       return_tensors='tf')
    tokens = {'input_ids': tf.cast(tokens['input_ids'], tf.float64), 'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)}
    headings = '''Non-offensive', 'identity_hate', 'neither', 'obscene','offensive', 'sexism'''
    probs = model_load.predict(tokens)[0]
    pred = np.argmax(probs)
    pred = hate.label_encoder.inverse_transform([pred])
    return {"headings":headings,
            "Probability":str(np.round(probs,3)),
            "Prediction":str(pred)}

#ImageCaption
class Entities_2(BaseModel):
    path: str
class EntitesOut_2(BaseModel):
    caption: str

model = imageCaption.model
transform = imageCaption.transform

#ImageCaption
@app.post('/captions', response_model=EntitesOut_2)
def captions(inputs: Entities_2):
    raw_image = inputs.path
    if raw_image == None:
        raise HTTPException(status_code=404, detail='Enter the path of the image')
    img = cv2.imread(raw_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    image = transform(im_pil).unsqueeze(0)     #.to(device)   
    with torch.no_grad():
        caption = model.generate(image, sample=False, num_beams=3, max_length=62, min_length=5)
        return {'caption':caption[0]}

#Islamohpobia
class Entities_3(BaseModel):
    text: str
class EntitesOut_3(BaseModel):
    Prediction: str

#Islamophobia
@app.post('/islamophobia', response_model=EntitesOut_3)
async def Islamophobia(text:Entities_3):
    text = text.text
    if text == '' or text == None:
        raise HTTPException(status_code=404, detail='Enter the Text')
    model_check_s = False
    model_check_w = False
    model_check_n = False
    check = ''
    rule_answer = ''
    model_answer = ''
    labels = ['Strong Islamophobia', 'Weak Islamophobia', 'No-Islamophobia']
    try:
        sentence = islamophobia([text], nlp=nlp).clean_tweet()
        for t in sentence:
            tex = t
        sent = ' '.join(tex)
        o = islamophobia(sent,model=model_load_2).prep_data_model()
        rule_answer = islamophobia(sent,nlp=nlp).sim_check()
        model_answer = o[1]
        if rule_answer == model_answer:
            return model_answer
        elif rule_answer == labels[0] and model_answer in ('No-Islamophobia','Weak Islamophobia'):
            model_check_s = True
        else:
            if o[0][2]>0.05 and o[0][2]<0.50 and o[0][1] < o[0][2]:
                model_check_w=True
            elif o[0][2]<0.010 and o[0][1]<0.010:
                model_check_n=True
    except:
        pass;
    if model_check_w:
        check = 'Weak Islamophobia'
    elif model_check_s:
        check = 'Strong Islamophobia'
    elif model_check_n:
        check = 'No-Islamophobia'
    for label in labels:
        if label in {check,rule_answer,model_answer[0]}:
            return {
    "Prediction":str(label)
    }

#NudityDetection
class Entities_4(BaseModel):
    path: str
class EntitesOut_4(BaseModel):
    Prediction: dict
    
#NudityDetection
@app.post('/nudity', response_model=EntitesOut_4)
def nudity_(text:Entities_4):
    nudity()
    path = text.path
    if path == '' or path == None:
        raise HTTPException(status_code=404, detail='Enter the Path')
    client = vision.ImageAnnotatorClient()
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.safe_search_detection(image=image)
    safe = response.safe_search_annotation

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'Contact owner, Fatum technologies'.format(
                response.error.message))

    return {
        'Prediction': {
        'adult':likelihood_name[safe.adult],
        'medical':likelihood_name[safe.medical],
        'spoofed':likelihood_name[safe.spoof],
        'violence':likelihood_name[safe.violence],
        'racy':likelihood_name[safe.racy]
        }
    }

#Tags
class Entities_5(BaseModel):
    text: str
class EntitesOut_5(BaseModel):
    tags: dict

#Tags
@app.post('/tags', response_model=EntitesOut_5)
def Tags(text:Entities_5):
    text = text.text
    if text == '' or text == None:
        raise HTTPException(status_code=404, detail='Enter the Text')
    tag = tags(text,nlp).tokens()
    tags_ = tags(token=tag).get_token()
    date = tags(token=tag).get_DT()
    return {
        'tags':{'Tags':tags_,'DateTime':date}
    }

#Translation
class Entities_6(BaseModel):
    text: str
    dest: str
class EntitesOut_6(BaseModel):
    output: (str or bool)

translator = translation.translator    

@app.post('/translation', response_model=EntitesOut_6)
def get_translation(inputs: Entities_6):
    text = inputs.text
    dest = inputs.dest
    if text == '' or text == None or dest=='' or dest == None:
        raise HTTPException(status_code=404, detail='Enter the Text or destination')
    if len(text.split(' ')) > 0:
        src = translator.detect(text).lang
        if src == "en" and dest == "ur":
            translated = translator.translate(text, src=src, dest=dest)
            return {'output': translated.text}
        elif src == "en" and dest == "hi":
            translated = translator.translate(text, src=src, dest=dest)
            return {'output':translated.pronunciation}
        elif (src == "hi" or src == "ur") and dest == "en":
            translated = translator.translate(text, src=src, dest=dest)
            return {'output': translated.text}
        else:
            return {'output':False}
    else:
        return {'output':False}

#Ahadees similarity
class Entities(BaseModel):
    text: list

class EntitesOut(BaseModel):
    Answer: dict
    
data = similarity.data
X = similarity.X
v = similarity.v

@app.post('/similarity', response_model=EntitesOut)
def similarity_(text:Entities):
    text = text.text
    if text == '' or text == None:
        raise HTTPException(status_code=404, detail='Enter the Text')
    sim_vecs, cosine_similarities = calculate_similarity(X, v, text,top_k=1)
    a,b,c = show_similar_documents(data,cosine_similarities, sim_vecs)
    return {'Answer': {
        'Similarity':a,
        'Hadees Number': b,
        'Hadees': c,
    }}