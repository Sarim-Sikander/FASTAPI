from fastapi import FastAPI
from pydantic import BaseModel
from starlette.status import HTTP_302_FOUND,HTTP_303_SEE_OTHER
from preprocess import *

class Entities(BaseModel):
    text: str

class EntitesOut(BaseModel):
    Prediction: str


app = FastAPI()

@app.post('/islamophobia', response_model=EntitesOut)
def prep_data(text:Entities):
    text = text.text
    model_check_s = False
    model_check_w = False
    model_check_n = False
    check = ''
    rule_answer = ''
    model_answer = ''
    labels = ['Strong Islamophobia', 'Weak Islamophobia', 'No-Islamophobia']
    try:
        sentence = clean_tweet([text])
        for t in sentence:
            tex = t
        sent = ' '.join(tex)
        o = prep_data_model(sent)
        rule_answer = sim_check(sent)
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