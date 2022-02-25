from numpy.core.arrayprint import str_format
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from starlette.responses import RedirectResponse
from starlette.status import HTTP_302_FOUND,HTTP_303_SEE_OTHER
import spacy
import pickle
import re
from string import punctuation

class Entities(BaseModel):
    text: str

class EntitesOut(BaseModel):
    tags: dict

nlp = spacy.load("en_core_web_lg",disable=["tok2vec", "parser"])

def tokens(sentence,nlp):
    sentence_nlp = nlp(sentence)
    return [(word.lemma_, word.ent_type_) for word in sentence_nlp if word.ent_type_]

def get_token(sentence):
    words = []
    for word, ner in sentence:
        if ner in ['WORK_OF_ART','LAW','PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT']:
            if word not in punctuation:
                words.append(word)
    return words

def get_DT(sentence):
    words = []
    for word, ner in sentence:
        if ner in ['DATE','TIME']:
            words.append(word)
    return words

def getdict(sentence):
    return {"entities": get_token(sentence),
            "dates": get_DT(sentence)}

app = FastAPI()

@app.post('/tags', response_model=EntitesOut)
def prep_data(text:Entities):
    text = text.text
    tag = tokens(text, nlp)
    tags = getdict(tag)
    return {
        'tags':tags
    }