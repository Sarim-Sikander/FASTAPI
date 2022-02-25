from logging import raiseExceptions
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
# from googletrans import Translator

class Entities(BaseModel):
    text: str
    dest: str

class EntitesOut(BaseModel):
    output: (str or bool)

translator=pickle.load(open('translator.pkl', 'rb'))
translator=translator()

app = FastAPI()

@app.post('/translation', response_model=EntitesOut)
def get_translation(inputs: Entities):
    text = inputs.text
    dest = inputs.dest
    
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