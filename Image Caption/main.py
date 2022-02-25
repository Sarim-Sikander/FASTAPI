from PIL import Image
import torch
import cv2
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
# from googletrans import Translator

class Entities(BaseModel):
    path: str

class EntitesOut(BaseModel):
    caption: str

transform = pickle.load(open('transform.pkl','rb'))
model = pickle.load(open('model.pkl', 'rb'))
model.eval()

app = FastAPI()

@app.post('/captions', response_model=EntitesOut)
def captions(inputs: Entities):
    raw_image = inputs.path
    img = cv2.imread(raw_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    image = transform(im_pil).unsqueeze(0)     #.to(device)   
    with torch.no_grad():
        caption = model.generate(image, sample=False, num_beams=3, max_length=62, min_length=5)
        return {'caption':caption[0]}