from fastapi import FastAPI
from pydantic import BaseModel
from google.cloud import vision
import io
from starlette.status import HTTP_302_FOUND,HTTP_303_SEE_OTHER
import os

class Entities(BaseModel):
    path: str

class EntitesOut(BaseModel):
    Prediction: dict

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "circular-cubist-339716-746fff6a0d6c.json"

app = FastAPI()

@app.post('/nudity', response_model=EntitesOut)
def prep_data(text:Entities):
    path = text.path
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
