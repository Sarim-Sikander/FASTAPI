from PIL import Image
import torch
import cv2
import pickle

transform = pickle.load(open('transform.pkl','rb'))
model = pickle.load(open('model.pkl', 'rb'))
model.eval()

def inference(raw_image):
    img = cv2.imread(raw_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    image = transform(im_pil).unsqueeze(0)     #.to(device)   
    with torch.no_grad():
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        return 'caption: '+caption[0]

result = inference('starry.jpg')
print(result)


# model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth'
# model_url = 'model__base_caption.pth'
    
# model = blip_decoder(pretrained=model_url, image_size=384, vit='base')