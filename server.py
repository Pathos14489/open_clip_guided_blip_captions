from nis import match
from IPython.display import Image 
from PIL import Image

import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from flask import Flask, request, jsonify

import torch
import numpy as np
from PIL import Image
import skimage.io as io
from models.blip import blip_decoder
import easyocr


from open_clip import tokenizer
import open_clip

import json

import requests


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    print(config)
    model = instantiate_from_config(config.model)

    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

api = Flask(__name__)

@api.route('/prompt', methods=['POST'])
def main():
    opt = request.form.to_dict()

    if opt is None:
        return jsonify({"error": "no json data"})
    if "type" not in opt:
        opt["type"] = 1
    if "max_length" not in opt:
        opt["max_length"] = 100
    if "min_length" not in opt:
        opt["min_length"] = 5
    if "top_p" not in opt:
        opt["top_p"] = 0.85
    if "num_beams" not in opt:
        opt["num_beams"] = 3
    if "batch_size" not in opt:
        opt["batch_size"] = 200
    if "pony_fix" not in opt:
        opt["pony_fix"] = False
    
    opt["type"] = int(opt["type"])
    opt["max_length"] = int(opt["max_length"])
    opt["min_length"] = int(opt["min_length"])
    opt["top_p"] = float(opt["top_p"])
    opt["num_beams"] = int(opt["num_beams"])
    opt["batch_size"] = int(opt["batch_size"])
    opt["pony_fix"] = bool(opt["pony_fix"])

        
    print(opt)
    print(type(opt))

    image = request.files['image'] 
    print(image)
    
    raw_image = Image.open(image).convert('RGB')
    clip_image = preprocess(raw_image).unsqueeze(0)
    imagePath = './images/' + image.filename
    raw_image.save(imagePath)   
    ocrOut = reader.readtext(imagePath, detail = 0)

    w,h = raw_image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)
    
    output = []

    # nucleus sampling
    if opt["type"] == 1:
        for i in range(opt["batch_size"]):
            print(f"{i}/{opt['batch_size']}")
            caption = model.generate(image, sample=True, top_p=opt["top_p"], max_length=opt["max_length"], min_length=opt["min_length"])[0]
            output.append(caption)
    # beam search
    else:
        for i in range(opt["batch_size"]):
            print(f"{i}/{opt['batch_size']}")
            caption = model.generate(image, sample=False, num_beams=opt["num_beams"], max_length=opt["max_length"], min_length=opt["min_length"])[0]
            output.append(caption)


    if opt["pony_fix"] == True:
        alt = [item.replace("pinkies", "ponies") for item in output]
        alt = [item.replace("pinkie", "pony") for item in alt]
        # replace pinkies and pinkie in the output list with ponies and pony
        output = alt


    texts = json.dumps(output)
    # print(texts)
    # texts = texts.replace("[", "")
    # texts = texts.replace("]", "")
    # texts = "[" + texts + "]"
    print(texts)
    texts = json.loads(texts)
    text = tokenizer.tokenize(texts)

    image_features = clip.encode_image(clip_image)
    text_features = clip.encode_text(text)

    text_probs = image_features @ text_features.T

    probDict = {}
    probList = []
    for i in range(len(texts)):
        probDict[texts[i]] = text_probs[0][i].item()
        probList.append({
            "text": texts[i],
            "prob": text_probs[0][i].item()
        })

    probList = sorted(probList, key=lambda x: x["prob"], reverse=True)
    
    clip_out = {"probDict": probDict, "probList": probList}

    return jsonify({
        "prediction": probList[0],
        "ocr": ocrOut,
        "settings": opt,
        "predictions": probList,
    })

@api.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = '*'
    return response

def load_demo_image(image_size,device):
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   

    w,h = raw_image.size
    display(raw_image.resize((w//5,h//5)))

    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image


if __name__ == "__main__":

    clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

    reader = easyocr.Reader(['en'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda')
    image_size = 384

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
        
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    print("Blip loaded, server running!")
    with torch.no_grad():
        api.run(host='0.0.0.0', port=5015)
