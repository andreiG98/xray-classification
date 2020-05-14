import argparse
import json
import os
import pandas as pd 
import numpy as np
from PIL import Image
import pandas as pd
import io
import cv2 as cv
import uuid
import base64

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from django.core.files.storage import default_storage

import xray_classification.settings as settings

class NetworkV1(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        self.base = base
        self.gradients = None

        if hasattr(base, 'fc'):
            in_features = self.base.fc.in_features
            self.base.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(in_features=in_features, out_features=512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(in_features=512, out_features=256),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(in_features=256, out_features=num_classes)
            )
        elif isinstance(base.classifier, nn.Linear): # densenet121
            in_features = self.base.classifier.in_features
            self.base.classifier = nn.Sequential(
                                        nn.Linear(in_features, num_classes),
                                        nn.Sigmoid())
        else: # mobilenetv2 / vgg19
            in_features = self.base.classifier[-1].in_features
            self.base.classifier[-1] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        layers = list(self.base.children())
        head = nn.Sequential(*layers[:-2])
        return head(x)

    def forward(self, x):
        if hasattr(self.base, 'fc'):
            layers = list(self.base.children())
            head = nn.Sequential(*layers[:-2])
            tail = nn.Sequential(layers[-2], nn.Flatten(), layers[-1])

            x = head(x)
            h = x.register_hook(self.activations_hook)
            fc = tail(x)
        else:
            fc = self.base(x)

        return fc

def construct_model(config, num_classes):
    if config['arch'] == 'resnext50':
        base = torchvision.models.resnext50_32x4d(pretrained=True)
    elif config['arch'] == 'resnet34':
        base = torchvision.models.resnet34(pretrained=True)
    elif config['arch'] == 'resnet50':
        base = torchvision.models.resnet50(pretrained=True)
    elif config['arch'] == 'mobilenetv2':
        base = torchvision.models.mobilenet_v2(pretrained=True)
    elif config['arch'] == 'vgg19':
        base = torchvision.models.vgg19(pretrained=True)
    elif config['arch'] == 'densenet121':
        base = torchvision.models.densenet121(pretrained=True)
    else:
        print("Invalid model name, exiting...")
        exit()

    if config['version'] == '1':
        model = NetworkV1(base, num_classes)
    elif config['version'] == '2':
        model = NetworkV2(base, num_classes)
    elif config['version'] == '3':
        model = NetworkV3(base, num_classes)
    elif config['version'] == '1CAM':
        model = NetworkV1Cam(base, num_classes)
    elif config['version'] == '3CAM':
        model = NetworkV3Cam(base, num_classes)

    return model

def transform_image(image_bytes, img_size):
    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]

    inference_transform = transforms.Compose([transforms.Resize(img_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean_nums, std_nums)])

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    return inference_transform(image).unsqueeze(0)

CATEGORY_NAMES = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_category_names(path):
    global CATEGORY_NAMES
    
    class_list_path = '/'.join(path.split('/')[:-2])
    class_names_json = os.path.join(class_list_path, 'class_names.json')
    
    with open(class_names_json) as json_file:
        CATEGORY_NAMES = json.load(json_file)
        
def load_model(path):
    print('Loading model for X-RAY classification....')
    get_category_names(path) # Load category names

    with open(os.path.join(path, 'config.json')) as json_file:
        config = json.load(json_file)
        
    best_path = os.path.join(path, 'best.pth')
    checkpoint = torch.load(best_path)
    model_state_dict = checkpoint['model']

    num_classes = len(CATEGORY_NAMES)
    model_cl = construct_model(config, num_classes)
    model_cl.load_state_dict(model_state_dict)
    model_cl.to(device)
    model_cl.eval()

    return model_cl, config['imgsize']

def compute_heatmap(model, output, tensor, image_bytes):
    _, pred = torch.max(output, 1)
    pred_label = pred.item()

    output[:, pred_label].backward()

    # get the activations of the last convolutional layer
    activations = model.get_activations(tensor).detach()

    # pull the gradients out of the model
    gradients = model.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    for i in range(pooled_gradients.shape[0]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze().cpu()

    # relu on top of the heatmap
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().numpy()

    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    img = open_cv_image[:, :, ::-1].copy() 

    heatmap = cv.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img

    return superimposed_img

def get_prediction(model, image_bytes, img_size):
    tensor = transform_image(image_bytes=image_bytes, img_size=img_size).to(device)

    print('forward model classification')
    output = model(tensor) # Pass the image to the model
    probabilities = nn.Softmax(dim=1)(output)
    scores, ids = torch.topk(probabilities, 5)
    scores, ids = scores.cpu(), ids.cpu()

    class_names = [CATEGORY_NAMES[str(label_id.item())] for label_id in ids[0]]
    scores = [str(round(float(score.item() * 100), 4)) for score in scores[0]]

    superimposed_img = compute_heatmap(model, output, tensor, image_bytes)

    heatmap_path = os.path.join(settings.MEDIA_URL, 'heatmaps')
    filename = str(uuid.uuid4().hex) + '.jpg'
    heatmap_path = os.path.join(heatmap_path, filename)
    cv.imwrite(heatmap_path, superimposed_img)

    classification_prediction = {}
    classification_prediction['class_names'] = class_names
    classification_prediction['scores'] = scores
    classification_prediction['heatmap_url'] = heatmap_path

    return classification_prediction