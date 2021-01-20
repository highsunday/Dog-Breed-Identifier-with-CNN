#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 17:08:15 2021

@author: highsunday
"""

from cnn_finetune import make_model
import torch.nn as nn
import torch
from torch.autograd import Variable
from torchvision import transforms

IMAGE_SIZE=(299,299)

def openModel():
    def make_classifier(in_features, num_classes):
        return nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    model = make_model('xception', num_classes=120, pretrained=True, input_size=IMAGE_SIZE, classifier_factory=make_classifier)
    model.load_state_dict(torch.load('model/run_2021_01_07-15_10_01_weights_65.pkl'))
    model.eval()
    model=model.cuda()
    return model

def returnTopN_Predict(images,model,all_breed_for_predict,n=5):
    preprocess = transforms.Compose([transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])])
    
    img_tensor = preprocess(images).float()
    img_tensor = img_tensor.unsqueeze_(0)
    img_tensor=img_tensor.cuda()
    fc_out = model(Variable(img_tensor))
    output=torch.softmax(fc_out,axis=1)
    output = output.detach().cpu().numpy()
    print(output)
    
    temp=sorted(range(len(output[0])), key=lambda i: output[0][i])[-n:]
    temp.reverse()
    top_n_result=[]
    count=0
    for i in (temp):
        if(output[0][i]*100<5):
            break
        temp2=(all_breed_for_predict[i],output[0][i]*100)
        count+=output[0][i]*100
        top_n_result.append(temp2)
    top_n_result.append(('Others',100-count))
    return top_n_result