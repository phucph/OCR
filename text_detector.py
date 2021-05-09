import numpy as np
import torch
from PIL import Image
import os
import torchvision.transforms as transforms
import torch.nn as nn
import collections
from torch.utils import data
from torch.autograd import Variable
import util
from test_ic15 import to_bboxes
from dataset import IC15TestLoader
from dataset import get_img, scale
import cv2
import models
import torch.nn.functional as F
from pixel_link import decode_batch,mask_to_bboxes


model_path = "checkpoints/ic19_vgg16_bs_4_ep_100_pretrain_ic17/checkpoint_75.pth"


#load model: 
def init_pixel_link(model_path):
    model = models.vgg16(pretrained=True,num_classes=18)
    model = model.cuda(device)
    if os.path.isfile(model_path):
        print(("Loading model and optimizer from checkpoint '{}'".format(model_path)))
        checkpoint = torch.load(model_path)
            
        # model.load_state_dict(checkpoint['state_dict'])
        d = collections.OrderedDict()
        for key, value in list(checkpoint['state_dict'].items()):
            tmp = key[7:]
            d[tmp] = value
        model.load_state_dict(d)

        print(("Loaded checkpoint '{}' (epoch {})"
                .format(model_path, checkpoint['epoch'])))

    else:
        print(("No checkpoint found at '{}'".format(args.resume)))
        
    return model
# img_path = 'dataset/test_image/X00016469670.jpg'
def get_image(img_path):
    img = get_img(img_path)
    print(img)
    return img

def get_org_img(img):
    scaled_img = scale(img,long_size=1280)
    scaled_img = Image.fromarray(scaled_img)
    scaled_img = scaled_img.convert('RGB')
    scaled_img = transforms.ToTensor()(scaled_img)
    scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
    return org_img
# def to_bboxes(image_data, pixel_pos_scores, link_pos_scores):
#     link_pos_scores=np.transpose(link_pos_scores,(0,2,3,1))    
#     mask = decode_batch(pixel_pos_scores, link_pos_scores,0.6,0.9)[0, ...]
#     bboxes = mask_to_bboxes(mask, image_data.shape)
#     return mask,bboxes

data_loader = IC15TestLoader(long_size=1280)
test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=True)
image = list(test_loader)[0]

img = image[1] 
# print(img)
org_img = image[0]

device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def text_detector(img, org_img):
    
    model = init_pixel_link(model_path)
    print("loaded model")
    # model.eval()
    img = img.to(device)
    org_img = org_img.numpy().astype('uint8')[0]
    cls_logits,link_logits = model(img)
    outputs=torch.cat((cls_logits,link_logits),dim=1)
    shape=outputs.shape
    

    pixel_pos_scores=F.softmax(outputs[:,0:2,:,:],dim=1)[:,1,:,:]
    link_scores=outputs[:,2:,:,:].view(shape[0],2,8,shape[2],shape[3])
    link_pos_scores=F.softmax(link_scores,dim=1)[:,1,:,:,:]
    
    mask,bboxes=to_bboxes(org_img,pixel_pos_scores.cpu().detach().numpy(),link_pos_scores.cpu().detach().numpy())

    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
        lines.append(line)
    print("line",lines)
    return lines

text_detector(img,org_img)


