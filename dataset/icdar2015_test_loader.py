# dataloader add 3.0 scale
# dataloader add filer text
import numpy as np
from PIL import Image
from torch.utils import data
import util
import cv2
import random
import torchvision.transforms as transforms
import torch
import config

# ic15_root_dir = '/home/gem/phucph/PixelLink.pytorch/dataset/
ic15_root_dir = 'dataset/'
ic15_test_data_dir = ic15_root_dir + 'test_images/'
# ic15_test_gt_dir = ic15_root_dir + 'test_gt/'

random.seed(123456)

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
        
    except Exception as e:
        print(img_path)
        raise
    return img 

def scale(img, long_size=2480):
    h, w = img.shape[0:2]
    if max(h,w) > long_size:
        scale = long_size  * 1.0 / max(h, w)
        h,w=h*scale,w*scale
        
    sw=int(w/32)*32
    sh=int(h/32)*32
    img = cv2.resize(img, dsize=(sw,sh))
    return img

class IC15TestLoader(data.Dataset):
    def __init__(self, part_id=0, part_num=1, long_size=2240):
        data_dirs = [ic15_test_data_dir]
        
        self.img_paths = []
        
        for data_dir in data_dirs:
            img_names = util.io.ls(data_dir, '.jpg')
            img_names.extend(util.io.ls(data_dir, '.png'))

            img_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)
            
            self.img_paths.extend(img_paths)

        part_size = int(len(self.img_paths) / part_num)
        l = part_id * part_size
        r = (part_id + 1) * part_size
        self.img_paths = self.img_paths[l:r]
        self.long_size = config.test_long_size # config longsize test

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img= get_img(img_path)
        # print("size:",height,weight)
        scaled_img = scale(img, self.long_size)
        cv2.imwrite('outputs/' + img_path.split('/')[-1], scaled_img)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
        
        return img[:, :, [2, 1, 0]], scaled_img