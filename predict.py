import sys
import os
import cv2
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

args = sys.argv
gpu = args[1]
model_name = args[2]
folds = list(map(int,args[3:]))

print("\n Model:", model_name)
print(" Using GPU:", gpu)
print(" Folds:", " ".join(map(str,folds)),"\n")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=gpu

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

import torch
import lovasz_losses as L
from collections import deque
from os import listdir
from models.ternausnet2 import TernausNetV2
from models.resnext50_unet import ResnextUnet
from models.resnet152_unet import Resnet152unet
from models.densenet_unet import DenseNetUnet
from torch.nn import functional as F
from torch.utils import data
from pathlib import Path
from torch.nn import functional as F
from torch.autograd import Variable
from tqdm import tqdm, tqdm_notebook

from torch.optim.lr_scheduler import ReduceLROnPlateau as ROP
from CyclicLR import CyclicLR

import albumentations as alb

size = {
        "resnext50": 256,
        "resnext50_128": 128,
        "resnet38": 224,
        "resnet38_128": 128,
        "resnet152": 128,
        "resnet152_256": 256,
        "densenet_128": 128,
        }

batch = {
         "resnext50": 14,
         "resnext50_128": 14,
         "resnet38": 20,
         "resnet38_128": 30,
         "resnet152": 32,
         "resnet152_256": 14,
         "densenet_128": 16,
        }

arch = {
         "resnext50": ResnextUnet(),
         "resnext50_128": ResnextUnet(),
         "resnet38": TernausNetV2(num_classes=1, pretrained=True),
         "resnet38_128": TernausNetV2(num_classes=1, pretrained=True),
         "densenet_128": DenseNetUnet(),
         "resnet152": Resnet152unet(),
         "resnet152_256": Resnet152unet(),
        }

directory = '../input'
model_save_path = "./checkpoints/"
ensure_dir(model_save_path)
predict_save_path = "./predictions/"
ensure_dir(predict_save_path)

img_size_ori = 101
img_size_target = size[model_name]
batch_size = batch[model_name]

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    
    k = 1 + int(img_size_target>img_size_ori*2)
    pad = (img_size_target-k*img_size_ori)//2
    if k==2:
        img = cv2.resize(img, (img_size_ori*2, img_size_ori*2), interpolation = cv2.INTER_NEAREST )
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    else:
        img = cv2.copyMakeBorder(img, pad, pad+1, pad, pad+1, cv2.BORDER_REFLECT_101) #top, bottom, left, right
    return img
    
def downsample(img):
    if img_size_ori == img_size_target:
        return img

    k = 1 + int(img_size_target>img_size_ori*2)   
    pad = (img_size_target-k*img_size_ori)//2
    
    if k==2:
        img = img[pad:-pad, pad:-pad]
        img = cv2.resize(img, (img_size_ori, img_size_ori), interpolation = cv2.INTER_NEAREST )
    else:
        img = img[pad:-pad-1, pad:-pad-1]
    return img

def get_model(model_path=None):
    model = arch[model_name]
    if model_path != None:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
    model.train()
    if torch.cuda.is_available():
        model.cuda()
    return model

def strong_aug(p=0.5):
    k = 0.2
    return alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.ShiftScaleRotate(shift_limit=0.7-k, scale_limit=0.6-k, rotate_limit=0, p=0.9),
        alb.RandomBrightness(limit=0.8-k, p=0.5),
        alb.RandomContrast(limit=0.9-k, p=0.7),
    ], p=p)

augmentation = strong_aug(p=1)

def post_process(img, mask = False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = upsample(img)
    
    if mask:
        # Convert mask to 0 and 1 format
        img = img[:, :, 0:1] // 255
        return torch.from_numpy(np.transpose(img, (2, 0, 1)).astype('float32'))
    else:
        img = img / 255.0
        return torch.from_numpy(np.transpose(img, (2, 0, 1)).astype('float32'))

def load_image(path):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)
    
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = cv2.imread(str(path))

    return img


class TGSSaltDataset(data.Dataset):
    def __init__(self, root_path, file_list, is_test = False, is_static = True):
        self.is_test = is_test
        self.root_path = root_path
        self.file_list = file_list
        self.is_static = is_static
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        
        file_id = self.file_list[index]
        
        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")
        
        mask_folder = os.path.join(self.root_path, "masks")
        mask_path = os.path.join(mask_folder, file_id + ".png")
        
        image = load_image(image_path)
        
        if self.is_test:
            image = post_process(image)
            return (image,)
        else:
            mask = load_image(mask_path)
            if not self.is_static:
                augmented = augmentation(**{"image":image, "mask":mask})
                image, mask = augmented["image"], augmented["mask"]
            

            image = post_process(image)
            mask = post_process(mask, mask = True)
            
            return image, mask

test_path = os.path.join(directory, 'test')
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = sorted([f.split('/')[-1].split('.')[0] for f in test_file_list])

test_dataset = TGSSaltDataset(test_path, test_file_list, is_test = True)

for i_fold in folds:
    print(" Predicting fold:", i_fold)

    model_paths = sorted([model_save_path+f for f in listdir(model_save_path) if f.startswith(model_name+"_fold_"+str(i_fold))])
    model_paths = model_paths[::-1][:min(3,len(model_paths))]
    for model_path in model_paths:
        print(" Load model:", model_path)
        model = get_model(model_path=model_path)

        predictions = []
        for image in tqdm(data.DataLoader(test_dataset, batch_size = batch_size)):
            image = image[0].type(torch.FloatTensor)
            image_hor = torch.tensor(image.data.numpy()[:,:,:,::-1].copy())

            image = image.cuda()
            image_hor = image_hor.cuda()
            
            y_pred = model(Variable(image))
            y_pred = F.sigmoid(y_pred).cpu().data.numpy()
            
            y_pred_hor = model(Variable(image_hor))
            y_pred_hor = F.sigmoid(y_pred_hor).cpu().data.numpy()[:,:,:,::-1]
            
            y_pred += y_pred_hor
            y_pred /= 2

            for j,_ in enumerate(y_pred):
                predictions.append(downsample(y_pred[j][0]))

        predictions = np.array(predictions).reshape(-1, img_size_ori*img_size_ori)
        df = pd.DataFrame(predictions, columns=[str(i) for i in range(img_size_ori*img_size_ori)])
        df["id"] = test_file_list
        prediction_file = predict_save_path + model_path.split("/")[-1][:-3] + 'feather'
        df.to_feather(prediction_file)