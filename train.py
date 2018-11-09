import sys
import os
import cv2
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

args = sys.argv
gpu = args[1]
model_name = args[2]
init_epoch = int(args[3])
folds = list(map(int,args[4:]))

print("\n Using GPU:", gpu)
print(" Model:", model_name)
print(" Init epoch:", str(init_epoch))
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
log_save_path = "./logs/"
ensure_dir(log_save_path)

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
        img = cv2.copyMakeBorder(img, pad, pad+1, pad, pad+1, cv2.BORDER_REFLECT_101)
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

depths_df = pd.read_csv(os.path.join(directory, 'train.csv'))
train_fold = pd.read_csv(os.path.join(directory, 'folds_stratified_by_coverage.csv'))
#train_fold = pd.read_csv(os.path.join(directory, 'folds.csv'))

train_path = os.path.join(directory, 'train')
file_list = list(depths_df['id'].values)

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def batch_iou(pred_masks: torch.Tensor, true_masks: torch.Tensor, th: float = 0.5):
    # masks shape is (batch_size, 1, height, width)
    pred_masks = (pred_masks > th).float()
    intersection = (pred_masks * true_masks).sum(dim=(1, 2, 3))
    union = pred_masks.sum(dim=(1, 2, 3)) + true_masks.sum(dim=(1, 2, 3))
    iou = intersection / (union - intersection)
    iou[iou != iou] = 1  # trick to set all NaN's to 1 (if we have None, we have correctly predicted empty mask)
    return float(iou.mean())


for i_fold in folds:
    print(" Training fold:", i_fold)

    file_list_val = train_fold.id[train_fold.fold == i_fold].values
    assert(len(file_list_val)!=0)
    file_list_train = train_fold.id[train_fold.fold != i_fold].values
    print(" Train val split:", len(file_list_train), len(file_list_val))

    dataset = TGSSaltDataset(train_path, file_list_train, is_static = False)
    dataset_val = TGSSaltDataset(train_path, file_list_val)
    
    model_paths = sorted([model_save_path+f for f in listdir(model_save_path) if f.startswith(model_name+"_fold_"+str(i_fold))]) 
    if model_paths:
        print(" Load model:", model_paths[-1])
        model = get_model(model_path=model_paths[-1])
    else:
        print(" Load empty model")
        model = get_model()

    best_metric, best_threshold = 0,0
    best_model = None

    epoch = 250
    epoch_ch_loss = 100 # change loss_fn
    epoch_ch_sch = 150 #150 # change scheduler 
    n_epoch_save = 3 # amount of models to save

    max_learning_rate = 1e-4
    med_learning_rate = 1e-5
    min_learning_rate = 1e-6
    step_size = 5 #half cycle in epochs
    epoch_eval = 50 # amount of epochs to train with bn turned off
    epoch = epoch + epoch_eval

    max_lr = med_learning_rate
    base_lr = min_learning_rate

    if init_epoch > epoch_ch_loss:
        loss_fn = L.lovasz_hinge
        optimizer = torch.optim.Adam(model.parameters(), lr=med_learning_rate)
    else:
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=max_learning_rate)
    
    if init_epoch > epoch_ch_sch:
        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                             step_size=step_size, mode='exp_range', gamma=0.99999, scale_mode='cycle')
    else:
        scheduler = ROP(optimizer, mode='min', factor=0.5, patience=10, 
                        verbose=True, threshold=0.0001, threshold_mode='rel', 
                        cooldown=0, min_lr=min_learning_rate, eps=1e-08)
    
    scores = deque()
    paths = deque()
    cur_min_score = 0
    for e in range(init_epoch, epoch):
        train_loss = []
        if e >= (epoch - epoch_eval):
            model.eval()
        else:
            model.train()
        
        for image, mask in tqdm(data.DataLoader(dataset, batch_size = batch_size, shuffle = True)):
            y_pred = model(Variable(image.cuda()))

            if e <= epoch_ch_loss:
                y_pred = F.sigmoid(y_pred)
            
            mask = Variable(mask.cuda())
            loss = loss_fn(y_pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.data[0])
            del image, mask, loss, y_pred

        val_loss = []
        IOUs = list()
        model.eval()

        for image, mask in data.DataLoader(dataset_val, batch_size = batch_size, shuffle = False):
            y_pred = model(Variable(image.cuda(), volatile=True))
            if e <= epoch_ch_loss:
                y_pred = F.sigmoid(y_pred)
            loss = loss_fn(y_pred, Variable(mask.cuda(), volatile=True))
            if e > epoch_ch_loss:
                y_pred = F.sigmoid(y_pred)
            IOUs.append(batch_iou(y_pred.cpu(),mask))
            val_loss.append(loss.data[0])
            del image, mask, loss, y_pred

        train_data = "Epoch: %d, Train: %.3f, Val: %.3f, Mean_IOU:%.5f, LR:%.7f \n" \
                %(e, np.mean(train_loss), np.mean(val_loss), np.mean(IOUs), optimizer.state_dict()["param_groups"][0]["lr"])
        print(train_data)
        cur_iou = np.mean(IOUs)

        try:
            with open("{}{}_fold_{}.txt".format(log_save_path,model_name,i_fold), "a") as logfile:
                logfile.write(train_data)
        except:
            os.system(">{}{}_fold_{}.txt".format(log_save_path,model_name,i_fold))

        if e <= epoch_ch_sch:
            scheduler.step(np.mean(val_loss))
        else:
            scheduler.batch_step()

        if cur_iou > cur_min_score:
            scores.append(cur_iou)
            cur_modelpath = "{}{}_fold_{}_{:.5f}.pth".format(model_save_path, model_name, i_fold, cur_iou)
            cur_min_score = min(scores)
            
            save_checkpoint(cur_modelpath, model, optimizer)
            
            paths.append(cur_modelpath)
            if len(paths)>n_epoch_save:
                del_path = paths.popleft()
                if os.path.exists(del_path):
                  os.remove(del_path)
                else:
                  print("The file does not exist "+del_path)
                scores.popleft()

        sys.stdout.flush()
        if e == epoch_ch_loss:
            print("Lovasz loss, lr:", med_learning_rate)
            loss_fn = L.lovasz_hinge
            optimizer = torch.optim.Adam(model.parameters(), lr=med_learning_rate)
            scheduler = ROP(optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=min_learning_rate, eps=1e-08)
            
        if e == epoch_ch_sch:
            print("CyclicLR:", base_lr, max_lr)
            scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                 step_size=step_size, mode='exp_range', gamma=0.99999, scale_mode='cycle')

        if e == (epoch - epoch_eval):
            model_paths = sorted([model_save_path+f for f in listdir(model_save_path) if f.startswith(model_name+"_fold_"+str(i_fold))]) 
            if model_paths:
                print(" Load model:", model_paths[-1])
                model = get_model(model_path=model_paths[-1])
            else:
                print(" Load empty model")
                model = get_model()
sys.exit(1)


