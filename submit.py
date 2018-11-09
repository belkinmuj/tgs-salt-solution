import sys

args = sys.argv
file = args[1]
threshhold = float(args[2])

import pandas as pd
import numpy as np

from os import listdir
from tqdm import tqdm

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

img_size_ori = 101

dt = pd.read_feather(file)
dt.index = dt["id"]
dt.drop("id",inplace=True, axis=1)

pred_dict = {idx: rle_encode(np.round(pred > threshhold)) for idx, pred in zip(dt.index.values, 
                                                                        dt.values.reshape((-1,img_size_ori,img_size_ori)))}

sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv("test_avg_"+file.split(".")[0]+"_"+str(threshhold)+".csv")