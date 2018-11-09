import sys

args = sys.argv
model = args[1]

import pandas as pd
import numpy as np

from os import listdir
from tqdm import tqdm

path = "./predictions/"

l = [c for c in listdir(path) if c.startswith(model) and c.endswith("feather")]
print(len(l), l[0])

dt = pd.DataFrame()
for file in tqdm(l):
    df = pd.read_feather(path+file)
    df.index = df["id"]
    df.drop("id", axis=1, inplace=True)
    #df = df.head(100)
    dt = dt.add(df, fill_value=0)
    #break
    
dt/=len(l)
dt.reset_index(inplace=True)

prediction_file = l[0].split("_")[0] + "_avg_" + str(len(l)) + '.feather'
dt.to_feather(prediction_file)