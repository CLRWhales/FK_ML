#this script looks at the states in the headr of files

from Multitask_loader import Multitask_loader
from torch.utils.data import DataLoader
import pandas as pd
import os
import torch
datapath = "D:\\DAS\\FK\\meta_test_hr_20250620T162109"
batch_size = 64
num_workers = 8
transform = None

dataset = Multitask_loader(datapath, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size,num_workers=num_workers)


if __name__ == '__main__':

    stats = []

    for _,_,reg in train_loader:
        stats.append(reg)

    all_stats = torch.cat(stats, dim=0)
    columns = ['Toff','Xoff','F','K','V']
    df = pd.DataFrame(all_stats.numpy(), columns =  columns)
    print(df.shape)
    pathname = os.path.join(datapath,'meta.csv')
    df.to_csv(pathname,index = False)