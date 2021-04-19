import os
import cv2
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class KaggleDataset(Dataset):

    def __init__(self, 
                is_train,
                data_root='/home/dahu/xueruini/onion_rain/pytorch/dataset/plant-pathology-2021-fgvc8/train_images/', 
                csv_root='/home/dahu/xueruini/onion_rain/pytorch/dataset/plant-pathology-2021-fgvc8/div/',
                ):
        self.data_root = data_root
        if is_train:
            csv_data = pd.read_csv(csv_root+"trainset.csv")
            self.transform = A.Compose([
                A.SmallestMaxSize(256),
                A.RandomCrop(224, 224),
                A.Flip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        
        else:
            csv_data = pd.read_csv(csv_root+"valset.csv")
            self.transform = A.Compose([
                A.SmallestMaxSize(256),
                A.CenterCrop(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        self.img_paths = csv_data['image'].values
        self.labels = csv_data['t_label'].values
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.data_root, self.img_paths[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']

        labels = self.labels[index]
        labels = labels[1:-1].split(" ")
        label = np.array([np.float32(x) for x in labels])
        
        return image, label


def denormalization(x):
    ''' de-normalization '''
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x    


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     data_dir = '../../data/train_images/'
#     csv_file = '../dataset_csv/all_train.csv'
#     trainset = KaggleDataSet(data_root=data_dir, csv_file=csv_file, mode='train')
#     trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True)

#     for _, data in enumerate(trainloader):
#         imgs, labels = data
#         imgs = imgs.numpy()
#         # labels = labels.numpy()
#         # print(imgs, imgs.shape)
#         print(labels)

#         figure, ax = plt.subplots(nrows=1, ncols=4, figsize=(24, 10))

#         for i in range(imgs.shape[0]):
#             img = denormalization(imgs[i])

#             ax[i].imshow(img)
        
#         plt.show()
#         plt.savefig("test_dataloader.png")

#         break