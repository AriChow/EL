from torch.utils import data
import pandas as pd
import cv2
import os
from PIL import Image
import pickle
from EL import CONSTS
import torch
import numpy as np


class ChexpertDataset(data.Dataset):
    def __init__(self, csv_file=None, root_dir=None, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []
        for id in range(len(self.df)):
            item = self.df.iloc[id]
            self.img_paths.append(str(item['Path']))
            self.labels.append(int(item['Label']))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        label = int(item['Label'])
        img = Image.open(os.path.join(self.root_dir, str(item['Path'])))
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class OncologyDataset(data.Dataset):
    def __init__(self, data_dict, data_type='features', train=True):
        self.train = train  # training set or test set
        legend = {'CD3': 0, 'CD20': 1, 'CD68': 2, 'Claudin1': 3}
        all_keys = list(data_dict.keys())
        labels = []
        ids = []
        for i in range(len(data_dict['label'])):
            label = data_dict['label'][i]
            if label in legend.keys():
                labels.append(legend[label])
                ids.append('Position_' + str(data_dict['Position'][i]) + '_' + 'cell-id_' + str(data_dict['Cell-ID'][i]))
        labels = np.asarray(labels)
        if data_type == 'features':
            keys = all_keys[5:]
            normed_data = None
            for i in range(len(data_dict['label'])):
                data_i = []
                label = data_dict['label'][i]
                if label in legend.keys():
                    for k in keys:
                        data_i.append(data_dict[k][i])
                    data_i = np.expand_dims(np.asarray(data_i), 0)
                    if normed_data is None:
                        normed_data = data_i
                    else:
                        normed_data = np.vstack((normed_data, data_i))
            norm_val = np.amax(normed_data, 0)
            normed_data = np.divide(normed_data, norm_val)
            # Get the data

        elif data_type == 'images':
            imgs = data_dict['image']
            normed_data = None
            for i, img in enumerate(imgs):
                label = data_dict['label'][i]
                if label in legend.keys():
                    data_i = cv2.resize(img, (22, 22))
                    if normed_data is None:
                        normed_data = np.expand_dims(data_i, 0)
                    else:
                        normed_data = np.concatenate((normed_data, np.expand_dims(data_i, 0)))
            normed_data = normed_data / np.amax(normed_data[:])
            normed_data = torch.FloatTensor(normed_data).permute(0, 3, 1, 2)
        self.data_tensor = torch.FloatTensor(normed_data)
        self.labels = labels
        self.ids = ids

    def __getitem__(self, index):
        return self.data_tensor[index], self.labels[index]

    def __len__(self):
        return len(self.labels)