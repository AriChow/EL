import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from EL import CONSTS
import shutil
import os


data_dir = os.path.join(CONSTS.DATA_DIR, 'symbolic', 'CheXpert')

def chexpert():
    data = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    ## Findings annotations
    labels = data['Pleural Effusion']
    y = []
    files = []
    for i in range(len(labels)):
        label = data['Pleural Effusion'][i]
        if np.isnan(label) or label == -1:
            continue
        else:
            y.append(int(label))
            files.append(data['Path'][i])

    y = np.asarray(y)
    files = np.asarray(files, dtype='S500')
    X = np.ones((len(y), 1))
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(X, y)
    train_index = None
    val_index = None
    for train_index, val_index in skf.split(X, y):
        break

    train_x = files[train_index]
    train_y = y[train_index]
    val_x = files[val_index]
    val_y = y[val_index]
    df1 = pd.DataFrame({'Path': train_x, 'Label': train_y})
    df2 = pd.DataFrame({'Path': val_x, 'Label': val_y})
    # df1 = data.loc[val_index]
    # df2 = data.loc[train_index]

    df1.to_csv(os.path.join(data_dir, 'train_pleural.csv'))
    df2.to_csv(os.path.join(data_dir, 'val_pleural.csv'))

chexpert()
