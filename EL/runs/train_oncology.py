from torch import optim
from torch.autograd import Variable
from torchvision import transforms, models
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from EL import CONSTS
import argparse
import numpy as np
import os
from EL.data.data import OncologyDataset
from EL.models.models import SenderOncoFeat, ReceiverOncoFeat
import torch.nn as nn
from aviation.utils.pytorchtools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle

LOG_DIR_PATH = os.path.join(CONSTS.RESULTS_DIR, 'logs')
# PLOT_DIR = CONSTS.OUTPUT_DIR

ex = Experiment('EL')
# ex.observers.append(FileStorageObserver(LOG_DIR_PATH))

@ex.config
def config():
    batch_size = 171
    epochs = 2000
    log_interval = 10
    img_size_x = 224
    img_size_y = 224
    exp_name = 'oncology_features_game'
    gpu = 0
    pretrained = False
    train_batches_per_epoch = None
    val_batches_per_epoch = None
    max_len = 3
    embed_dim = 200
    lr = 1e-3
    hidden_size = 100

class OncoModel(nn.Module):

    def __init__(self, sender, receiver):
        super(OncoModel, self).__init__()
        self.sender = sender
        self.receiver = receiver

    def forward(self, x):
        out = self.sender(x)
        out = self.receiver(out)
        return out


def loss_function(output, label):
    return nn.CrossEntropyLoss()(output, label)

def train_epoch(epoch, model, train_data_loader, device, optimizer, args):
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, datas in enumerate(train_data_loader):
        if datas is None:
            continue

        label = datas[1].to(device)
        data = Variable(datas[0])
        data = data.to(device)
        data = data.type(torch.cuda.FloatTensor)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out, label)
        loss.backward()
        _, pred = torch.max(out, 1)
        total += len(label)
        correct += (pred == label).sum().item()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_data_loader.dataset),
                       100. * batch_idx / len(train_data_loader),
                       loss.data / len(data), correct / total))
    train_loss = train_loss / len(train_data_loader.dataset)
    train_accuracy = correct / total
    return train_loss, train_accuracy

def val(model, val_data_loader, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    '''operations inside don't track history so don't allocate extra memory in GPU '''
    with torch.no_grad():
        for i, datas in enumerate(val_data_loader):
            if datas is None:
                continue
            label = datas[1].to(device)
            data = datas[0].to(device)
            data = data.type(torch.cuda.FloatTensor)
            out = model(data)
            val_loss += loss_function(out, label).data
            _, pred = torch.max(out, 1)
            total += len(label)
            correct += (pred == label).sum().item()
        val_loss /= len(val_data_loader.dataset)
        val_accuracy = correct / total

    return val_loss, val_accuracy


@ex.automain
def main(_run):

    # ===============
    # INTRO
    # ===============

    args = argparse.Namespace(**_run.config)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.gpu > -1:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = {
        'train': transforms.Compose([
            # transforms.Resize((22, 22)),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
    }


    ## DATASET
    with open(os.path.join(CONSTS.DATA_DIR, 'pathology', 'train.pkl'), 'rb') as file:
        train_data = pickle.load(file)

    with open(os.path.join(CONSTS.DATA_DIR, 'pathology', 'test.pkl'), 'rb') as file:
        val_data = pickle.load(file)

    train_dataset = OncologyDataset(data_dict=train_data, data_type='features')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = OncologyDataset(data_dict=val_data, data_type='features')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


    sender = SenderOncoFeat(hidden_size=args.hidden_size)
    receiver = ReceiverOncoFeat(input_size=args.hidden_size)



    model_path = os.path.join(CONSTS.RESULTS_DIR, 'models', args.exp_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    output_dir = os.path.join(CONSTS.RESULTS_DIR, 'outputs', args.exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tensorboard_path = os.path.join(CONSTS.RESULTS_DIR, 'logs', 'tensorboard', args.exp_name)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    model = OncoModel(sender, receiver)
    model.load_state_dict(torch.load(os.path.join(model_path, 'best_model.pth')))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    writer = SummaryWriter(tensorboard_path, comment=args.exp_name)

    patience = 20

    early_stopping = EarlyStopping(patience=patience, verbose=True, save=os.path.join(model_path, 'best_model.pth'))
    for epoch in range(1, args.epochs + 1):
        # train_epoch_loss, train_epoch_accuracy = train_epoch(epoch, model, train_loader, device, optimizer, args)
        # print('====> Epoch: {} Average training loss: {:.4f}'.format(
        #     epoch, train_epoch_loss))
        # print('====> Epoch: {} Average training accuracy: {:.4f} %'.format(
        #     epoch, train_epoch_accuracy*100))
        val_epoch_loss, val_epoch_accuracy = val(model, val_loader, device)
        print('====> val set loss: {:.4f}'.format(val_epoch_loss))
        print('====> val set accuracy: {:.4f} %'.format(val_epoch_accuracy*100))
    #     writer.add_scalar('Loss/train', train_epoch_loss, epoch)
    #     writer.add_scalar('Loss/val', val_epoch_loss, epoch)
    #     writer.add_scalar('Accuracy/train', train_epoch_accuracy, epoch)
    #     writer.add_scalar('Accuracy/val', val_epoch_accuracy, epoch)
    #
    #     scheduler.step(train_epoch_loss)
    #     early_stopping(val_epoch_loss, model)
    #
    # writer.close()