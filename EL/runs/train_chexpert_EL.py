from EL.models import resnet
import os
from EL import CONSTS
import torch.nn as nn
from torchvision import transforms
import torch
from sacred import Experiment
import argparse
import numpy as np
from EL.data.data import ChexpertDataset
from EL.models.models import SenderChexpertFull as SenderChexpert, ReceiverChexpert
from torch.utils.tensorboard import SummaryWriter
from EL.experiments import Trainer
from EL.utils.callbacks import TensorboardLogger, TemperatureUpdater, ConsoleLogger, EarlyStopper
from EL.utils.utils import load_model_weights

LOG_DIR_PATH = os.path.join(CONSTS.RESULTS_DIR, 'logs')
# PLOT_DIR = CONSTS.OUTPUT_DIR

ex = Experiment('EL')
# ex.observers.append(FileStorageObserver(LOG_DIR_PATH))

@ex.config
def config():
    batch_size = 128
    epochs = 2000
    log_interval = 10
    img_size_x = 224
    img_size_y = 224
    exp_name = 'chexpert_pleural_game_gs_new_1'
    gpu = 0
    pretrained = True
    train_batches_per_epoch = None
    val_batches_per_epoch = None
    max_len = 3
    embed_dim = 75
    lr = 1e-4
    hidden_size = 100
    vocab_size = 100
    temperature = 1

@ex.named_config
def gs():
    vocab_size = 100
    hidden_size = 50
    temperature = 1
    exp_name = 'chexpert_pleural_game_gs_new_1'

@ex.named_config
def rnn():
    embed_dim = 500
    vocab_size = 100
    hidden_size = 100
    temperature = 0.7
    max_len = 4
    exp_name = 'chexpert_pleural_game_rnn'



def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    loss = nn.CrossEntropyLoss()(receiver_output, _labels)
    acc = (torch.argmax(receiver_output, 1) == _labels).sum().item()
    return loss, {'acc': acc/len(_labels)}

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
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ]),
    }
    train_dataset = ChexpertDataset(os.path.join(CONSTS.DATA_DIR, 'CheXpert', 'train_pleural.csv'),
                                    root_dir=CONSTS.DATA_DIR, transform=data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = ChexpertDataset(os.path.join(CONSTS.DATA_DIR, 'CheXpert', 'val_pleural.csv'),
                                    root_dir=CONSTS.DATA_DIR, transform=data_transforms['val'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    exp_name = 'chexpert_pleural'
    model_path = os.path.join(CONSTS.RESULTS_DIR, 'models', exp_name, 'best_model.pth')
    # model = resnet.resnet50(pretrained=False)
    #
    # if args.pretrained:
    #     model = load_model_weights(model, model_path)

    sender = SenderChexpert(output_size=args.vocab_size)
    receiver = ReceiverChexpert(input_size=args.hidden_size)



    model_path = os.path.join(CONSTS.RESULTS_DIR, 'models', args.exp_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    output_dir = os.path.join(CONSTS.RESULTS_DIR, 'outputs', args.exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tensorboard_path = os.path.join(CONSTS.RESULTS_DIR, 'logs', 'tensorboard', args.exp_name)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    run_type = None
    if len(_run.meta_info['options']['UPDATE']) != 0:
        run_type = _run.meta_info['options']['UPDATE'][0]
    if run_type == 'rnn':
        from EL.models.multi_symbol_gs import RnnSenderGS, RnnReceiverGS, SenderReceiverRnnGS
        sender = RnnSenderGS(agent=sender, cell='lstm', max_len=args.max_len, embed_dim=args.embed_dim,
                              force_eos=True, vocab_size=args.vocab_size, hidden_size=args.hidden_size,
                              temperature=args.temperature, straight_through=False, trainable_temperature=False)
        receiver = RnnReceiverGS(agent=receiver, cell='lstm', embed_dim=args.embed_dim, vocab_size=args.vocab_size,
                                 hidden_size=args.hidden_size)
        game = SenderReceiverRnnGS(sender=sender, receiver=receiver, loss=loss, length_cost=0.5)
    else:
        from EL.models.single_symbol_gs import GumbelSoftmaxWrapper, SymbolReceiverWrapper, SymbolGameGS
        sender = GumbelSoftmaxWrapper(sender, temperature=args.temperature) # wrapping into a GS interface, requires GS temperature
        receiver = SymbolReceiverWrapper(receiver, args.vocab_size, agent_input_size=args.hidden_size)

        game = SymbolGameGS(sender, receiver, loss)
    optimizer = torch.optim.Adam(game.parameters(), lr=args.lr)

    # checkpoint = torch.load(os.path.join(model_path, 'best_model.pth'))
    # game.load_state_dict(checkpoint)


    writer = SummaryWriter(comment=args.exp_name)

    trainer = Trainer(
        game=game, optimizer=optimizer, train_data=train_loader, train_batches_per_epoch=args.train_batches_per_epoch,
        validation_data=val_loader, val_batches_per_epoch=args.val_batches_per_epoch, callbacks=[TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1),
                                                ConsoleLogger(), TensorboardLogger(writer=writer),
                                               EarlyStopper(save=os.path.join(model_path, 'best_model.pth'))]
    )

    trainer.train(n_epochs=100)
    # trainer.eval()