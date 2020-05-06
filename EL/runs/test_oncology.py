from EL.models import resnet
import os
from EL import CONSTS
import torch.nn as nn
from torchvision import transforms
import torch
from sacred import Experiment
import argparse
import numpy as np
from EL.data.data import OncologyDataset
import pickle
from EL.models.models import SenderOncoFeat, ReceiverOncoFeat
from EL.utils.utils import dump_sender_receiver
from EL.experiments import Trainer
import pandas as pd
from captum.attr import NeuronConductance
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

LOG_DIR_PATH = os.path.join(CONSTS.RESULTS_DIR, 'logs')
# PLOT_DIR = CONSTS.OUTPUT_DIR

ex = Experiment('EL')
# ex.observers.append(FileStorageObserver(LOG_DIR_PATH))

@ex.config
def config():
    batch_size = 32
    epochs = 2000
    log_interval = 10
    img_size_x = 224
    img_size_y = 224
    exp_name = 'oncology_features_game_gs_new'
    gpu = 0
    pretrained = False
    train_batches_per_epoch = None
    val_batches_per_epoch = None
    max_len = 3
    embed_dim = 200
    lr = 1e-3
    hidden_size = 100
    vocab_size = 100
    temperature = 5
    latent_dim = 50

@ex.named_config
def gs():
    vocab_size = 100
    hidden_size = 50
    temperature = 1
    exp_name = 'chexpert_pleural_game_gs'

@ex.named_config
def rnn():
    embed_dim = 75
    vocab_size = 100
    hidden_size = 100
    temperature = 0.7
    max_len = 4
    exp_name = 'chexpert_pleural_game_rnn'

def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features", exp_name='', fig_name='neural-conductance.png'):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(12, 10))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, rotation=60, fontsize=8)
        plt.xlabel(axis_title)
        plt.title(title)
        plt.savefig(os.path.join(CONSTS.RESULTS_DIR, 'outputs', exp_name, fig_name))



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
    with open(os.path.join(CONSTS.DATA_DIR, 'pathology', 'test.pkl'), 'rb') as file:
        test_data = pickle.load(file)

    test_dataset = OncologyDataset(data_dict=test_data, data_type='features')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


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
        game = SenderReceiverRnnGS(sender=sender, receiver=receiver, loss=loss, length_cost=0.0)
    else:
        from EL.models.single_symbol_gs import GumbelSoftmaxWrapper, SymbolReceiverWrapper, SymbolGameGS
        sender = GumbelSoftmaxWrapper(sender, temperature=args.temperature) # wrapping into a GS interface, requires GS temperature
        receiver = SymbolReceiverWrapper(receiver, args.vocab_size, agent_input_size=args.hidden_size)

        game = SymbolGameGS(sender, receiver, loss)


    checkpoint = torch.load(os.path.join(model_path, 'best_model.pth'))
    game.load_state_dict(checkpoint)
    game.to(device)
    optimizer = torch.optim.Adam(game.parameters(), lr=1e-3)

    trainer = Trainer(
        game=game, optimizer=optimizer, train_data=test_loader, train_batches_per_epoch=None,
        validation_data=test_loader, val_batches_per_epoch=None)
    # trainer.eval()

    sender_inputs, messages, receiver_inputs, receiver_outputs, labels = \
        dump_sender_receiver(game=game, dataset=test_loader, gs=True, variable_length=False, device=device)


    msgs = []
    for m in messages:
        msgs.append(int(m.cpu()))

    predictions = []
    for pred in receiver_outputs:
        predictions.append(int(torch.argmax(pred).cpu()))

    lbls = []
    for l in labels:
        lbls.append(int(l.cpu()))
    f = f1_score(lbls, predictions, average='weighted')

    df = pd.DataFrame({'ID': test_dataset.ids, 'Ground Truth': lbls, 'Predictions': predictions, 'Message': msgs})
    # df.to_csv(os.path.join(CONSTS.RESULTS_DIR, 'EL_oncology_single_symbol_new.csv'), index=False)


    ## Interpretation
    feature_names = list(test_data.keys())[5:]
    fnames = []
    for f in feature_names:
        if 'Mean' in f and 'CD20' in f:
            fnames.append('Mean_CD20')
        elif 'Mean' in f and 'CD3' in f:
            fnames.append('Mean_CD3')
        elif 'Mean' in f and 'CD68' in f:
            fnames.append('Mean_CD68')
        elif 'Mean' in f and 'C' in f:
            fnames.append('Mean_Claudin1')
        elif 'Std' in f and 'CD20' in f:
            fnames.append('Std_CD20')
        elif 'Std' in f and 'CD3' in f:
            fnames.append('Std_CD3')
        elif 'Std' in f and 'CD68' in f:
            fnames.append('Std_CD68')
        elif 'Std' in f and 'C' in f:
            fnames.append('Std_Claudin1')
        elif 'Quant01' in f and 'CD20' in f:
            fnames.append('Quant01_CD20')
        elif 'Quant01' in f and 'CD3' in f:
            fnames.append('Quant01_CD3')
        elif 'Quant01' in f and 'CD68' in f:
            fnames.append('Quant01_CD68')
        elif 'Quant01' in f and 'C' in f:
            fnames.append('Quant01_Claudin1')
        elif 'Quant25' in f and 'CD20' in f:
            fnames.append('Quant25_CD20')
        elif 'Quant25' in f and 'CD3' in f:
            fnames.append('Quant25_CD3')
        elif 'Quant25' in f and 'CD68' in f:
            fnames.append('Quant25_CD68')
        elif 'Quant25' in f and 'C' in f:
            fnames.append('Quant25_Claudin1')
        elif 'Quant50' in f and 'CD20' in f:
            fnames.append('Quant50_CD20')
        elif 'Quant50' in f and 'CD3' in f:
            fnames.append('Quant50_CD3')
        elif 'Quant50' in f and 'CD68' in f:
            fnames.append('Quant50_CD68')
        elif 'Quant50' in f and 'C' in f:
            fnames.append('Quant50_Claudin1')
        elif 'Quant75' in f and 'CD20' in f:
            fnames.append('Quant75_CD20')
        elif 'Quant75' in f and 'CD3' in f:
            fnames.append('Quant75_CD3')
        elif 'Quant75' in f and 'CD68' in f:
            fnames.append('Quant75_CD68')
        elif 'Quant75' in f and 'C' in f:
            fnames.append('Quant75_Claudin1')
        elif 'Quant99' in f and 'CD20' in f:
            fnames.append('Quant99_CD20')
        elif 'Quant99' in f and 'CD3' in f:
            fnames.append('Quant99_CD3')
        elif 'Quant99' in f and 'CD68' in f:
            fnames.append('Quant99_CD68')
        elif 'Quant99' in f and 'C' in f:
            fnames.append('Quant99_Claudin1')

    cond = NeuronConductance(game, game.sender.gs_layer)

    cond_vals = {47: None, 58: None, 4: None, 20: None}
    for i in range(len(lbls)):
        cond_val = cond.attribute(sender_inputs[i].unsqueeze(0).to(device), neuron_index=msgs[i], target=lbls[i])
        if cond_vals[msgs[i]] is None:
            cond_vals[msgs[i]] = cond_val.cpu().numpy()
        else:
            cond_vals[msgs[i]] = np.vstack((cond_vals[msgs[i]], cond_val.cpu().numpy()))
    visualize_importances(fnames, cond_vals[47].mean(0), title="Average Feature Importances for Neuron 47 for CD68",
                          exp_name=args.exp_name, fig_name='CD68_symbol_47.png')
    visualize_importances(fnames, cond_vals[20].mean(0), title="Average Feature Importances for Neuron 20 for Claudin1",
                          exp_name=args.exp_name, fig_name='Claudin1_symbol_20.png')
    visualize_importances(fnames, cond_vals[58].mean(0), title="Average Feature Importances for Neuron 58 for CD3",
                          exp_name=args.exp_name, fig_name='CD3_symbol_58.png')
    visualize_importances(fnames, cond_vals[4].mean(0), title="Average Feature Importances for Neuron 4 for CD20",
                          exp_name=args.exp_name, fig_name='CD20_symbol_4.png')
