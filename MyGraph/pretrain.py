import argparse
import datetime
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, \
    balanced_accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler
import pickle

from torch.utils.tensorboard import SummaryWriter

from models.graph_mae import GraphMAE

torch.set_printoptions(threshold=np.inf)

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


def get_args(args):
    parser = argparse.ArgumentParser(
        description="Train a model.",
        usage="training.py [<args>] [-h | --help]"
    )
    # ------------- Train ------------------------
    parser.add_argument("--lr", type=float, default=0.001, help="Path to pre-trained checkpoint.")
    parser.add_argument("--epochs", type=int, default=200, help="epoch for train")
    parser.add_argument("--device", type=str, default="0", help="device")
    parser.add_argument("--log_step", type=int, default=20, help="when to accelerator.print log")
    parser.add_argument("--log_dir", type=str, default="./output/logs/")
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--model", type=str, default="split")

    # ------------- Data ------------------------
    parser.add_argument("--graph", type=str, default="../data/DrugCombDB/processed/graph.pkl")

    parser.add_argument("--output", type=str, default="./output/mae/", help="output file fold")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")

    # ------------- Model ------------------------
    parser.add_argument("--num_layer", type=int, default=1, help="gnn layer num")

    parser.add_argument("--hid", type=int, default=768, help="hidden channels in model")

    parser.add_argument("--dropout", type=float, default=0.0, help="dropout weight")
    parser.add_argument("--num_drug", type=int, default=764)
    parser.add_argument("--num_protein", type=int, default=15970)
    parser.add_argument("--num_cell", type=int, default=76)
    parser.add_argument("--head", type=int, default=1)
    parser.add_argument("--alpha_l", type=int, default=2)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--dmask", action='store_true')

    args = parser.parse_args(args)

    if not os.path.exists(args.output):
        os.mkdir(args.output)
        print(str(args.output + "/logs"))
        os.mkdir(str(args.output + "/logs"))
    args.log_dir = str(args.output + "/logs")

    args.model_output = args.output + 'model.model'

    return args


def main(args=None):
    args = get_args(args)

    writer = SummaryWriter(args.log_dir)

    if torch.cuda.is_available():
        device_index = 'cuda:' + args.device
        device = torch.device(device_index)
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    for k, v in sorted(vars(args).items()):
        print(k, '=', v)

    # ----------- File Read ------------------------------------------------------
    with open(args.graph, 'rb') as f:
        graph = pickle.load(f)
        graph = graph.to(device)

    # ----------- Model Prepare ---------------------------------------------------
    modeling = GraphMAE
    # online_model = modeling(args).to(device)
    model = modeling(args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    # 学习率调整器，检测准确率的状态，然后衰减学习率
    scheduler = StepLR(step_size=20, gamma=0.1, optimizer=optimizer)

    print("model:", model)

    # ----------- Training ---------------------------------------------------

    epochs = args.epochs
    print('Training begin!')
    best_loss = torch.inf

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        loss = model(graph.collect("x"), graph.collect("edge_index"), args.mask_ratio)

        loss.backward()
        optimizer.step()

        print("[Train] {} Epoch[{}/{}]  loss={}".format(
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1, args.epochs, loss))

        writer.add_scalar("Loss/Training", loss, epoch)

        # save data
        if loss < best_loss:
            torch.save(model.state_dict(), args.model_output)

        scheduler.step()


if __name__ == '__main__':
    main()
