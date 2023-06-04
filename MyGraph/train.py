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

from models.emb_model import EmbModel
from models.emb_split import Emb_Split_Model

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
    parser.add_argument("--train_batch_size", type=int, default=512, help="train batch size")
    parser.add_argument("--valid_batch_size", type=int, default=512)
    parser.add_argument("--test_batch_size", type=int, default=512, help="test batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Path to pre-trained checkpoint.")
    parser.add_argument("--epochs", type=int, default=200, help="epoch for train")
    parser.add_argument("--device", type=str, default="0", help="device")
    parser.add_argument("--log_step", type=int, default=20, help="when to accelerator.print log")
    parser.add_argument("--log_dir", type=str, default="./output/logs/")
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--model", type=str, default="emb")

    # ------------- Data ------------------------
    parser.add_argument("--data", type=str, default="../data/DrugCombDB/processed/dataset.pkl")
    parser.add_argument("--graph", type=str, default="../data/DrugCombDB/processed/graph.pkl")

    parser.add_argument("--output", type=str, default="./output/", help="output file fold")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")

    # ------------- Model ------------------------
    parser.add_argument("--num_layer", type=int, default=1, help="gnn layer num")

    parser.add_argument("--hid", type=int, default=768, help="hidden channels in model")

    parser.add_argument("--dropout", type=float, default=0.0, help="dropout weight")
    parser.add_argument("--num_drug", type=int, default=764)
    parser.add_argument("--num_protein", type=int, default=15970)
    parser.add_argument("--num_cell", type=int, default=76)
    parser.add_argument("--head", type=int, default=1)

    args = parser.parse_args(args)

    if not os.path.exists(args.output):
        os.mkdir(args.output)
        print(str(args.output + "/logs"))
        os.mkdir(str(args.output + "/logs"))
    args.log_dir=str(args.output + "/logs")

    args.model_output = args.output + 'model.model'
    args.result_output = args.output + 'AUCs.txt'

    return args


def val(device, graph_data, loader_val, loss_fn, model, epoch, args, writer):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader_val):
            drug1_ids, drug2_ids, cell_ids, labels = data
            drug1_ids = drug1_ids.to(device)
            drug2_ids = drug2_ids.to(device)
            cell_ids = cell_ids.to(device)
            labels = labels.to(device)

            if loss_fn == F.binary_cross_entropy_with_logits:
                labels = labels.unsqueeze(-1)
                labels = torch.zeros(labels.shape[0], 2).to(labels).scatter_(-1, labels, 1) * 1.0

            logits = model(graph_data, drug1_ids, drug2_ids, cell_ids)
            loss = loss_fn(logits, labels)
            loss_list.append(loss)
            print("[Val] {} Epoch[{}/{}] step[{}/{}] loss={}".format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1, args.epochs, batch_idx + 1, len(loader_val), loss))
    avg_loss = sum(loss_list) / len(loss_list)
    writer.add_scalar("Loss/Validation", avg_loss, epoch)
    print("**********[Val] Epoch[{}/{}]  avg_loss={}".format(
        epoch + 1, args.epochs, avg_loss))


def predict(model, device, loader_test, graph_data, writer):
    model.eval()

    with torch.no_grad():
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        total_predlabels = torch.Tensor()
        print('Make prediction for {} samples...'.format(len(loader_test.dataset)))

        for step, data in enumerate(loader_test):
            print("Test Step[{}/{}]".format(step + 1, len(loader_test)))

            drug1_ids, drug2_ids, cell_ids, labels = data
            drug1_ids = drug1_ids.to(device)
            drug2_ids = drug2_ids.to(device)
            cell_ids = cell_ids.to(device)
            labels = labels.to(device)

            logits = model(graph_data, drug1_ids, drug2_ids, cell_ids)

            ys = F.softmax(logits, 1).to('cpu').data.numpy()

            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[-1], ys))

            total_predlabels = torch.cat((total_predlabels, torch.Tensor(predicted_labels)), 0)
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_labels = torch.cat((total_labels, labels.cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_predlabels.numpy().flatten()


def train(device, graph_data, loader_train, loss_fn, model, optimizer, epoch, args, writer):
    model.train()
    train_loss = 0.0
    for batch_idx, data in enumerate(loader_train):

        drug1_ids, drug2_ids, cell_ids, labels = data
        drug1_ids = drug1_ids.to(device)
        drug2_ids = drug2_ids.to(device)
        cell_ids = cell_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(graph_data, drug1_ids, drug2_ids, cell_ids)

        if loss_fn == F.binary_cross_entropy_with_logits:
            labels = labels.unsqueeze(-1)
            labels = torch.zeros(labels.shape[0], 2).to(labels).scatter_(-1, labels, 1) * 1.0

        loss = loss_fn(logits, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        for name, para in model.named_parameters():
            if para.grad is None:
                print(name)

        if batch_idx % args.log_step == 0:
            print("[Train] {} Epoch[{}/{}] step[{}/{}] loss={}".format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1, args.epochs, batch_idx + 1, len(loader_train), loss))

    writer.add_scalar("Loss/Training", train_loss / len(loader_train), epoch)


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
    with open(args.data, "rb") as f:
        dataset_dict = pickle.load(f)

    train_dataset = dataset_dict['train']
    val_dataset = dataset_dict['valid']
    test_dataset = dataset_dict['test']

    with open(args.graph, 'rb') as f:
        graph = pickle.load(f)
        graph = graph.to(device)

    loader_train = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    loader_val = DataLoader(val_dataset, batch_size=args.valid_batch_size, shuffle=None,
                            num_workers=args.num_workers, pin_memory=True)
    loader_test = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=None,
                             num_workers=args.num_workers, pin_memory=True)

    # ----------- Model Prepare ---------------------------------------------------
    if args.model=="emb":
        modeling = EmbModel
    elif args.model == "split":
        modeling = Emb_Split_Model
    # online_model = modeling(args).to(device)
    model = modeling(args).to(device)

    loss_fn = F.binary_cross_entropy_with_logits

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    # 学习率调整器，检测准确率的状态，然后衰减学习率
    scheduler = StepLR(step_size=20, gamma=0.1, optimizer=optimizer)

    print("model:", model)

    # ----------- Output Prepare ---------------------------------------------------
    AUCs = "%-10s%-15s%-15s%-15s%-15s%-15s%-15s%-15s" % ('Epoch', 'ACC', 'PREC', 'RECALL', 'AUC_ROC', 'AUC_PR', 'F1', 'KAPPA')
    # AUCs = 'Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL'
    with open(args.result_output, 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(str(k) + '=' + str(v) + "\n")
        f.write(str(model) + '\n')
        f.write(AUCs + '\n')

    # ----------- Training ---------------------------------------------------
    best_auc = 0

    epochs = args.epochs
    print('Training begin!')
    for epoch in range(epochs):
        train(device, graph, loader_train, loss_fn, model, optimizer, epoch, args, writer)

        val(device, graph, loader_val, loss_fn, model, epoch, args, writer)

        # T is correct label
        # S is predict score
        # Y is predict label
        T, S, Y = predict(model, device, loader_test, graph, writer)

        # compute preformence
        AUC = roc_auc_score(T, S)
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        RECALL = tp / (tp + fn)
        PREC = precision_score(T, Y)
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)
        F1 = f1_score(T, Y)

        writer.add_scalar("Metrics/ACC", ACC, epoch)
        writer.add_scalar("Metrics/PREC", PREC, epoch)
        writer.add_scalar("Metrics/RECALL", RECALL, epoch)
        writer.add_scalar("Metrics/AUC_ROC", AUC, epoch)
        writer.add_scalar("Metrics/AUC_PR", PR_AUC, epoch)
        writer.add_scalar("Metrics/F1", F1, epoch)

        # save data
        if best_auc < AUC:
            best_auc = AUC
            # AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall]
            AUCs = "%-10d%-15.4f%-15.4f%-15.4f%-15.4f%-15.4f%-15.4f%-15.4f" % (epoch, ACC, PREC, RECALL, AUC, PR_AUC, F1, KAPPA)

            with open(args.result_output, 'a') as f:
                f.write(AUCs + '\n')

            torch.save(model.state_dict(), args.model_output)

        print('best_auc', best_auc)
        scheduler.step()


if __name__ == '__main__':
    main()
