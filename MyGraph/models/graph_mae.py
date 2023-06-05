from functools import partial

import torch.nn as nn
import torch
from torch_geometric.nn.conv import GATConv
import torch.nn.functional as F


class GraphMAE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.emb = nn.Embedding(args.num_protein, args.hid)

        self.enc = GATConv(args.hid, args.hid, args.head, dropout=args.dropout)

        self.dec = GATConv(args.hid, args.hid, args.head, dropout=args.dropout)

        self.encoder_to_decoder = nn.Linear(args.hid, args.hid, bias=False)

        self.mask = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, args.hid)).float(), requires_grad=True)

        self.alpha_l = args.alpha_l

        if args.dmask:
            self.dmask = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, args.hid)).float(), requires_grad=True)
        else:
            self.dmask = None

    def forward(self, x_dict, edge_index_dict, mask_ratio=0.3):
        x = x_dict['protein']
        edge = edge_index_dict[('protein', 'p-p', 'protein')]

        x_emb = self.emb(x)

        x_masked, mask_select = self.enc_mask(x_emb, mask_ratio)

        x_enc = self.enc(x_masked, edge)

        x_rep = self.encoder_to_decoder(x_enc)

        if self.dmask is not None:
            x_rep = self.dec_mask(x_rep, mask_select)

        x_dec = self.dec(x_rep, edge)

        loss = self.cal_loss(x_emb[mask_select], x_dec[mask_select])

        return loss

    def enc_mask(self, x, mask_ratio):
        num = x.shape[0]
        mask_select = torch.randperm(num)[:int(mask_ratio * num)]

        x[mask_select] = self.mask

        return x, mask_select

    def dec_mask(self, x, mask_select):
        x[mask_select] = self.dmask
        return x

    def cal_loss(self, origin, output):
        def sce_loss(x, y, alpha=3):
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)

            # loss =  - (x * y).sum(dim=-1)
            # loss = (x_h - y_h).norm(dim=1).pow(alpha)

            loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

            loss = loss.mean()
            return loss

        return sce_loss(origin, output, self.alpha_l)
