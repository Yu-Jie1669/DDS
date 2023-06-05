import torch
import torch.nn as nn

import torch.nn.functional as F
from modules.gnn import GNNBase


class EmbSplitModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.drug_emb = nn.Embedding(args.num_drug, args.hid)
        self.protein_emb = nn.Embedding(args.num_protein, args.hid)
        self.cell_emb = nn.Embedding(args.num_cell, args.hid)

        self.num_layer = args.num_layer
        self.dropout = args.dropout

        self.gnn = GNNBase(args.hid, args.hid, args.num_layer, args.head, args.dropout)

        self.classifier = nn.Sequential(
            nn.Linear(3 * args.hid, 6 * args.hid),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(6 * args.hid, 2 * args.hid),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(2 * args.hid, 2)
        )

    def forward(self, graph, drug1, drug2, cell):
        x_dict = graph.collect("x")

        emb_drugs = self.drug_emb(x_dict['drug'])
        emb_proteins = self.protein_emb(x_dict['protein'])
        emb_cells = self.cell_emb(x_dict['cell'])

        emb_x_dict = {
            'drug': emb_drugs,
            'protein': emb_proteins,
            'cell': emb_cells
        }

        edge_index_dict = graph.collect("edge_index")

        hid_x_dict = self.gnn(emb_x_dict, edge_index_dict)

        # [b, d]
        hid_drug1 = F.normalize(hid_x_dict['drug'][drug1], 2, 1)
        hid_drug2 = F.normalize(hid_x_dict['drug'][drug2], 2, 1)
        hid_cell = F.normalize(hid_x_dict['cell'][cell], 2, 1)

        hid = torch.cat((hid_drug1, hid_drug2, hid_cell), dim=1)

        logits = self.classifier(hid)

        return logits
