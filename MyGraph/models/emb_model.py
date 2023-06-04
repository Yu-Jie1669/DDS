import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import GATConv, GCNConv, HeteroConv
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F


class EmbModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.drug_emb = nn.Embedding(args.num_drug, args.hid)
        self.protein_emb = nn.Embedding(args.num_protein, args.hid)
        self.cell_emb = nn.Embedding(args.num_cell, args.hid)

        self.num_layer = args.num_layer
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.dropout = args.dropout

        self.GNN = GATConv

        for layer in range(args.num_layer):
            gnn = HeteroConv({
                ("drug", "d-d", "drug"): self.GNN(args.hid, args.hid, args.head, edge_dim=1, dropout=args.dropout),
                # ("drug", "d-p", "protein"): self.GNN(args.hid, args.hid, args.head, dropout=args.dropout, add_self_loops=False),
                ("protein", "rev_d-p", "drug"): self.GNN(args.hid, args.hid, args.head, dropout=args.dropout, add_self_loops=False),
                # ("protein", "p-p", "protein"): self.GNN(args.hid, args.hid, args.head, dropout=args.dropout),
                # ("cell", "c-p", "protein"): self.GNN(args.hid, args.hid, args.head, dropout=args.dropout, add_self_loops=False),
                ("protein", "rev_c-p", "cell"): self.GNN(args.hid, args.hid, args.head, dropout=args.dropout, add_self_loops=False),
            })
            self.convs.append(gnn)

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

        for layer in range(self.num_layer):
            emb_x_dict = self.convs[layer](emb_x_dict, graph.collect("edge_index"), graph.collect("edge_attr"))
            emb_x_dict = {key: F.relu(x) for key, x in emb_x_dict.items()}

        # [b, d]
        hid_drug1 = F.normalize(emb_x_dict['drug'][drug1], 2, 1) 
        hid_drug2 = F.normalize(emb_x_dict['drug'][drug2], 2, 1) 
        hid_cell = F.normalize(emb_x_dict['cell'][cell], 2, 1) 

        hid = torch.cat((hid_drug1, hid_drug2, hid_cell), dim=1)

        logits = self.classifier(hid)

        return logits
