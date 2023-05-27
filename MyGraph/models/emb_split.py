import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import GATConv, GCNConv, HeteroConv
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F


class Emb_Split_Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.drug_emb = nn.Embedding(args.num_drug, args.hid)
        self.protein_emb = nn.Embedding(args.num_protein, args.hid)
        self.cell_emb = nn.Embedding(args.num_cell, args.hid)

        self.num_layer = args.num_layer
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.dropout = args.dropout

        # for layer in range(args.num_layer):
        #     self.batch_norms.append(torch.nn.BatchNorm1d(args.hid))

        self.GNN = GATConv

        for layer in range(args.num_layer):
            model_dict = nn.ModuleDict()
            # model_dict['d-d'] = self.GNN(args.hid, args.hid, args.head, edge_dim=1, dropout=args.dropout)
            model_dict['d-p'] = self.GNN(args.hid, args.hid, args.head, dropout=args.dropout, add_self_loops=False)
            model_dict['rev_d-p'] = self.GNN(args.hid, args.hid, args.head, dropout=args.dropout, add_self_loops=False)
            model_dict['p-p'] =  self.GNN(args.hid, args.hid, args.head, dropout=args.dropout)
            model_dict['c-p'] = self.GNN(args.hid, args.hid, args.head, dropout=args.dropout, add_self_loops=False)
            model_dict['rev_c-p'] = self.GNN(args.hid, args.hid, args.head, dropout=args.dropout, add_self_loops=False)

            self.convs.append(model_dict)

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

        edge_attr_dict = graph.collect("edge_attr")
        
        # update_edge_attr_dict = {}
        # for key in edge_attr_dict.keys():
        #     update_edge_attr_dict[key] = edge_attr_dict[key]*(-1.0)

        for layer in range(self.num_layer):
            emb_x_dict = self.convs[layer](emb_x_dict, graph.collect("edge_index"), graph.collect("edge_attr"))
            emb_x_dict = {key: F.relu(x) for key, x in emb_x_dict.items()}

            # if layer == self.num_layer - 1:
            #     emb_x_dict = {key: F.dropout(self.batch_norms[layer](x), self.dropout, training=self.training)
            #                   for key, x in emb_x_dict.items()}
            # else:
            #     emb_x_dict = {key: F.dropout(F.relu(self.batch_norms[layer](x)), self.dropout, training=self.training)
            #                   for key, x in emb_x_dict.items()}

        # [b, d]
        hid_drug1 = F.normalize(emb_x_dict['drug'][drug1], 2, 1) 
        hid_drug2 = F.normalize(emb_x_dict['drug'][drug2], 2, 1) 
        hid_cell = F.normalize(emb_x_dict['cell'][cell], 2, 1) 

        hid = torch.cat((hid_drug1, hid_drug2, hid_cell), dim=1)

        logits = self.classifier(hid)

        return logits
