import torch
import torch.nn as nn
from torch_geometric.nn.conv import GATConv, GCNConv, HeteroConv


class EmbModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.drug_emb = nn.Embedding(args.num_drug, args.hid)
        self.protein_emb = nn.Embedding(args.num_protein, args.hid)
        self.cell_emb = nn.Embedding(args.num_cell, args.hid)

        self.convs = torch.nn.ModuleList()
        for _ in range(args.num_layers):
            gnn = HeteroConv({
                ("drug", "d-d", "drug"): GATConv(args.hid, args.hid, args.head),
                ("drug", "d-p", "protein"): GATConv(args.hid, args.hid, args.head, add_self_loops=False),
                ("protein", "rev_d-p", "drug"): GATConv(args.hid, args.hid, args.head, add_self_loops=False),
                ("protein", "p-p", "protein"): GATConv(args.hid, args.hid, args.head),
                ("cell", "c-p", "protein"): GATConv(args.hid, args.hid, args.head, add_self_loops=False),
                ("protein", "rev_c-p", "cell"): GATConv(args.hid, args.hid, args.head, add_self_loops=False)
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

        for conv in self.convs:
            emb_x_dict = conv(emb_x_dict, graph.collect("edge_index"), graph.collect("edge_attr"))
            emb_x_dict = {key: x.relu() for key, x in emb_x_dict.items()}

        # [b, d]
        hid_drug1 = torch.norm(emb_x_dict['drug'][drug1])
        hid_drug2 = torch.norm(emb_x_dict['drug'][drug2])
        hid_cell = torch.norm(emb_x_dict['cell'][cell])

        hid = torch.cat((hid_drug1, hid_drug2, hid_cell), dim=1)

        logits = self.classifier(hid)

        return logits
