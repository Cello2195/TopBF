import torch
import pandas as pd
import numpy as np
import os
import os.path as osp
import torch.nn as nn
# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
from rdkit import Chem, RDLogger
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import QM9, ZINC
from torch_geometric.utils import sort_edge_index
from itertools import product
from torch_geometric.loader import DataLoader as GraphDataLoader
# import wandb
from tqdm import tqdm
import torch_geometric.utils
from torch_geometric.data import Batch
from torch_geometric.utils import to_undirected
import torch_geometric.utils
from torch_geometric.utils import to_dense_adj, to_dense_batch
from math import sqrt
from collections import defaultdict
from core.evaluation.stats import eval_graph_list
import pathlib
import moses
import pdb


def make_molecule(x, e, size, dataset):

    if dataset in ['qm9_wo_H', 'zinc']:
        _dict = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'Br': 4, 'Cl': 5, 'I': 6, 'P': 7, 'S': 8}
        x = torch.argmax(x, dim=-1)
        e = torch.argmax(e, dim=-1)
    elif dataset in ['qm9_wo_H_test', 'zinc_test']:
        _dict = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'Br': 4, 'Cl': 5, 'I': 6, 'P': 7, 'S': 8}
    elif dataset == 'moses':
        _dict = {'C': 0, 'N': 1, 'S': 2, 'O': 3, 'F': 4, 'Cl': 5, 'Br': 6}
    
    atom_dict = {v: k for k, v in _dict.items()}
    
    # pdb.set_trace()

    molecule = Chem.RWMol()
    for i in range(size):
        atom = x[i].item()
        molecule.AddAtom(Chem.Atom(atom_dict[atom]))

    # bonds = torch.argmax(e, dim=-1)

    # pdb.set_trace()

    for i, j in product(range(size), range(size)):
        if i < j:
            bond = e[i, j].item()
            if bond == 4:
                molecule.AddBond(i, j, Chem.BondType.AROMATIC)
            if bond == 3:
                molecule.AddBond(i, j, Chem.BondType.TRIPLE)
            elif bond == 2:
                molecule.AddBond(i, j, Chem.BondType.DOUBLE)
            elif bond == 1:
                molecule.AddBond(i, j, Chem.BondType.SINGLE)

    return molecule


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_dense(x, edge_index, edge_attr, batch, max_nodes):
    x = x.long()
    edge_index = edge_index.long()
    edge_attr = edge_attr.long()

    X, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=max_nodes)

    # pdb.set_trace()
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_nodes)
    # E = encode_no_edge(E)

    m = E.sum(dim=3) == 0
    ten = torch.zeros(E.shape[-1], device=E.device)
    ten[0] = 1

    E[m] = ten.long()

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=None), node_mask

def to_dense_pro(x, edge_index, edge_attr, batch, max_nodes):
    x = x.long()
    edge_index = edge_index.long()
    edge_attr = edge_attr.long()
    
    X, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=max_nodes)
    # pdb.set_trace()
    
    X_1based = torch.zeros(X.shape[0], X.shape[1], X.shape[2] + 1, device=X.device, dtype=X.dtype)
    
    # 将原始 one-hot 填充到新张量的 1: 之后的位置
    X_1based[..., 1:] = X

    m = X.sum(dim=2) == 0
    ten = torch.zeros(X.shape[2]+1, device=X.device)
    ten[0] = 1
    X_1based[m] = ten.long()

    # pdb.set_trace()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_nodes)
    # E = encode_no_edge(E)

    m = E.sum(dim=3) == 0
    ten = torch.zeros(E.shape[-1], device=E.device)
    ten[0] = 1

    E[m] = ten.long()

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X_1based, E=E, y=None), node_mask


def get_loaders(cfg):
    if cfg.dataset.name == 'qm9_wo_H':
        max_nodes = 9
        edge_feats = 4
        node_feats = 4
        if cfg.test:
            test_graphs = torch.load(f'dataset/{cfg.dataset.name}_test_graphs_{cfg.dataset.small_data}.pt')
            test_loader = GraphDataLoader(test_graphs, batch_size=cfg.optimization.batch_size, shuffle=False)
            return None, None, test_loader, node_feats, edge_feats, max_nodes
        # import pdb; pdb.set_trace()
        if os.path.exists(f'dataset/{cfg.dataset.name}_train_graphs_{cfg.dataset.small_data}.pt'):
            train_graphs = torch.load(f'dataset/{cfg.dataset.name}_train_graphs_{cfg.dataset.small_data}.pt')
            val_graphs = torch.load(f'dataset/{cfg.dataset.name}_val_graphs_{cfg.dataset.small_data}.pt')
            test_graphs = torch.load(f'dataset/{cfg.dataset.name}_test_graphs_{cfg.dataset.small_data}.pt')
            # pdb.set_trace()
        else:
            train_dataset, val_dataset, test_dataset = generate_loader(cfg.dataset.name, max_nodes, cfg.optimization.batch_size,
                                                                       cfg.dataset.small_data)
            # pdb.set_trace()
            train_graphs, train_smiles = process_graphs(train_dataset, max_nodes, cfg.dataset.name)
            val_graphs, val_smiles = process_graphs(val_dataset, max_nodes, cfg.dataset.name)
            test_graphs, test_smiles = process_graphs(test_dataset, max_nodes, cfg.dataset.name)
            
            torch.save(train_graphs, f'dataset/{cfg.dataset.name}_train_graphs_{cfg.dataset.small_data}.pt')
            torch.save(val_graphs, f'dataset/{cfg.dataset.name}_val_graphs_{cfg.dataset.small_data}.pt')
            torch.save(test_graphs, f'dataset/{cfg.dataset.name}_test_graphs_{cfg.dataset.small_data}.pt')

        train_loader = GraphDataLoader(train_graphs[:int(cfg.dataset.data_size * len(train_graphs))], batch_size=cfg.optimization.batch_size, shuffle=True)
        val_loader = GraphDataLoader(val_graphs[:int(cfg.dataset.data_size * len(val_graphs))], batch_size=cfg.optimization.batch_size, shuffle=False)
        test_loader = GraphDataLoader(test_graphs, batch_size=cfg.optimization.batch_size, shuffle=False)
        # get subset of loader

        # save dataloader

    elif cfg.dataset.name == 'zinc':

        max_nodes = 38
        edge_feats = 4
        node_feats = 9
        if cfg.test:
            test_graphs = torch.load(f'dataset/{cfg.dataset.name}_test_graphs_{cfg.dataset.small_data}.pt')
            test_loader = GraphDataLoader(test_graphs, batch_size=cfg.optimization.batch_size, shuffle=False)
            return None, None, test_loader, node_feats, edge_feats, max_nodes

        if os.path.exists(f'dataset/{cfg.dataset.name}_train_graphs_{cfg.dataset.small_data}.pt'):
            train_graphs = torch.load(f'dataset/{cfg.dataset.name}_train_graphs_{cfg.dataset.small_data}.pt')
            val_graphs = torch.load(f'dataset/{cfg.dataset.name}_val_graphs_{cfg.dataset.small_data}.pt')
            test_graphs = torch.load(f'dataset/{cfg.dataset.name}_test_graphs_{cfg.dataset.small_data}.pt')
            # pdb.set_trace()
        else:
            # from torch_geometric.datasets import ZINC
            train_dataset, val_dataset, test_dataset = generate_loader(cfg.dataset.name, max_nodes=38, batch_size=cfg.optimization.batch_size, small_data=cfg.dataset.small_data)
            train_graphs, train_smiles = process_graphs(train_dataset, 38)
            val_graphs, val_smiles = process_graphs(val_dataset, 38)
            test_graphs, test_smiles = process_graphs(test_dataset, 38)

            torch.save(train_graphs, f'dataset/{cfg.dataset.name}_train_graphs_{cfg.dataset.small_data}.pt')
            torch.save(val_graphs, f'dataset/{cfg.dataset.name}_val_graphs_{cfg.dataset.small_data}.pt')
            torch.save(test_graphs, f'dataset/{cfg.dataset.name}_test_graphs_{cfg.dataset.small_data}.pt')
        # pdb.set_trace()
        train_loader = GraphDataLoader(train_graphs, batch_size=cfg.optimization.batch_size, shuffle=True)
        val_loader = GraphDataLoader(val_graphs, batch_size=cfg.optimization.batch_size, shuffle=False)
        test_loader = GraphDataLoader(test_graphs, batch_size=cfg.optimization.batch_size, shuffle=False)
    elif cfg.dataset.name == 'zinc_pro':

        max_nodes = 38
        edge_feats = 4
        node_feats = 9
        if cfg.test:
            test_graphs = torch.load(f'dataset/{cfg.dataset.name}_test_graphs_{cfg.dataset.small_data}.pt')
            test_loader = GraphDataLoader(test_graphs, batch_size=cfg.optimization.batch_size, shuffle=False)
            return None, None, test_loader, node_feats, edge_feats, max_nodes

        if os.path.exists(f'dataset/{cfg.dataset.name}_train_graphs_{cfg.dataset.small_data}.pt'):
            train_graphs = torch.load(f'dataset/{cfg.dataset.name}_train_graphs_{cfg.dataset.small_data}.pt')
            val_graphs = torch.load(f'dataset/{cfg.dataset.name}_val_graphs_{cfg.dataset.small_data}.pt')
            test_graphs = torch.load(f'dataset/{cfg.dataset.name}_test_graphs_{cfg.dataset.small_data}.pt')
        else:
            # from torch_geometric.datasets import ZINC
            train_dataset, val_dataset, test_dataset = generate_loader(cfg.dataset.name, max_nodes=38, batch_size=cfg.optimization.batch_size, small_data=cfg.dataset.small_data)
            train_graphs, train_smiles = process_graphs(train_dataset, 38, cfg.dataset.name)
            val_graphs, val_smiles = process_graphs(val_dataset, 38, cfg.dataset.name)
            test_graphs, test_smiles = process_graphs(test_dataset, 38, cfg.dataset.name)

            torch.save(train_graphs, f'dataset/{cfg.dataset.name}_train_graphs_{cfg.dataset.small_data}.pt')
            torch.save(val_graphs, f'dataset/{cfg.dataset.name}_val_graphs_{cfg.dataset.small_data}.pt')
            torch.save(test_graphs, f'dataset/{cfg.dataset.name}_test_graphs_{cfg.dataset.small_data}.pt')
        # pdb.set_trace()
        train_loader = GraphDataLoader(train_graphs, batch_size=cfg.optimization.batch_size, shuffle=True)
        val_loader = GraphDataLoader(val_graphs, batch_size=cfg.optimization.batch_size, shuffle=False)
        test_loader = GraphDataLoader(test_graphs, batch_size=cfg.optimization.batch_size, shuffle=False)

    elif cfg.dataset.name == 'cifar10':
        # get cifar 10
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('data', train=True, download=True),
                                                    batch_size=cfg.optimization.batch_size, shuffle=True)

        for a in train_loader:
            print(a)
            quit()

    elif cfg.dataset.name == 'moses':
        max_nodes = 27
        edge_feats = 5
        node_feats = 7
        if cfg.test:
            test_graphs = torch.load(f'dataset/{cfg.dataset.name}_test_graphs_fitered_test.pt')
            test_loader = GraphDataLoader(test_graphs, batch_size=cfg.optimization.batch_size, shuffle=False)
            return None, None, test_loader, node_feats, edge_feats, max_nodes
        
        if os.path.exists(f'dataset/{cfg.dataset.name}_train_graphs_fitered.pt'):
            train_graphs = torch.load(f'dataset/{cfg.dataset.name}_train_graphs_fitered.pt')
            val_graphs = torch.load(f'dataset/{cfg.dataset.name}_val_graphs_fitered.pt')
            test_graphs = torch.load(f'dataset/{cfg.dataset.name}_test_graphs_fitered.pt')
            # pdb.set_trace()
        else:
            # from core.data.load_moses_data import MOSESDataset
            # train_dataset, val_dataset, test_dataset = generate_loader(cfg.dataset.name, max_nodes=38, batch_size=cfg.optimization.batch_size, small_data=cfg.dataset.small_data)
            train_dataset = MOSESDataset(root='core/data/data', split='train')
            val_dataset = MOSESDataset(root='core/data/data', split='val')
            test_dataset = MOSESDataset(root='core/data/data', split='test')
            
            train_graphs, train_smiles = process_graphs(train_dataset, max_nodes, 'moses')
            val_graphs, val_smiles = process_graphs(val_dataset, max_nodes, 'moses')
            test_graphs, test_smiles = process_graphs(test_dataset, max_nodes, 'moses')

            torch.save(train_graphs, f'dataset/{cfg.dataset.name}_train_graphs_fitered.pt')
            torch.save(val_graphs, f'dataset/{cfg.dataset.name}_val_graphs_fitered.pt')
            torch.save(test_graphs, f'dataset/{cfg.dataset.name}_test_graphs_fitered.pt')
        
        train_loader = GraphDataLoader(train_graphs, batch_size=cfg.optimization.batch_size, shuffle=True)
        val_loader = GraphDataLoader(val_graphs, batch_size=cfg.optimization.batch_size, shuffle=False)
        test_loader = GraphDataLoader(test_graphs, batch_size=cfg.optimization.batch_size, shuffle=False)

    elif cfg.dataset.name == 'mnist':
        # get mnist
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        # get loaders with transform
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform),
                                                    batch_size=cfg.optimization.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, download=True, transform=transform),
                                                    batch_size=cfg.optimization.batch_size, shuffle=False)

        test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, download=True, transform=transform),
                                                    batch_size=cfg.optimization.batch_size, shuffle=False)
        #
        #
        # for x_1, _ in train_loader:
        #     # visualize image
        #     # import Image
        #     import numpy as np
        #     from PIL import Image
        #
        #     img = x_1[0].squeeze().numpy()
        #     img = img * 255
        #     img = img.astype(np.uint8)
        #     img = Image.fromarray(img)
        #     img.show()
        #
        #
        #     quit()

        node_feats = 1
        edge_feats = 1
        max_nodes = 28 * 28

    else:
        raise ValueError(f'Invalid task: {cfg.dataset.name}')


    return train_loader, val_loader, test_loader, node_feats, edge_feats, max_nodes


def process_graphs(dataset, max_nodes, dataset_name=None):
    all_graphs, smiles = [], []
    loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
    c = 0

    for i, graph in tqdm(enumerate(loader)):
        if graph.x.shape[0] == 1:
            continue

        graphs, mask = to_dense(graph.x, graph.edge_index, graph.edge_attr, graph.batch, max_nodes)
        # pdb.set_trace()
        mols = [make_molecule(x, e, max_nodes, dataset_name) for x, e in zip(graphs.X, graphs.E)]
        # get largest fragment
        for mol in mols:
            try:
                Chem.SanitizeMol(mol)
                Chem.Kekulize(mol)

                # visualize mol
                # img = Draw.MolToImage(mol)
                # img.show()
                # import Draw
                from rdkit.Chem import Draw

                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                # img = Draw.MolToImage(largest_mol)
                # img.show()

                smile = Chem.MolToSmiles(largest_mol)
                smiles.append(smile)
                all_graphs.append(graph)

            except:
                continue

    return all_graphs, smiles


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1
        # e_mask = (node_mask.unsqueeze(2) & node_mask.unsqueeze(1)).unsqueeze(-1)  # bs, n, n, 1 xyd
        # import pdb; pdb.set_trace()
        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2 # xyd
            # self.E = self.E * e_mask
            # self.E = 0.5 * (self.E + self.E.transpose(1, 2)) # xyd
            
            # assert torch.allclose(self.E, torch.transpose(self.E, 1, 2)) # xyd
        return self
    

def get_tau_sched(tau_sched, tau_max=5, tau_min=0.1):
    if tau_sched == 'constant':
        tau_t = lambda t: torch.ones_like(t)
    elif tau_sched == 'linear':
        tau_t = lambda t: 1 - t
    elif tau_sched == 'cosine':
        tau_t = lambda t: 0.5 * (1 + torch.cos(t * 3.14159))
    else:
        raise ValueError(f'Invalid tau_sched: {tau_sched}')

    tau_rem = lambda t: (tau_max - tau_min) * tau_t(t) + tau_min

    return tau_rem


def get_model(cfg, node_feats, edge_feats):
    from core.module.transformer import GraphTransformer

    if cfg.dataset.small_model == 1:
        hidden_dims = {'dx': 16, 'de': 8, 'dy': 8, 'n_head': 2, 'dim_ffX': 16, 'dim_ffE': 8, 'dim_ffy': 8}
        hidden_mlp_dims = {'X': 32, 'E': 16, 'y': 16}
    elif cfg.dataset.name == 'abstract':
        hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 64, 'dim_ffy': 256}
        hidden_mlp_dims = {'X': 128, 'E': 64, 'y': 128}
    else:
        # old
        # hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 64, 'dim_ffy': 256}
        # hidden_mlp_dims = {'X': 128, 'E': 64, 'y': 128}
        hidden_dims = {'dx': 128, 'de': 64, 'dy': 128, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 64, 'dim_ffy': 256}
        hidden_mlp_dims = {'X': 256, 'E': 128, 'y': 128}


    model = GraphTransformer(
        input_dims={'X': node_feats, 'E': edge_feats, 'y': 1},
        # input_dims={'X': node_feats + 3, 'E': edge_feats, 'y': 1 + 6},
        hidden_dims=hidden_dims,
        hidden_mlp_dims=hidden_mlp_dims,
        output_dims={'X': node_feats, 'E': edge_feats, 'y': 1},
        # output_dims={'X': node_feats - 1, 'E': edge_feats, 'y': 1},
        n_layers=cfg.dynamics.n_layers,
        act_fn_in=nn.ReLU(),
        act_fn_out=nn.ReLU(),
    )

    return model

def project_simplex(n):
    fac_1 = torch.sqrt(torch.tensor([1 + 1/n])) * torch.ones(n, n)
    fac_2 = torch.pow(torch.tensor([n]), -(3/2)) * torch.ones(n, n)
    fac_3 = (torch.sqrt(torch.tensor([n + 1])) + 1) * torch.ones(n, n)

    verts = fac_1 * torch.eye(n) - fac_2 * fac_3
    extra_vert = torch.ones(1, n) * torch.pow(torch.tensor([n]), -(1/2))
    verts = torch.cat((verts, extra_vert), 0)
    return verts


def get_smiles(loader, max_nodes):
    test_smiles = []
    for mol in tqdm(loader.dataset):
        # get RDKit mol
        # make dense
        # import pdb; pdb.set_trace()
        mol, _ = to_dense(mol.x, mol.edge_index, mol.edge_attr, mol.batch, max_nodes)

        mol = make_molecule(mol.X.squeeze(), mol.E.squeeze(), max_nodes, None)
        smiles = Chem.MolToSmiles(mol)
        test_smiles.append(smiles)

    return test_smiles


def generate_loader(task, max_nodes, batch_size, small_data):
    if task[:3] == 'qm9':

        dataset = QM9(root='dataset/QM9')

        dataset = dataset.shuffle()
        dataset.shuffle()

        train_dataset = dataset[:100000]
        val_dataset = dataset[100000:120000]
        test_dataset = dataset[120000:]

        if small_data:
            train_dataset = [process_graph_qm9(mol, max_nodes=max_nodes) for mol in train_dataset[:1000]]
            val_dataset = [process_graph_qm9(mol, max_nodes=max_nodes) for mol in val_dataset[:1000]]
            test_dataset = [process_graph_qm9(mol, max_nodes=max_nodes) for mol in test_dataset[:1000]]
        else:
            train_dataset = [process_graph_qm9(mol, max_nodes=max_nodes) for mol in train_dataset]
            val_dataset = [process_graph_qm9(mol, max_nodes=max_nodes) for mol in val_dataset]
            test_dataset = [process_graph_qm9(mol, max_nodes=max_nodes) for mol in test_dataset]

    if task == 'zinc':
        train_dataset = ZINC(root='dataset/ZINC', subset=False, split='train')
        val_dataset = ZINC(root='dataset/ZINC', subset=False, split='val')
        test_dataset = ZINC(root='dataset/ZINC', subset=False, split='test')

        inds = torch.tensor([0,2,1,3,0,8,5,2,1,4,1,1,1,1,8,6,7,2,1,2,8,7,7,0,7,8,0,7])

        processes_graphs = {}

        datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}

        for k, v in datasets.items():
            gs = []

            for graph in tqdm(v):
                feat = graph.x
                feat = inds[feat]
                graph.x = torch.nn.functional.one_hot(feat.long(), num_classes=9).float()

                e = graph.edge_index.T.tolist()
                f = graph.edge_attr.tolist()

                edges, features = [], []
                d = {tuple(ex): ef for ex, ef in zip(e, f)}

                # edge index to list of edges

                # list of edges to set of edges
                # for i, j in product(range(graph.x.shape[0]), range(graph.x.shape[0])):
                #
                #     if (i, j) in d:
                #         edges.append([i, j])
                #
                #         fea = d[(i, j)]
                #         fea = fea + 1
                #
                #         features.append(fea)
                #
                #         print(fea)
                #
                # quit()
                edge_index = torch.tensor(edges).T
                edge_attr = torch.tensor(features)
                edge_attr = torch.nn.functional.one_hot(edge_attr.long(), num_classes=4).squeeze()

                process_graph = Data(x=graph.x.squeeze(), edge_index=graph.edge_index, edge_attr=torch.nn.functional.one_hot(graph.edge_attr.long(), num_classes=4).squeeze())

                gs.append(process_graph)

            processes_graphs[k] = gs

        train_dataset = processes_graphs['train']
        val_dataset = processes_graphs['val']
        test_dataset = processes_graphs['test']

    return train_dataset, val_dataset, test_dataset


def process_graph_qm9(mol, max_nodes):
    # get non-hydrogen nodes
    X = mol.x[:, 1:5]
    # remove all rows that where hydrogen
    non_H_nodes = ~X.eq(0).all(dim=1)
    num_non_H_nodes = torch.sum(non_H_nodes)
    X = X[non_H_nodes]
    # pad x with zeros to max_nodes
    # X = torch.cat((X, torch.zeros(size=(max_nodes - num_non_H_nodes, 4))), dim=0)
    # dict that maps each edge to its edge attribute
    non_H_indices = torch.arange(len(non_H_nodes))[non_H_nodes].tolist()
    dict = {old_i: i for i, old_i in enumerate(non_H_indices)}

    # loop over edges in index
    edge_index = mol.edge_index
    edge_attr = mol.edge_attr

    e = edge_index.T.tolist()
    f = edge_attr.tolist()

    edges, features = [], []
    d = {tuple(ex): ef for ex, ef in zip(e, f)}

    # edge index to list of edges

    # list of edges to set of edges
    for i, j in product(range(num_non_H_nodes), range(num_non_H_nodes)):


        if (i, j) in d:
            edges.append([i, j])


            fea = d[(i, j)]
            fea = np.argmax(fea) + 1

            features.append(fea)
        #
        # else:
        #     features.append(0)

    edge_index = torch.tensor(edges).T
    edge_attr = torch.tensor(features)
    edge_attr = torch.nn.functional.one_hot(edge_attr.long(), num_classes=4).squeeze()

    mask = torch.tensor([num_non_H_nodes * [True] + (max_nodes - num_non_H_nodes) * [False]]).squeeze().unsqueeze(-1)

    if X.shape[0] == 1:
        edge_attr = edge_attr.unsqueeze(0)

    return Data(x=X, edge_index=edge_index, edge_attr=edge_attr)


def allocate_memory_on_gpu(gpu_id, memory_in_mb):
    device = torch.device(f'cuda:{gpu_id}')
    memory_in_bytes = memory_in_mb * 1024 * 1024  # 将 MB 转为字节
    num_elements = memory_in_bytes // 4  # float32 占 4 字节
    tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
    return tensor


# def make_molecule(x, e, size, type):

#     dict = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'Br': 4, 'Cl': 5, 'I': 6, 'P': 7, 'S': 8}
#     atom_dict = {v: k for k, v in dict.items()}
#     #
#     # if type == 'qm9':
#     #     atom_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F'}
#     # import pdb; pdb.set_trace()
#     molecule = Chem.RWMol()
#     for i in range(size):
#         atom = torch.argmax(x[i]).item()
#         atom = Chem.Atom(atom_dict[atom])
#         molecule.AddAtom(atom)

#     bonds = torch.argmax(e, dim=-1)

#     for i, j in product(range(size), range(size)):
#         if i < j:
#             bond = bonds[i, j]
#             if bond == 4:
#                 molecule.AddBond(i, j, Chem.BondType.AROMATIC)
#             if bond == 3:
#                 molecule.AddBond(i, j, Chem.BondType.TRIPLE)
#             elif bond == 2:
#                 molecule.AddBond(i, j, Chem.BondType.DOUBLE)
#             elif bond == 1:
#                 molecule.AddBond(i, j, Chem.BondType.SINGLE)

#     return molecule


def make_final_molecule(x, e, size, type):

    dict = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'Br': 4, 'Cl': 5, 'I': 6, 'P': 7, 'S': 8}
    atom_dict = {v: k for k, v in dict.items()}
    #
    # if type == 'qm9':
    #     atom_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F'}
    # import pdb; pdb.set_trace()
    molecule = Chem.RWMol()
    for i in range(size):
        atom = Chem.Atom(atom_dict[x[i].item()])
        molecule.AddAtom(atom)

    for i, j in product(range(size), range(size)):
        if i < j:
            bond = e[i, j].item()
            if bond == 4:
                molecule.AddBond(i, j, Chem.BondType.AROMATIC)
            if bond == 3:
                molecule.AddBond(i, j, Chem.BondType.TRIPLE)
            elif bond == 2:
                molecule.AddBond(i, j, Chem.BondType.DOUBLE)
            elif bond == 1:
                molecule.AddBond(i, j, Chem.BondType.SINGLE)

    return molecule

def make_pro_molecule(x, e, size, type):
    """
    Creates an RDKit molecule from predicted node and edge categories.

    Args:
        x (Tensor): Node features after argmax, shape [size]. 
                    Index 0 represents "no atom".
        e (Tensor): Edge features after argmax, shape [size, size]. 
                    Index 0 represents "no bond".
        size (int): The max number of nodes.
        type (str): Dataset type, e.g., 'zinc' or 'qm9' to select atom dictionary.
    """

    # --- 1. 定义原子类型映射 ---
    # 假设您的数据处理后，类别索引 0 是“无原子”，所以真实原子从 1 开始
    if type.startswith("qm9") or type.startswith("zinc"):
        atom_decoder = {1: 'C', 2: 'N', 3: 'O', 4: 'F', 5: 'Br', 6: 'Cl', 7: 'I', 8: 'P', 9: 'S'}
    elif type.startswith("moses"):
        atom_decoder = {1: 'C', 2: 'N', 3: 'S', 4: 'O', 5: 'F', 6: 'Cl', 7: 'Br'}
    
    molecule = Chem.RWMol()
    
    # --- 2. 添加原子并建立索引映射 ---
    # `node_map` 将原始索引 i 映射到 molecule 中的新索引
    node_map = {} 
    
    for i in range(size):
        atom_type_idx = x[i].item()
        # 只有当类别索引 > 0 时，才是一个有效原子
        if atom_type_idx > 0:
            # `new_idx` 是原子在 rdkit molecule 对象中的真实索引
            new_idx = molecule.AddAtom(Chem.Atom(atom_decoder[atom_type_idx]))
            node_map[i] = new_idx
    
    # --- 3. 添加化学键，并进行有效性检查 ---
    for i, j in product(range(size), range(size)):
        # 只遍历上三角，避免重复和自环
        if i >= j:
            continue

        bond_type_idx = e[i, j].item()
        
        # 检查1：只有当键类型 > 0 时，才是一个有效化学键
        if bond_type_idx > 0:
            # 检查2：【关键】只有当这条键连接的两个节点都是有效原子时，才添加
            if i in node_map and j in node_map:
                # 使用 node_map 获取在 molecule 中的真实索引
                start_idx = node_map[i]
                end_idx = node_map[j]

                # 根据索引添加不同类型的化学键
                if bond_type_idx == 4:
                    bond_type = Chem.BondType.AROMATIC
                elif bond_type_idx == 3:
                    bond_type = Chem.BondType.TRIPLE
                elif bond_type_idx == 2:
                    bond_type = Chem.BondType.DOUBLE
                elif bond_type_idx == 1:
                    bond_type = Chem.BondType.SINGLE
                else:
                    # 如果有其他键类型，可以在此添加，否则跳过
                    continue
                
                molecule.AddBond(start_idx, end_idx, bond_type)

    return molecule


def eval_and_log(mols, log, smiles, device):
    all_valid_mols, all_unique_mols, all_novel_mols = defaultdict(list), defaultdict(list), defaultdict(list)

    # for k, k_mols in mols.items():

    valid_mols, unique_mols, novel_mols = [], [], []
    valid, unique, novel = 0, 0, 0

    for mol in mols:
        try:
            Chem.SanitizeMol(mol)
            Chem.Kekulize(mol)
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            valid_mols.append(largest_mol)
            valid += 1
        except:
            continue

    train_smiles, test_smiles = smiles
    unique_set = set([Chem.MolToSmiles(mol) for mol in valid_mols])
    # pdb.set_trace()
    # moses.get_all_metrics(unique_set)
    # novel_set = unique_set - set(train_smiles)

    percentage_valid = valid / len(mols)
    percentage_unique = len(unique_set) / len(valid_mols) if len(valid_mols) > 0 else 0
    # percetnage_novel = len(novel_set) / len(unique_set) if len(unique_set) > 0 else 0

    # can_train = [Chem.CanonSmiles(train_smile) for train_smile in train_smiles]
    # can_test = [Chem.CanonSmiles(test_smile) for test_smile in test_smiles]
    # can_gen = [Chem.CanonSmiles(mol) for mol in unique_mols]
    # can_novel = set(can_gen) - set(can_test)
    # from fcd_torch import FCD
    #
    unique_list = list(unique_set)
    # test_list = list(test_smiles)

    if len(unique_list) > 0:
        from fcd import get_fcd, load_ref_model, canonical_smiles, get_predictions, calculate_frechet_distance
        # model = load_ref_model()
        can_test = [w for w in canonical_smiles(test_smiles) if w is not None]
        can_gen = [w for w in canonical_smiles(unique_list) if w is not None]
        can_train = [w for w in canonical_smiles(train_smiles) if w is not None]

        # get novel ones
        can_novel = set(can_gen) - set(can_train)
        novelty = len(can_novel) / len(can_gen)

        # test_fc = get_predictions(model, can_test)
        # gen_fc = get_predictions(model, can_gen)

        # def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        #     import numpy as np
        #     from scipy.linalg import sqrtm
        #     """Calculate the Frechet distance between two multivariate Gaussians."""
        #     mu_diff = mu1 - mu2

        #     # Add a small epsilon to the covariance matrices to ensure positive semi-definiteness
        #     sigma1 += np.eye(sigma1.shape[0]) * eps
        #     sigma2 += np.eye(sigma2.shape[0]) * eps

        #     # Compute the square root of the product of covariance matrices
        #     covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        #     if np.iscomplexobj(covmean):
        #         covmean = covmean.real

        #     # Calculate the Fréchet distance
        #     trace_term = np.trace(sigma1 + sigma2 - 2 * covmean)
        #     return (mu_diff.dot(mu_diff) + trace_term).real

        # # Example usage
        # mu_real, sigma_real = np.mean(test_fc, axis=0), np.cov(test_fc, rowvar=False)
        # mu_gen, sigma_gen = np.mean(gen_fc, axis=0), np.cov(gen_fc, rowvar=False)
        # try:
        #     fcd_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        # except:
        #     fcd_score = 1e8
        # import pdb; pdb.set_trace()
        from fcd_torch import FCD
        fcd = FCD(device='cuda:0', n_jobs=8)
        fcd_score = fcd(can_gen, can_test)
        mol_test = [Chem.MolFromSmiles(w) for w in can_test]
        mol_gen = [Chem.MolFromSmiles(w) for w in can_gen]
        scores_nspdk = eval_graph_list(mols_to_nx(mol_test), mols_to_nx(mol_gen), methods=['nspdk'])['nspdk']
    else:
        novelty = 0
        fcd_score = 1e8
        can_gen = []
        can_novel = []

    print(f"Valid: {percentage_valid}\nUnique: {percentage_unique}\nNovelty: {novelty}\nFCD: {fcd_score}\nNSPDK: {scores_nspdk}")

    s = max(percentage_unique, percentage_valid)

    print(percentage_valid, percentage_unique, s)

    if log:
        wandb.log({
            f'Validity': percentage_valid,
            f'Uniqueness': percentage_unique,
            f'Novelty': novelty,
            f'FCD': fcd_score,
            f'NSPDK': scores_nspdk,
            f'Ablation': s
        })

    return valid_mols, percentage_valid, percentage_unique, novelty, fcd_score, scores_nspdk

def eval_and_log_moses(mols, log, smiles, device):
    all_valid_mols, all_unique_mols, all_novel_mols = defaultdict(list), defaultdict(list), defaultdict(list)

    # for k, k_mols in mols.items():

    valid_mols, unique_mols, novel_mols = [], [], []
    valid, unique, novel = 0, 0, 0

    for mol in mols:
        try:
            Chem.SanitizeMol(mol)
            Chem.Kekulize(mol)
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            valid_mols.append(largest_mol)
            valid += 1
        except:
            continue

    train_smiles, test_smiles = smiles
    unique_set = set([Chem.MolToSmiles(mol) for mol in valid_mols])
    # pdb.set_trace()
    metrics = moses.get_all_metrics(unique_set, k=len(unique_set), n_jobs=8, test=test_smiles, train=train_smiles)
    # scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=self.device[0], n_jobs=8, test=test_smiles, train=train_smiles)

    # novel_set = unique_set - set(train_smiles)

    percentage_valid = valid / len(mols)
    percentage_unique = len(unique_set) / len(valid_mols) if len(valid_mols) > 0 else 0
    
    unique_list = list(unique_set)
    # test_list = list(test_smiles)

    if len(unique_list) > 0:
        from fcd import get_fcd, load_ref_model, canonical_smiles, get_predictions, calculate_frechet_distance
        # model = load_ref_model()
        can_test = [w for w in canonical_smiles(test_smiles) if w is not None]
        can_gen = [w for w in canonical_smiles(unique_list) if w is not None]
        can_train = [w for w in canonical_smiles(train_smiles) if w is not None]

        # get novel ones
        can_novel = set(can_gen) - set(can_train)
        novelty = len(can_novel) / len(can_gen)

        mol_test = [Chem.MolFromSmiles(w) for w in can_test]
        mol_gen = [Chem.MolFromSmiles(w) for w in can_gen]

        fcd_score = metrics['FCD/Test']

        # scores_nspdk = eval_graph_list(mols_to_nx(mol_test), mols_to_nx(mol_gen), methods=['nspdk'])['nspdk']
        scores_nspdk = 0
    else:
        novelty = 0
        fcd_score = 1e8
        can_gen = []
        can_novel = []

    print(f"Valid: {percentage_valid}\nUnique: {percentage_unique}\nNovelty: {novelty}\nFCD: {fcd_score}\nNSPDK: {scores_nspdk}")

    s = max(percentage_unique, percentage_valid)

    print(percentage_valid, percentage_unique, s)

    return valid_mols, percentage_valid, percentage_unique, novelty, fcd_score, scores_nspdk, metrics

def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       label=atom.GetSymbol())
                    #    atomic_num=atom.GetAtomicNum(),
                    #    formal_charge=atom.GetFormalCharge(),
                    #    chiral_tag=atom.GetChiralTag(),
                    #    hybridization=atom.GetHybridization(),
                    #    num_explicit_hs=atom.GetNumExplicitHs(),
                    #    is_aromatic=atom.GetIsAromatic())
                    
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       label=int(bond.GetBondTypeAsDouble()))
                    #    bond_type=bond.GetBondType())
        
        nx_graphs.append(G)
    return nx_graphs


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


class MOSESDataset(InMemoryDataset):
    train_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv'
    val_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv'
    test_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv'
    atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br']

    def __init__(self, root, split, filter_dataset=True, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        # self.atom_decoder = atom_decoder
        self.filter_dataset = filter_dataset
        self.file_idx = {'train': 0, 'val': 1, 'test': 2}

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx[split]])

    @property
    def raw_file_names(self):
        return ['train_moses.csv', 'val_moses.csv', 'test_moses.csv']
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'moses', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'moses', 'processed')

    @property
    def split_paths(self):
        return [osp.join(self.raw_dir, f) for f in self.raw_file_names]

    @property
    def processed_file_names(self):
        if self.filter_dataset:
            return ['train_filtered.pt', 'val_filtered.pt', 'test_filtered.pt']
        else:
            return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        train_path = download_url(self.train_url, self.raw_dir)
        os.rename(train_path, osp.join(self.raw_dir, 'train_moses.csv'))

        test_path = download_url(self.test_url, self.raw_dir)
        os.rename(test_path, osp.join(self.raw_dir, 'val_moses.csv'))

        valid_path = download_url(self.val_url, self.raw_dir)
        os.rename(valid_path, osp.join(self.raw_dir, 'test_moses.csv'))


    def process(self):
        RDLogger.DisableLog('rdApp.*')
        types = {atom: i for i, atom in enumerate(self.atom_decoder)}

        bonds = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}

        file_idx2name = {0: 'train', 1: 'val', 2: 'test'}

        for file_idx in range(len(file_idx2name)):
            print(file_idx2name[file_idx])

            path = self.split_paths[file_idx]
            smiles_list = pd.read_csv(path)['SMILES'].values

            data_list = []
            smiles_kept = []

            for i, smile in enumerate(tqdm(smiles_list)):
                mol = Chem.MolFromSmiles(smile)
                mol = Chem.RemoveHs(mol) # xyd
                N = mol.GetNumAtoms()

                type_idx = []
                for atom in mol.GetAtoms():
                    type_idx.append(types[atom.GetSymbol()])

                row, col, edge_type = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    edge_type += 2 * [bonds[bond.GetBondType()] + 1]

                if len(row) == 0:
                    continue

                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)
                edge_attr = torch.nn.functional.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

                x = torch.nn.functional.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
                y = torch.zeros(size=(1, 0), dtype=torch.float)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

                if self.filter_dataset:
                    # Try to build the molecule again from the graph. If it fails, do not add it to the training set
                    dense_data, node_mask = to_dense_pro(data.x, data.edge_index, data.edge_attr, data.batch, max_nodes=27)
                    dense_data = dense_data.mask(node_mask, collapse=True)
                    X, E = dense_data.X, dense_data.E
                    # pdb.set_trace()
                    assert X.size(0) == 1
                    atom_types = X[0]
                    edge_types = E[0]
                    mol = make_pro_molecule(atom_types, edge_types, 27, "moses")
                    smiles = mol2smiles(mol)
                    if smiles is not None:
                        try:
                            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                            if len(mol_frags) == 1:
                                data_list.append(data)
                                smiles_kept.append(smiles)

                        except Chem.rdchem.AtomValenceException:
                            print("Valence error in GetmolFrags")
                        except Chem.rdchem.KekulizeException:
                            print("Can't kekulize molecule")
                else:
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)

            torch.save(self.collate(data_list), self.processed_paths[file_idx])

            if self.filter_dataset:
                smiles_save_path = osp.join(pathlib.Path(self.raw_paths[file_idx]).parent, f'new_{file_idx2name[file_idx]}.smiles')
                print(smiles_save_path)
                with open(smiles_save_path, 'w') as f:
                    f.writelines('%s\n' % s for s in smiles_kept)
                print(f"Number of molecules kept: {len(smiles_kept)} / {len(smiles_list)}")