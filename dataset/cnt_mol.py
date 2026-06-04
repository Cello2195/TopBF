import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import defaultdict
import torch
from tqdm import tqdm
import argparse
import pdb
from core.utils.loader import load_data
from core.utils.graph_utils import node_flags
from core.config.config import Config

def count_atoms_and_bonds_from_loader(loader):
    atom_counts = defaultdict(int)
    bond_counts = defaultdict(int)

    for batch in tqdm(loader):
        x, adj = batch  # x: [B, N, D], adj: [B, N, N]
        atom_types = x.argmax(dim=-1)           # [B, N]
        node_mask = node_flags(adj).bool()   # [B, N]

        B, N = atom_types.shape

        for b in range(B):
            for i in range(N):
                if not node_mask[b, i]:
                    continue
                atom_type = atom_types[b, i].item()
                atom_counts[atom_type] += 1

                for j in range(N):
                    if not node_mask[b, j] or i == j:
                        continue
                    bond_type = adj[b, i, j].item()
                    bond_counts[bond_type] += 1

    # # 注意：由于 i-j 和 j-i 都统计了 bond，会重复计数一次
    # for bond_type in bond_counts:
    #     bond_counts[bond_type] = bond_counts[bond_type] // 2

    return dict(atom_counts), dict(bond_counts)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Count atom and bond types in molecule dataset.')
    parser.add_argument("--config_file", type=str, default="configs/bfn4molgen_v13.yaml") # Assuming a config file structure
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--ckpt_pattern", type=str, default="last.ckpt")
    parser.add_argument('--dataset', type=str, default='QM9', choices=['QM9', 'ZINC250k'])
    args = parser.parse_args()
    data_name = args.dataset
    cfg = Config(**args.__dict__) # Load configuration
    cfg.dataset.name = args.dataset

    train_loader, test_loader = load_data(cfg)

    atom_count_1, bond_count_1 = count_atoms_and_bonds_from_loader(train_loader)
    atom_count_2, bond_count_2 = count_atoms_and_bonds_from_loader(test_loader)

    # 合并两个字典
    from collections import Counter
    total_atoms = dict(Counter(atom_count_1) + Counter(atom_count_2))
    total_bonds = dict(Counter(bond_count_1) + Counter(bond_count_2))

    print("Atom counts:")
    for k, v in sorted(total_atoms.items()):
        print(f"Atom type {k}: {v}")

    print("Bond counts:")
    for k, v in sorted(total_bonds.items()):
        print(f"Bond type {k}: {v}")
