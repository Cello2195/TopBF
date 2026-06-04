# geobfn_train.py (Modified)

import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 只使用第1张卡
import json
import pickle
from typing import Any, Optional
import pytorch_lightning as pl
import argparse
import copy
from pytorch_lightning.utilities.types import STEP_OUTPUT

from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
from core.config.config import Config # Assuming Config class is available
from core.model.bfn.bfn_base_QM9_qw import bfn4GraphTransformer # Our new GraphTransformer-based model
import torch
import os
import datetime, pytz
import pdb


from core.losses import loss # Assuming core.losses still relevant for other parts
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import wandb
from core.callbacks.basic import (
    Gradient_clip,
    DebugCallback,
    NormalizerCallback,
    RecoverCallback,
    EMACallback,
)
from absl import logging
from core.data.prefetch import PrefetchLoader # Keep if batching is used
import core.utils.ctxmgr as ctxmgr
from core.utils.utils import *
from core.utils.loader import load_data
from core.utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles
from core.evaluation.molsets import get_all_metrics


class BFN4GraphGenTrain(pl.LightningModule): # Renamed class for clarity
    def __init__(self, config, node_feats, edge_feats, max_nodes):
        super().__init__()
        self.cfg = config
        # self._device = device
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.max_nodes = max_nodes

        qw_kwargs = {
            'use_qw_loss': self.cfg.dynamics.use_qw_loss,
            'qw_weight': self.cfg.dynamics.qw_weight,
            'qw_eps': self.cfg.dynamics.qw_eps,
            'qw_iters': self.cfg.dynamics.qw_iters,
            'qw_on_edges': self.cfg.dynamics.qw_on_edges,
        }

        # Instantiate our new GraphTransformer-based BFN model
        self.dynamics = bfn4GraphTransformer(
            # in_node_nf and in_edge_nf for bfn4GraphTransformer are effectively 1 for continuous input,
            # but we pass out_node_dim and out_edge_dim for K_X and K_A
            # These are used to determine the `num_classes` in map_discrete_to_continuous.
            # No `in_node_nf` or `in_edge_nf` explicitly passed here, as it's determined internally.
            hidden_nf=self.cfg.dynamics.hidden_nf,
            n_layers=self.cfg.dynamics.n_layers,
            n_head=self.cfg.dynamics.n_head,
            dim_ffX=self.cfg.dynamics.dim_ffX,
            dim_ffE=self.cfg.dynamics.dim_ffE,
            dim_ffy=self.cfg.dynamics.dim_ffy,
            # dropout=self.cfg.dynamics.dropout,
            max_nodes=max_nodes,
            # device=self.device,
            n_node_histogram = self.cfg.dataset.n_node_histogram,
            t_min=self.cfg.dynamics.t_min,
            # IMPORTANT: Pass sigma1_A and sigma1_X as per GeoBFN original
            sigma1_X=self.cfg.dynamics.sigma1_X, # Placeholder for coordinates
            sigma1_A=self.cfg.dynamics.sigma1_A, # Assuming this is the sigma for discrete features
            bins_X=node_feats,
            bins_A=edge_feats,
            **qw_kwargs
        )
        # pdb.set_trace()
        self.save_hyperparameters(vars(config)) # Save config for logging

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        # The core training logic is now within bfn4GraphTransformer's training_step
        # We just pass the batch data directly.
        # self.dynamics = self.dynamics.to(self._device)
        # pdb.set_trace()
        # batch = batch.to(self.device)
        loss = self.dynamics.training_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.optimization.batch_size)
        return loss

    def configure_optimizers(self):
        # Standard Adam optimizer setup
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optimization.lr, weight_decay=self.cfg.optimization.weight_decay)
        return optimizer

    # Evaluation/validation_step will need to be adapted for graph metrics
    def validation_step(self, batch: Any, batch_idx: int):
        # 1. 计算验证损失
        # self.dynamics = self.dynamics.to(self._device) # 这一行通常不需要，因为模型已在正确设备上
        # pdb.set_trace()
        # batch = batch.to(self.device)
        loss = self.dynamics.training_step(batch) # 复用 training_step 计算损失
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.optimization.batch_size)
        return {'loss': loss, 'generated_data': batch}

    # Removed on_train_start, on_train_epoch_end, on_validation_epoch_end, on_test_epoch_end for brevity,
    # as they would also need specific adaptations for graph generation metrics/logging.

# --- Main script part ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/QM9_qw.yaml") # Assuming a config file structure
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--logging_level", type=str, default="warning")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sigma1_X", type=float, default=0.2)
    parser.add_argument("--sigma1_A", type=float, default=0.2)
    parser.add_argument("--beta1", type=float, default=2.0)
    parser.add_argument("--sample_steps", type=int, default=1000)
    parser.add_argument("--eval_data_num", type=int, default=5)
    parser.add_argument("--checkpoint_freq", type=int, default=10)
    parser.add_argument("--exp_version", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ckpt_pattern", type=str, default="last-v1.ckpt")
    # parser.add_argument("--ckpt_pattern", type=str, default="epoch=1919-val_loss=0.7931125164031982.ckpt")
    # parser.add_argument("--ckpt_pattern", type=str, default="epoch=589-val_loss=0.8985304236412048-last=0.ckpt")
    _args = parser.parse_args()

    cfg = Config(**_args.__dict__) # Load configuration
    # print(f"The config of this process is:\n{cfg}")
    device = torch.device("cuda")

    # memory_in_mb = int(1024*9.295)
    # tensor = allocate_memory_on_gpu(0, memory_in_mb)
    # pdb.set_trace()

    # For now, we'll create dummy loaders for minimal testing setup
    # train_loader, val_loader, test_loader, node_feats, edge_feats, max_nodes = get_loaders(cfg)
    train_loader, test_loader = load_data(cfg)
    # pdb.set_trace()

    # Initialize model
    model = BFN4GraphGenTrain(cfg, node_feats=4, edge_feats=4, max_nodes=9).to(device)
    # pdb.set_trace()
    
    # For now, keep basic callbacks.
    # import pdb; pdb.set_trace()
    callbacks = [
        RecoverCallback(
            latest_ckpt=cfg.accounting.checkpoint_path,
            resume=cfg.optimization.resume or cfg.test,
            recover_trigger_loss=cfg.optimization.recover_trigger_loss,
            skip_count_limit=cfg.optimization.skip_count_limit,
        ),
        Gradient_clip(
            maximum_allowed_norm=cfg.optimization.maximum_allowed_norm,
            # maximum_allowed_norm=5.0,
        ),
        NormalizerCallback(normalizer_dict=cfg.dataset.normalizer_dict),
        EMACallback(decay=0.9999, ema_device="cuda"),
        ModelCheckpoint(
            dirpath=cfg.accounting.checkpoint_dir,
            # filename="{epoch}-{val_loss:.6f}",
            filename="{epoch}-{val_loss}", # <--- 临时修改为不带格式化
            every_n_epochs=cfg.accounting.checkpoint_freq,
            save_last=True,
            save_top_k=100,
            mode="min",
            monitor="val_loss",
        ),
    ]

    # Initialize WandbLogger if cfg.accounting.use_wandb is enabled
    # wandb_logger = None
    # if cfg.accounting.use_wandb:
    #     wandb_logger = WandbLogger(project=cfg.accounting.project_name, name=cfg.accounting.run_name)

    trainer = pl.Trainer(
        limit_test_batches=1 if cfg.test else None, # Only limit test batches if in test mode
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.optimization.epochs,
        check_val_every_n_epoch=cfg.accounting.checkpoint_freq,
        devices=1,
        # accelerator="gpu" if cfg.train.gpus > 0 else "cpu", # Use accelerator based on GPU config
        accelerator="gpu",
        # logger=wandb_logger,
        callbacks=callbacks,
        num_sanity_val_steps=2,
        # overfit_batches=10, # Uncomment for debugging/overfitting a small batch
    )

    if not cfg.test:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    else:
        # cfg.accounting.checkpoint_freq = "samping!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # print(cfg.accounting.ckpt_pattern)
        # In test mode, we might want to trigger sampling
        # For now, just validation loss.
        trainer.validate(
            model,
            dataloaders=test_loader,
        )
        
        # 直接切到 EMA 权重（采样用）
        ckpt = torch.load(cfg.accounting.checkpoint_path, map_location="cpu")
        assert "ema_state_dict" in ckpt, "checkpoint has no ema_state_dict"
        model.load_state_dict(ckpt["ema_state_dict"], strict=False)

        model.eval().to(device)
        print("Starting graph generation sampling...")
        with torch.no_grad(): # 在采样/推理时，使用 no_grad() 是好习惯，可以节省显存并加速
            x_prob, A_idx, node_mask = model.dynamics.sampling(
                num_samples=10000,
                max_nodes=9,
                T_steps=cfg.dynamics.sample_steps,
                K_X=4,
                K_A=4,
                batch_size_sampling=5000
            )
        # pdb.set_trace()
        # Convert to mols using your existing pipeline (same as your sanity-check)
        bs, n = x_prob.shape[0], x_prob.shape[1]
        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(A_idx).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(bs, n, n)
        edge_mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        edge_mask = edge_mask & diag_mask

        adj = A_idx * edge_mask
        samples_int = adj - 1
        samples_int[samples_int == -1] = 3
        adj_1hot = torch.nn.functional.one_hot(samples_int.long(), num_classes=4).permute(0, 3, 1, 2)

        # ============ xyd 2025/1/1 ============
        # x = torch.where(x_prob > 0.5, 1, 0)
        # x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)  # [B,N,4] -> [B,N,5] with padding class

        x_idx = x_prob.argmax(-1)
        x_1hot = torch.nn.functional.one_hot(x_idx, num_classes=4).float() * node_mask.unsqueeze(-1)
        pad = (1.0 - node_mask.float()).unsqueeze(-1)      # [B,N,1]
        x = torch.cat([x_1hot, pad], dim=-1)               # [B,N,10]  (最后一维是 padding)
        # ======================================

        gen_mols, num_mols_wo_correction = gen_mol(x, adj_1hot, cfg.dataset.name)
        gen_smiles = mols_to_smiles(gen_mols)
        gen_smiles = [smi for smi in gen_smiles if len(smi)]

        train_smiles, test_smiles = load_smiles(cfg.dataset.name)
        train_smiles, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles)

        with open(f"../dataset/{cfg.dataset.name.lower()}_test_nx.pkl", "rb") as f:
            test_graph_list = pickle.load(f)

        scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=device, n_jobs=8,
                                test=test_smiles, train=train_smiles)
        scores_nspdk = eval_graph_list(test_graph_list, mols_to_nx(gen_mols), methods=["nspdk"])["nspdk"]

        print(f"Num mols: {len(gen_mols)}")
        print(f"val w/o corr: {num_mols_wo_correction / max(1, len(gen_mols)):.4f}")
        for metric in ["FCD/Test", "Scaf/Test", "Frag/Test", "SNN/Test", f"unique@{len(gen_smiles)}", "Novelty", "valid"]:
            print(f"{metric:12s}: {scores[metric]:.4f}")
        print(f"NSPDK MMD   : {scores_nspdk:.5f}")

        with open("./logs/result.txt", "a") as f:
            f.write(f"Valid: {num_mols_wo_correction / max(1, len(gen_mols))}\nUnique@{len(gen_smiles)}: {scores[f'unique@{len(gen_smiles)}']}\nNovelty: {scores['Novelty']}\nFCD: {scores['FCD/Test']}\nNSPDK: {scores_nspdk}\nScaf/Test: {scores['Scaf/Test']}\nFrag/Test: {scores['Frag/Test']}\nSNN/Test: {scores['SNN/Test']}\n{cfg.exp_name}\nEpoch: {ckpt['epoch']}\t{cfg.accounting.ckpt_pattern}\n{cfg.dynamics.sample_steps}\n\n")

        pdb.set_trace()
        # Save generated molecule images
        # for k in val_mols.keys():
        #     try:
        #         img = Draw.MolsToGridImage(generated_mols[k][:min(100, len(generated_mols[k]))], molsPerRow=10)
        #         time = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        #         if args.log:
        #             wandb.log({f'Generated Molecules {k}': wandb.Image(img)})
        #         img.save(f'images/{name}_sample_all_{time}_{k}.png')
        #     except:
        #         continue

        #     if len(val_mols[k]) > 0:
        #         img = Draw.MolsToGridImage(val_mols[k][:min(100, len(val_mols[k]))], molsPerRow=10)
        #         if args.log:
        #             wandb.log({f'Valid Molecules {k}': wandb.Image(img)})
        #         img.save(f'images/{name}_sample_val_{time}_{k}.png')


if __name__ == "__main__":
    main()
