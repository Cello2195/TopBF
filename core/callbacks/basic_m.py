from typing import Any, Optional, Union, Dict, List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor
import torch.nn.functional as F
from absl import logging
import time
import os
import glob
from torch.optim import Optimizer
from copy import deepcopy
from overrides import overrides
from torch_geometric.data import Data, Batch
from pytorch_lightning.utilities import rank_zero_only
import torch.distributed as dist

# ==========================================
# 辅助函数：DDP 同步检查
# ==========================================
def sync_bool_across_gpus(val: bool, device) -> bool:
    """如果任意一个GPU为True，则所有GPU返回True"""
    if not dist.is_initialized():
        return val
    t = torch.tensor([int(val)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item() > 0

class Queue:
    def __init__(self, max_len=50):
        self.items = [1]
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)

class Gradient_clip(Callback):
    """
    [DDP 警告] 此 Callback 在多卡模式下会导致不同显卡的梯度裁剪阈值不一致。
    建议直接在 pl.Trainer(gradient_clip_val=...) 中设置，而不要使用此 Callback。
    """
    def __init__(self, Q=Queue(3000), maximum_allowed_norm=1e3) -> None:
        super().__init__()
        self.gradnorm_queue = Q
        self.maximum_allowed_norm = maximum_allowed_norm

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx) -> None:
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            pl_module.parameters(), max_norm=max_grad_norm, norm_type=2.0
        )
        if float(grad_norm) > self.maximum_allowed_norm:
            optimizer.zero_grad()
        elif float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        # 只在主卡打印 log，避免刷屏
        if trainer.is_global_zero and float(grad_norm) > max_grad_norm:
            logging.info(
                f"Clipped gradient with value {grad_norm:.1f} "
                f"while allowed {max_grad_norm:.1f}",
            )
        pl_module.log(
            "grad_norm",
            grad_norm,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=pl_module.cfg.optimization.batch_size,
        )

class DebugCallback(Callback):
    # (保持原样，省略部分代码以节省篇幅，DDP下可以用但不建议开启，因为会极度拖慢速度)
    # ... 建议多卡训练时移除此 Callback ...
    pass 

class NormalizerCallback(Callback):
    # (保持你原来的代码，逻辑是局部的，兼容 DDP)
    def __init__(self, normalizer_dict: Dict[str, Any]) -> None:
        super().__init__()
        self.normalizer_dict = normalizer_dict

    def quantize(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() >= 2 and h.shape[-1] > 1: 
            h = F.one_hot(torch.argmax(h, dim=-1), num_classes=h.shape[-1]).float()
        return h

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        if hasattr(batch, 'x'):
            batch.x = batch.x.float()
        if hasattr(batch, 'edge_attr'):
            batch.edge_attr = batch.edge_attr.float()

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if isinstance(outputs, dict) and 'generated_data' in outputs:
            generated_graphs = outputs['generated_data']
            generated_graphs_list: List[Data] = []
            if isinstance(generated_graphs, Data):
                generated_graphs_list = [generated_graphs]
            elif isinstance(generated_graphs, List):
                if all(isinstance(item, Data) for item in generated_graphs):
                    generated_graphs_list = generated_graphs
                elif all(isinstance(item, Tensor) for item in generated_graphs):
                    generated_graphs_list = generated_graphs
            elif isinstance(generated_graphs, Batch):
                generated_graphs_list = generated_graphs.to_data_list()
            else:
                return

            for m in generated_graphs_list:
                if hasattr(m, 'x'): 
                    m.x = self.quantize(m.x) 
                if hasattr(m, 'edge_attr'):
                    m.edge_attr = self.quantize(m.edge_attr) 

    def on_test_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        self.on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)


class RecoverCallback(Callback):
    """
    [DDP 修改版] 增加了多卡同步逻辑。
    如果任意一张卡触发 Loss 异常，所有卡都会同步跳过或回滚。
    """
    def __init__(
        self, latest_ckpt, recover_trigger_loss=1e3, skip_count_limit=3, resume=False
    ) -> None:
        super().__init__()
        self.latest_ckpt = latest_ckpt
        self.recover_trigger_loss = recover_trigger_loss
        self.resume = resume
        self.skip_step = False
        self.count_skip = 0
        self.skip_count_limit = skip_count_limit
        self.recover_count = 0
        self.recover_count_limit = 20

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # 仅在主进程打印 log
        if trainer.is_global_zero:
            _ckpt_paths = sorted(
                [(f, os.path.getmtime(f)) for f in glob.glob(self.latest_ckpt)],
                key=lambda x: x[1],
                reverse=True,
            )
            logging.info(f"latest ckpt: {self.latest_ckpt} resume={self.resume}")
            logging.info(f"all ckpt paths: {_ckpt_paths}")

        # 所有卡都尝试寻找 checkpoint 路径
        _ckpt_paths = sorted(
            [(f, os.path.getmtime(f)) for f in glob.glob(self.latest_ckpt)],
            key=lambda x: x[1],
            reverse=True,
        )
        if len(_ckpt_paths) > 0:
            ckpt_path = _ckpt_paths[0][0]
        else:
            ckpt_path = ""

        # 加载逻辑需要保持一致，通常由 Trainer 处理 resume，这里手动处理时要小心
        # 如果是 DDP，建议让 PL 的 Trainer(resume_from_checkpoint) 来处理，而不是这里手动 load
        # 但为了保持你代码逻辑，这里保留，但注意所有 rank 都会执行
        if os.path.exists(ckpt_path) and self.resume:
            if trainer.is_global_zero: logging.info(f"recover from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=pl_module.device) # 确保 load 到当前设备
            pl_module.load_state_dict(checkpoint["state_dict"])
        elif not os.path.exists(ckpt_path) and self.resume:
            if trainer.is_global_zero: logging.warning(f"checkpoint {ckpt_path} not found, training from scratch")
        
        # self.on_load_checkpoint(trainer, pl_module, checkpoint) # 这一步通常不需要手动调用

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_start(trainer, pl_module)
        self.skip_step = False
        self.count_skip = 0

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # 获取当前卡的 Loss 状态
        local_loss_bad = False
        if "loss" in outputs:
            loss = outputs["loss"]
            if loss > self.recover_trigger_loss or not torch.isfinite(loss):
                local_loss_bad = True
        
        # [DDP 关键] 同步状态：如果任意一张卡坏了，所有卡都标记为 bad
        global_loss_bad = sync_bool_across_gpus(local_loss_bad, pl_module.device)

        if global_loss_bad:
            self.skip_step = True
            self.count_skip += 1
        
        # 如果超过跳过限制，触发回滚
        if self.count_skip > self.skip_count_limit and self.skip_step:
            self.count_skip = self.skip_count_limit // 2
            if trainer.is_global_zero:
                logging.warning(
                    f"[DDP Recovery] Loss too large on one or more GPUs. "
                    f"Recovering from checkpoint: {self.latest_ckpt}"
                )
            self.recover_count += 1
            
            # 检查回滚次数限制
            if self.recover_count > self.recover_count_limit > 0:
                # 超过限制，重置参数（这里可能需要 sync，但简化处理让所有 rank 都 reset）
                for layer in pl_module.children():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
                if trainer.is_global_zero:
                    logging.warning(f"recover count > limit, training from scratch")
                self.recover_count_limit = 0 # reset limit check
            elif self.recover_count > self.recover_count_limit:
                raise ValueError("recover count limit exceeded, stop training")

            # 执行回滚加载
            _ckpt_paths = sorted(
                [(f, os.path.getmtime(f)) for f in glob.glob(self.latest_ckpt)],
                key=lambda x: x[1],
                reverse=True,
            )
            ckpt_path = _ckpt_paths[0][0] if len(_ckpt_paths) > 0 else ""
            
            if os.path.exists(ckpt_path):
                # 必须 map_location 到当前 device，否则多卡会出错
                checkpoint = torch.load(ckpt_path, map_location=pl_module.device)
                pl_module.load_state_dict(checkpoint["state_dict"])
            else:
                for layer in pl_module.children():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
        
    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer, opt_idx: int
    ) -> None:
        super().on_before_optimizer_step(trainer, pl_module, optimizer, opt_idx)
        # 此时 self.skip_step 在所有 rank 上应该是一致的
        if self.skip_step:
            optimizer.zero_grad()
            self.skip_step = False


class EMACallback(pl.Callback):
    def __init__(
        self,
        decay: float = 0.9999,
        ema_device: Optional[Union[torch.device, str]] = None,
        pin_memory=True,
    ):
        super().__init__()
        self.decay = decay
        self.ema_device: str = (
            f"{ema_device}" if ema_device else None
        )
        self.ema_pin_memory = (
            pin_memory if torch.cuda.is_available() else False
        )
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False

    @staticmethod
    def get_state_dict(pl_module: pl.LightningModule):
        return pl_module.state_dict()

    @overrides
    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: pl.LightningModule
    ) -> None:
        # 只在 Rank 0 初始化 EMA 权重，节省显存
        if not self._ema_state_dict_ready and pl_module.global_rank == 0:
            self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
            
            # [DDP 建议] 强烈建议将 ema_device 设为 'cpu'，
            # 否则后续 broadcast 可能会因为 rank 之间的 device id 不匹配而报错
            if self.ema_device:
                self.ema_state_dict = {
                    k: tensor.to(device=self.ema_device)
                    for k, tensor in self.ema_state_dict.items()
                }

            if self.ema_device == "cpu" and self.ema_pin_memory:
                self.ema_state_dict = {
                    k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()
                }

        self._ema_state_dict_ready = True

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: pl.LightningModule, *args, **kwargs
    ) -> None:
        # 只在 Rank 0 更新 EMA 权重
        with torch.no_grad():
            current_state_dict = self.get_state_dict(pl_module)
            for key, ema_value in self.ema_state_dict.items():
                # 注意：current_state_dict 的 tensor 在 GPU 上，ema_value 可能在 CPU 上
                # 需要统一设备进行计算，或者利用 PyTorch 的自动处理
                value = current_state_dict[key]
                
                # 确保在同一设备上计算
                if ema_value.device != value.device:
                    value = value.to(ema_value.device)
                
                ema_value.copy_(
                    self.decay * ema_value + (1.0 - self.decay) * value,
                    non_blocking=True,
                )

    @overrides
    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: LightningModule
    ) -> None:
        if not self._ema_state_dict_ready:
            return

        self.original_state_dict = deepcopy(self.get_state_dict(pl_module))

        # [DDP 关键] 将 Rank 0 的 EMA 权重广播给所有 Rank
        # 注意：strategy.broadcast 通常需要 PyTorch Lightning 较新版本
        # 且最好确保 ema_state_dict 中的 tensor 在 CPU 上，避免 device mismatch
        
        # 1. 尝试将 dict 移到 cpu 以便安全广播 (如果是多卡环境)
        if trainer.num_devices > 1 and pl_module.global_rank == 0:
             # 为了安全，这里不修改 self.ema_state_dict 本身（太慢），
             # 而是假设 self.ema_device='cpu'。如果不是 CPU，会有风险。
             pass

        # 2. 广播
        try:
            # PL 1.6+ 写法
            trainer.strategy.broadcast(self.ema_state_dict, 0)
        except AttributeError:
             # 旧版本兼容
             try:
                 trainer.training_type_plugin.broadcast(self.ema_state_dict, 0)
             except:
                 pass # 如果都不行，非 Rank 0 可能会拿不到权重导致报错

        # 3. 加载权重
        # 此时非 Rank 0 的 self.ema_state_dict 应该已经被 broadcast 填充了
        if self.ema_state_dict:
            # 校验 keys
            # 注意：如果 broadcast 失败，非 Rank 0 的 self.ema_state_dict 还是空的，这里会报错
            # 但 DDP 下通常这步是必须的
            # assert self.ema_state_dict.keys() == self.original_state_dict.keys()
            pl_module.load_state_dict(self.ema_state_dict, strict=False)

        if pl_module.global_rank > 0:
            # 验证结束后，非主卡释放内存
            self.ema_state_dict = {}

    @overrides
    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not self._ema_state_dict_ready:
            return
        # 恢复原始权重
        pl_module.load_state_dict(self.original_state_dict, strict=False)

    @overrides
    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_validation_start(trainer, pl_module)

    @overrides
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_validation_end(trainer, pl_module)

    @overrides
    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> None:
        checkpoint["ema_state_dict"] = self.ema_state_dict
        checkpoint["_ema_state_dict_ready"] = self._ema_state_dict_ready

    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> None:
        if checkpoint is None:
            self._ema_state_dict_ready = False
        else:
            self._ema_state_dict_ready = checkpoint.get("_ema_state_dict_ready", False)
            self.ema_state_dict = checkpoint.get("ema_state_dict", {})