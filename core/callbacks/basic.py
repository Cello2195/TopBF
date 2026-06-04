from typing import Any, Optional, Union, Dict
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
from typing import Any, Dict, Optional, List
from torch_geometric.data import Data, Batch # 导入 Data 和 Batch 类型，用于类型检查

# from typing import Optional Any
# import pytorch_lightning as pl
# import torch
from pytorch_lightning.utilities import rank_zero_only
import pdb


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
    # gradient clupping for
    def __init__(self, Q=Queue(3000), maximum_allowed_norm=1e3) -> None:
        super().__init__()
        # self.max_norm = max_norm
        self.gradnorm_queue = Q
        self.maximum_allowed_norm = maximum_allowed_norm

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx) -> None:
        # zero graidents if they are not finite
        # if not all([torch.isfinite(t.grad).all() for t in pl_module.parameters()]):
        #     logging.warning("Gradients are not finite number")
        #     pl_module.zero_grad()
        #     return None
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

        if float(grad_norm) > max_grad_norm:
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
    # gradient clupping for
    def __init__(self) -> None:
        super().__init__()
        # self.max_norm = max_norm

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx) -> None:
        if not all([torch.isfinite(t.grad).all() for t in pl_module.parameters()]):
            for t in pl_module.parameters():
                if not torch.isfinite(t.grad).all():
                    print(t.name, t.grad)
            raise ValueError("gradient is not finite number")

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        self._start_time = time.time()

    def on_before_backward(
        self, trainer: Trainer, pl_module: LightningModule, loss: Tensor
    ) -> None:
        super().on_before_backward(trainer, pl_module, loss)
        _cur_time = time.time()
        logging.info(
            f"from trainbatch start to before backward took {_cur_time - self._start_time} secs"
        )

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_after_backward(trainer, pl_module)
        _cur_time = time.time()
        logging.info(
            f"from trainbatch start to after backward took {_cur_time - self._start_time} secs"
        )

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx) -> None:
        super().on_before_optimizer_step(trainer, pl_module, optimizer, opt_idx)
        _cur_time = time.time()
        logging.info(
            f"from trainbatch start to before optimizer step took {_cur_time - self._start_time} secs"
        )

    def on_before_zero_grad(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer
    ) -> None:
        super().on_before_zero_grad(trainer, pl_module, optimizer)
        _cur_time = time.time()
        logging.info(
            f"from trainbatch start to before zero grad took {_cur_time - self._start_time} secs"
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        _cur_time = time.time()
        logging.info(f"train batch took {_cur_time - self._start_time} secs")


class NormalizerCallback(Callback):
    def __init__(self, normalizer_dict: Dict[str, Any]) -> None:
        super().__init__()
        self.normalizer_dict = normalizer_dict

    def quantize(self, h: torch.Tensor) -> torch.Tensor:
        """
        将模型输出的连续 logits 转换为独热编码的离散类别。
        适用于节点特征 x 或边特征 edge_attr。
        """
        if h.dim() >= 2 and h.shape[-1] > 1: # 确保最后一维代表类别数量
            h = F.one_hot(torch.argmax(h, dim=-1), num_classes=h.shape[-1]).float()
        return h

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        """
        对于离散的独热编码特征，输入阶段通常不需要归一化。
        此处主要用于移除旧的、不相关的归一化代码。
        """
        # 你的 x 和 edge_attr 已经是 0/1 独热编码，
        # 所以这里的除法操作（如果 normalizer_dict['x'] 或 ['edge_attr'] 为 1.0）没有实际效果，
        # 或者可以完全移除这些行，因为它们不需要额外的“归一化”。
        # 关键是移除对 batch.pos 和 batch.charges 的访问。

        # 例如，如果你想确保它们是浮点类型：
        if hasattr(batch, 'x'):
            batch.x = batch.x.float()
        if hasattr(batch, 'edge_attr'):
            batch.edge_attr = batch.edge_attr.float()
        
        # 理论上，BFN 的输入是连续的，但如果你的数据是离散的，通常会在模型内部的某处进行转换。
        # 如果模型内部要求 x 和 edge_attr 在某个特定连续范围内（如 [0,1]），
        # 且你的独热编码不是严格的 [0,1]，则这里仍然可以应用一个乘法缩放。
        # 但对于 0/1 独热编码，通常不需要额外缩放。
        pass # 或者保留上面 float() 转换

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        在验证批次结束后，将模型输出的连续预测量化回离散形式。
        """
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        
        if isinstance(outputs, dict) and 'generated_data' in outputs:
            generated_graphs = outputs['generated_data']
            
            generated_graphs_list: List[Data] = []
            if isinstance(generated_graphs, Data):
                generated_graphs_list = [generated_graphs]
            elif isinstance(generated_graphs, List):
                # 检查列表中的元素是否为 Data 对象
                if all(isinstance(item, Data) for item in generated_graphs):
                    generated_graphs_list = generated_graphs
                if all(isinstance(item, Tensor) for item in generated_graphs):
                    generated_graphs_list = generated_graphs
                else:
                    print(f"Warning: 'generated_data' list contains non-Data objects. Skipping post-processing.")
                    return
            elif isinstance(generated_graphs, Batch):
                generated_graphs_list = generated_graphs.to_data_list()
            else:
                print(f"Warning: Unexpected type for 'generated_data' in validation outputs: {type(generated_graphs)}. Skipping post-processing.")
                return

            for m in generated_graphs_list:
                # 核心：量化节点特征 x
                if hasattr(m, 'x'): # 确保有 x 属性
                    # 如果模型输出的 m.x 是连续值，这里将其量化为独热编码
                    m.x = self.quantize(m.x) 

                # 核心：量化边特征 edge_attr（如果模型生成且需要量化）
                if hasattr(m, 'edge_attr'): # 确保有 edge_attr 属性
                    # 如果模型输出的 m.edge_attr 是连续值，这里将其量化
                    m.edge_attr = self.quantize(m.edge_attr) # 复用 quantize 或编写专用函数
        else:
            print(f"Warning: validation_step output format not as expected by NormalizerCallback. Outputs: {outputs}")

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        self.on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """测试阶段输出处理，逻辑与验证阶段类似。"""
        # 复用 on_validation_batch_end 的逻辑
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)



class RecoverCallback(Callback):
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
        super().setup(trainer, pl_module, stage)
        _ckpt_paths = sorted(
            [(f, os.path.getmtime(f)) for f in glob.glob(self.latest_ckpt)],
            key=lambda x: x[1],
            reverse=True,
        )
        logging.info(f"latest ckpt: {self.latest_ckpt} resume={self.resume}")
        logging.info(f"all ckpt paths: {_ckpt_paths}")

        if len(_ckpt_paths) > 0:
            ckpt_path = _ckpt_paths[0][0]
        else:
            ckpt_path = ""
        if os.path.exists(ckpt_path) and self.resume:
            logging.info(f"recover from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            pl_module.load_state_dict(checkpoint["state_dict"])
            # pl_module.load_from_checkpoint(self.latest_ckpt)
        elif not os.path.exists(ckpt_path) and self.resume:
            logging.warning(f"checkpoint {ckpt_path} not found, training from scratch")
            checkpoint = None
        else:
            checkpoint = None

        self.on_load_checkpoint(trainer, pl_module, checkpoint)

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
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
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if "loss" not in outputs:
            return
        if (
            outputs["loss"] > self.recover_trigger_loss
            or not outputs["loss"].isfinite()
        ):
            self.skip_step = True
            self.count_skip += 1
        if self.count_skip > self.skip_count_limit and self.skip_step:
            self.count_skip = self.skip_count_limit // 2
            logging.warning(
                f"loss too large or non-finite: {outputs}\n recovering from checkpoint: {self.latest_ckpt}"
            )
            self.recover_count += 1
            if self.recover_count > self.recover_count_limit > 0:
                for layer in pl_module.children():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
                logging.warning(
                    f"recover count {self.recover_count} > {self.recover_count_limit}, training from scratch"
                )
                self.recover_count_limit = 0
            elif self.recover_count > self.recover_count_limit:
                raise ValueError(
                    f"recover count {self.recover_count} > {self.recover_count_limit}, stop training"
                )
            _ckpt_paths = sorted(
                [(f, os.path.getmtime(f)) for f in glob.glob(self.latest_ckpt)],
                key=lambda x: x[1],
                reverse=True,
            )
            if len(_ckpt_paths) > 0:
                ckpt_path = _ckpt_paths[0][0]
            else:
                ckpt_path = ""
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path)
                pl_module.load_state_dict(checkpoint["state_dict"])
                # pl_module.load_from_checkpoint(ckpt_path)
            else:
                for layer in pl_module.children():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
                logging.warning(
                    f"checkpoint {ckpt_path} not found, training from scratch"
                )
                checkpoint = None
            self.on_load_checkpoint(trainer, pl_module, checkpoint)
        else:
            pass

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer, opt_idx: int
    ) -> None:
        super().on_before_optimizer_step(trainer, pl_module, optimizer, opt_idx)
        if self.skip_step:
            optimizer.zero_grad()
            self.skip_step = False
        else:
            pass


class EMACallback(pl.Callback):
    """Implements EMA (exponential moving average) to any kind of model.
    EMA weights will be used during validation and stored separately from original model weights.

    How to use EMA:
        - Sometimes, last EMA checkpoint isn't the best as EMA weights metrics can show long oscillations in time. See
          https://github.com/rwightman/pytorch-image-models/issues/102
        - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
          discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
          https://github.com/rwightman/pytorch-image-models/issues/224
        - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16

    Implementation detail:
        - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
        - When multi gpu, we broadcast ema weights and the original weights in order to only hold 1 copy in memory.
          This is specially relevant when storing EMA weights on CPU + pinned memory as pinned memory is a limited
          resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
          performance.
    """

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
        )  # perform ema on different device from the model
        self.ema_pin_memory = (
            pin_memory if torch.cuda.is_available() else False
        )  # Only works if CUDA is available
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False

    @staticmethod
    def get_state_dict(pl_module: pl.LightningModule):
        """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
        For example, in pl_module has metrics, you don't want to return their parameters.

        code:
            # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
            # like losses, metrics, etc.
            patterns_to_ignore = ("metrics1", "metrics2")
            return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
        """
        return pl_module.state_dict()

    @overrides
    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: pl.LightningModule
    ) -> None:
        # Only keep track of EMA weights in rank zero.
        if not self._ema_state_dict_ready and pl_module.global_rank == 0:
            self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
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
        # Update EMA weights
        with torch.no_grad():
            for key, value in self.get_state_dict(pl_module).items():
                ema_value = self.ema_state_dict[key]
                try:
                    ema_value.copy_(
                        self.decay * ema_value + (1.0 - self.decay) * value,
                        non_blocking=True,
                    )
                except:
                    import pdb; pdb.set_trace()

    @overrides
    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        self.original_state_dict = deepcopy(self.get_state_dict(pl_module))

        trainer.strategy.broadcast(self.ema_state_dict, 0)

        assert self.ema_state_dict.keys() == self.original_state_dict.keys(), (
            f"There are some keys missing in the ema static dictionary broadcasted. "
            f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
        )
        pl_module.load_state_dict(self.ema_state_dict, strict=False)

        if pl_module.global_rank > 0:
            # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.ema_state_dict = {}

    @overrides
    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        # Replace EMA weights with training weights
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
        # return {"ema_state_dict": self.ema_state_dict, "_ema_state_dict_ready": self._ema_state_dict_ready}

    # @overrides # xyd
    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> None:
        if checkpoint is None:
            self._ema_state_dict_ready = False
        else:
            self._ema_state_dict_ready = checkpoint["_ema_state_dict_ready"]
            self.ema_state_dict = checkpoint["ema_state_dict"]
