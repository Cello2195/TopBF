import torch
from torch.utils.data import DataLoader
from typing import List, Union, Literal, Dict, Any, Tuple, Optional
import pdb

import math
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

try:
    import psi4
except ModuleNotFoundError:
    psi4 = None

from core.utils.graph_utils import node_flags, node_flags_from_XA


# ----------------------------- target extraction & normalization -----------------------------

def extract_targets_from_batch(batch_y, targets: List[str], cols: List[str]) -> torch.Tensor:
    if isinstance(batch_y, dict):
        cols_t = []
        for name in targets:
            if name not in batch_y:
                raise KeyError(f"Target '{name}' not found in batch dict keys: {list(batch_y.keys())}")
            y_col = batch_y[name]
            if y_col.dim() == 1:
                y_col = y_col.unsqueeze(-1)
            cols_t.append(y_col)
        y = torch.cat(cols_t, dim=-1)
        return y.float()

    elif torch.is_tensor(batch_y):
        if cols is None or len(cols) == 0:
            raise ValueError("When batch[2] is a tensor, 'cols' (list of column names) must be provided.")
        name_to_idx = {name: i for i, name in enumerate(cols)}
        idxs = []
        for name in targets:
            if name not in name_to_idx:
                raise KeyError(f"Target '{name}' not found in provided 'cols' list.")
            idxs.append(name_to_idx[name])
        y = batch_y[:, idxs]
        return y.float()

    else:
        raise TypeError("Unsupported Y type in batch; expected dict or Tensor.")

@torch.no_grad()
def compute_norm_stats(
    loader: DataLoader,
    targets: List[str],
    cols: List[str],
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = 0
    sum_y = None
    sum_y2 = None

    for bi, batch in enumerate(loader):
        if len(batch) < 3:
            raise ValueError("Batch must include target labels at index 2. Please modify load_data(cfg).")
        y = extract_targets_from_batch(batch[2], targets, cols)  # [B,D]
        y = y.detach().cpu()

        if sum_y is None:
            D = y.shape[-1]
            sum_y = torch.zeros(D, dtype=torch.float64)
            sum_y2 = torch.zeros(D, dtype=torch.float64)

        sum_y += y.sum(dim=0).to(torch.float64)
        sum_y2 += (y**2).sum(dim=0).to(torch.float64)
        n += y.shape[0]

        if max_batches is not None and (bi + 1) >= max_batches:
            break

    mean = (sum_y / max(1, n)).to(torch.float32)
    var = (sum_y2 / max(1, n)).to(torch.float32) - mean**2
    std = torch.sqrt(var.clamp_min(1e-12))
    return mean, std

# ----------------------------- noisy builder (G_t) -----------------------------

def map_discrete_to_continuous(idx_tensor, num_classes):
    """
    Maps integer indices from [0, num_classes-1] to a continuous range [0, 1].
    Example: 0 -> 0.0, 1 -> 0.5, 2 -> 1.0 for num_classes=3
    """
    if num_classes <= 1: # Avoid division by zero if only one class
        return torch.zeros_like(idx_tensor).float()
    return (2 * idx_tensor.float() + 1) / num_classes - 1

@torch.no_grad()
def build_noisy_inputs_like_generator(
    X: torch.Tensor,      # [B,N,K_X]
    A: torch.Tensor,             # [B,N,N] or [B,N,N,1] or [B,N,N,K_A]
    K_X: int,
    K_A: int,
    sigma1_X: float,
    sigma1_A: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate the forward-noise process used in bfn_base.training_step to produce noisy (mu_X, mu_A) and masks.
    Returns: mu_X:[B,N,1], mu_A:[B,N,N,1], t:[B,1], node_mask:[B,N] (bool)
    """
    device = X.device
    B, N, _ = X.shape

    # node mask from adjacency (same as generator)
    # node_mask = node_flags_from_XA(X_onehot=X, A=A)  # [B,N]
    node_mask = node_flags(A).bool()
    edge_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)


    # X: one-hot -> indices -> continuous centers in [-1,1]*scale
    X = X.argmax(dim=-1)  # [B,N]
    X = map_discrete_to_continuous(X, K_X).unsqueeze(-1).float()  # [B,N,1]
    X = X * node_mask.unsqueeze(-1)

    # A: normalize to integer indices, then to continuous centers in [-1,1]*scale
    A = map_discrete_to_continuous(A, K_A).float()  # [B,N,N,1]
    A = A * edge_mask

    # Sample time
    t = torch.rand(B, 1, device=device).clamp_min(1e-5)  # [B,1]

    # Gammas (broadcast per-graph)
    sigma1_X_t = torch.tensor(float(sigma1_X), device=device)
    sigma1_A_t = torch.tensor(float(sigma1_A), device=device)
    gamma_X = 1.0 - sigma1_X_t**(2.0 * t)  # [B,1]
    gamma_A = 1.0 - sigma1_A_t**(2.0 * t)  # [B,1]

    # Add Gaussian noise with correct variance
    # pdb.set_trace()
    mu_X = gamma_X.unsqueeze(1) * X + torch.randn_like(X) * torch.sqrt(gamma_X.unsqueeze(1) * (1.0 - gamma_X.unsqueeze(1)))
    mu_A = gamma_A.unsqueeze(1) * A + torch.randn_like(A) * torch.sqrt(gamma_A.unsqueeze(1) * (1.0 - gamma_A.unsqueeze(1)))

    # clamp + mask (match generator)
    mu_X = mu_X.clamp(-10.0, 10.0)
    mu_A = mu_A.clamp(-10.0, 10.0)
    mu_X = torch.where(node_mask.unsqueeze(-1), mu_X, torch.zeros_like(mu_X))
    mu_A = torch.where(edge_mask, mu_A, torch.zeros_like(mu_A))

    return mu_X, mu_A, t, node_mask



def _broadcast_targets(
    targets: Union[float, List[float], np.ndarray],
    n_valid: int
) -> np.ndarray:
    """把标量/列表/数组广播成长度 n_valid 的 float64 向量。"""
    if np.isscalar(targets):
        return np.full(n_valid, float(targets), dtype=np.float64)
    arr = np.asarray(targets, dtype=np.float64)
    if arr.size == 1:
        return np.full(n_valid, float(arr.item()), dtype=np.float64)
    if arr.size != n_valid:
        raise ValueError(f"target length ({arr.size}) must equal number of valid molecules ({n_valid}) or be scalar.")
    return arr


def psi4_mae_for_mols(
    rdkit_mols: List[Chem.Mol],
    target_values: Union[
        float,
        List[float],
        np.ndarray,
        Dict[str, Union[float, List[float], np.ndarray]],
        Tuple[Union[float, List[float], np.ndarray], Union[float, List[float], np.ndarray]]
    ],
    property_type: Literal["mu", "homo", "both"] = "mu",
    level: str = "b3lyp/6-31G*",
    nthread: int = 4,
    memory: str = "5GB",
    random_seed: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    用 Psi4 对一批 RDKit 分子计算 μ(偶极矩, Debye) / HOMO (eV) / 或二者，并与目标计算 MAE。

    参数
    ----
    rdkit_mols : RDKit Mol 列表（例如 geobfn_train_v12.py 生成的分子）
    target_values :
        - property_type 为 "mu" 或 "homo"：可为标量或长度与有效分子数相同的列表/数组。
        - property_type 为 "both"：可为
            1) dict: {"mu": ..., "homo": ...}；或
            2) 二元组/列表: (targets_mu, targets_homo)。
    property_type : "mu" | "homo" | "both"
    level : Psi4 计算层级（默认 B3LYP/6-31G*）
    nthread : Psi4 线程数
    memory : Psi4 内存
    random_seed : ETKDGv3 随机种
    verbose : 是否打印中间信息

    返回
    ----
    - "mu"/"homo"：
      {
        "mae": float,
        "values": List[float],
        "n_valid": int,
        "n_total": int,
        "failed_idx": List[int],
      }
    - "both"：
      {
        "mae": float,          # 两种属性绝对误差合在一起的平均（等权）
        "mae_mu": float,
        "mae_homo": float,
        "values_mu": List[float],
        "values_homo": List[float],
        "n_valid": int,
        "n_total": int,
        "failed_idx": List[int],
      }
    """
    if psi4 is None:
        raise RuntimeError("Psi4 is not installed or not importable in this environment.")

    prop = property_type.lower()
    assert prop in ("mu", "homo", "both"), "property_type must be 'mu', 'homo', or 'both'."

    # ---- Psi4 全局设置 ----
    psi4.set_num_threads(nthread)
    psi4.set_memory(memory)
    psi4.core.set_output_file('psi4_output.dat', False)

    # ---- 构型准备 ----
    params = ETKDGv3()
    params.randomSeed = int(random_seed)

    values_mu: List[float] = []
    values_homo: List[float] = []
    failed_idx: List[int] = []

    for i, mol_in in tqdm(enumerate(rdkit_mols)):
        mol = Chem.Mol(mol_in)

        # 规范化 + 3D + MMFF
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            if verbose: print(f"[{i}] RDKit Sanitize 失败")
            failed_idx.append(i);  continue

        mol = Chem.AddHs(mol)
        try:
            EmbedMolecule(mol, params)
        except Exception:
            if verbose: print(f"[{i}] 3D Embed 失败")
            failed_idx.append(i);  continue

        try:
            _ = MMFFOptimizeMolecule(mol)
        except Exception:
            if verbose: print(f"[{i}] MMFF 优化失败")
            failed_idx.append(i);  continue

        conf = mol.GetConformer()

        # 电荷、旋多
        formal_charge = int(mol.GetProp("FormalCharge")) if mol.HasProp("FormalCharge") else Chem.GetFormalCharge(mol)
        if mol.HasProp("SpinMultiplicity"):
            spin_mult = int(mol.GetProp("SpinMultiplicity"))
        else:
            num_rad = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
            spin_mult = int(2 * (num_rad / 2.0) + 1)

        # 组装 Psi4 geometry
        geom = f"{formal_charge} {spin_mult}"
        for a in mol.GetAtoms():
            pos = conf.GetAtomPosition(a.GetIdx())
            geom += f"\n {a.GetSymbol()} {pos.x} {pos.y} {pos.z}"

        try:
            molecule = psi4.geometry(geom)
        except Exception:
            if verbose: print(f"[{i}] Psi4 geometry 解析失败")
            failed_idx.append(i);  continue

        try:
            energy, wfn = psi4.energy(level, molecule=molecule, return_wfn=True)
        except Exception as e:
            if verbose: print(f"[{i}] Psi4 计算失败: {type(e).__name__}")
            failed_idx.append(i);  continue

        # ---- 提取 μ / HOMO（同一次 SCF，几乎不增算）----
        # μ：a.u. → Debye
        try:
            dip = wfn.variable('SCF DIPOLE')  # (x,y,z) in a.u.
            mu_val = math.sqrt(dip[0]**2 + dip[1]**2 + dip[2]**2) * 2.5417464519
        except Exception:
            mu_val = float('nan')
        # HOMO：a.u. → eV
        try:
            LUMO_idx = wfn.nalpha()
            HOMO_idx = LUMO_idx - 1
            homo_val = wfn.epsilon_a_subset("AO", "ALL").np[HOMO_idx] * 27.211324570273
        except Exception:
            homo_val = float('nan')

        if prop in ("mu", "both") and not np.isnan(mu_val):
            values_mu.append(float(mu_val))
        elif prop == "mu" and np.isnan(mu_val):
            failed_idx.append(i);  continue

        if prop in ("homo", "both") and not np.isnan(homo_val):
            values_homo.append(float(homo_val))
        elif prop == "homo" and np.isnan(homo_val):
            failed_idx.append(i);  continue

        if prop == "both" and (np.isnan(mu_val) or np.isnan(homo_val)):
            failed_idx.append(i)
            if not np.isnan(mu_val) and values_mu: values_mu.pop()
            if not np.isnan(homo_val) and values_homo: values_homo.pop()
            continue

    n_total = len(rdkit_mols)

    if prop == "mu":
        n_valid = len(values_mu)
        if n_valid == 0:
            return {"mae": float("nan"), "values": [], "n_valid": 0, "n_total": n_total, "failed_idx": failed_idx}
        vals = np.asarray(values_mu, dtype=np.float64)
        targets = _broadcast_targets(target_values, n_valid)
        mae = float(np.mean(np.abs(vals - targets)))
        return {"mae": mae, "values": values_mu, "n_valid": n_valid, "n_total": n_total, "failed_idx": failed_idx}

    if prop == "homo":
        n_valid = len(values_homo)
        if n_valid == 0:
            return {"mae": float("nan"), "values": [], "n_valid": 0, "n_total": n_total, "failed_idx": failed_idx}
        vals = np.asarray(values_homo, dtype=np.float64)
        targets = _broadcast_targets(target_values, n_valid)
        mae = float(np.mean(np.abs(vals - targets)))
        return {"mae": mae, "values": values_homo, "n_valid": n_valid, "n_total": n_total, "failed_idx": failed_idx}

    # both
    n_valid = min(len(values_mu), len(values_homo))  # 实际上二者应相等
    if n_valid == 0:
        return {
            "mae": float("nan"),
            "mae_mu": float("nan"),
            "mae_homo": float("nan"),
            "values_mu": [],
            "values_homo": [],
            "n_valid": 0,
            "n_total": n_total,
            "failed_idx": failed_idx,
        }

    vals_mu = np.asarray(values_mu, dtype=np.float64)
    vals_homo = np.asarray(values_homo, dtype=np.float64)

    if isinstance(target_values, dict):
        if "mu" not in target_values or "homo" not in target_values:
            raise ValueError("When property_type='both', target_values must include keys 'mu' and 'homo'.")
        targ_mu = _broadcast_targets(target_values["mu"], n_valid)
        targ_homo = _broadcast_targets(target_values["homo"], n_valid)
    elif isinstance(target_values, (tuple, list)) and len(target_values) == 2:
        targ_mu = _broadcast_targets(target_values[0], n_valid)
        targ_homo = _broadcast_targets(target_values[1], n_valid)
    else:
        raise ValueError("When property_type='both', target_values must be dict {'mu':..., 'homo':...} "
                         "or a 2-tuple/list (targets_mu, targets_homo).")

    mae_mu = float(np.mean(np.abs(vals_mu - targ_mu)))
    mae_homo = float(np.mean(np.abs(vals_homo - targ_homo)))
    # 整体 MAE：把两种属性的绝对误差拼起来平均（等权重）
    mae = float(np.mean(np.concatenate([np.abs(vals_mu - targ_mu), np.abs(vals_homo - targ_homo)], axis=0)))

    return {
        "mae": mae,
        "mae_mu": mae_mu,
        "mae_homo": mae_homo,
        "values_mu": values_mu,
        "values_homo": values_homo,
        "n_valid": n_valid,
        "n_total": n_total,
        "failed_idx": failed_idx,
    }
