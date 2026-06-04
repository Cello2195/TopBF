import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter_mean
import torch.distributions as dist
import numpy as np
from absl import logging
import pdb
from typing import Optional
from tqdm import tqdm

# Import the Graph Transformer module
from core.module.transformer_new import GraphTransformer_Mol
from core.utils.utils import *
from core.utils.graph_utils import node_flags, mask_x, mask_adjs

# --- Helper functions ---
class bfnBase(nn.Module):
    # This is a general method which could be used for implementing vector fields in CNF or other models
    def __init__(self, *args, **kwargs):
        super(bfnBase, self).__init__(*args, **kwargs)

# Utility function to map discrete categories to continuous values
def map_discrete_to_continuous(idx_tensor, num_classes):
    """
    Maps integer indices from [0, num_classes-1] to a continuous range [0, 1].
    Example: 0 -> 0.0, 1 -> 0.5, 2 -> 1.0 for num_classes=3
    """
    if num_classes <= 1: # Avoid division by zero if only one class
        return torch.zeros_like(idx_tensor).float()
    return (2 * idx_tensor.float() + 1) / num_classes - 1

# Utility function to map continuous values back to discrete indices (for final sampling)
def map_continuous_to_discrete(continuous_tensor, num_classes):
    """
    Maps continuous values from [0, 1] (or a similar range) back to integer indices.
    Rounds to nearest integer and clamps.
    """
    if num_classes <= 1:
        return torch.zeros_like(continuous_tensor).long()

    # Scale back to [0, num_classes-1] then round
    scaled_tensor = ((continuous_tensor + 1) * num_classes - 1) / 2
    return torch.round(scaled_tensor).long().clamp(0, num_classes - 1)


class bfn4GraphTransformer(nn.Module):
    def __init__(
        self,
        # in_node_nf,      # Input node feature dimension (1 for continuous scalar)
        hidden_nf,       # Hidden dimension for transformer layers (dx, de, dy)
        n_layers,        # Number of transformer layers
        n_head,          # Number of attention heads
        dim_ffX,         # Feedforward dimension for node features
        dim_ffE,         # Feedforward dimension for edge features
        dim_ffy,
        # dropout,         # Dropout rate
        max_nodes,
        # device,
        n_node_histogram,
        t_min=1e-5,      # Minimum time for stability
        sigma1_X: float = 0.15, # Corresponds to sigma_h in paper, for charges/bonds
        sigma1_A: float = 0.15,  # For coordinates, not directly used for discrete, but kept for consistency
        bins_X: int = 4, # For continuous feature discretization (e.g., in evaluation), though not in core BFN
        bins_A: int = 4,
        **kwargs # Pass other args to base
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.t_min = t_min
        self.n_node_histogram = n_node_histogram
        # self._device = device

        # k_c_X, self.k_l_X, self.k_r_X = self.get_k_params(bins_X)
        # self.K_c_X = torch.tensor(k_c_X).to(self._device)
        # k_c_A, self.k_l_A, self.k_r_A = self.get_k_params(bins_A)
        # self.K_c_A = torch.tensor(k_c_A).to(self._device)

        # self.sigma1_X = torch.tensor(sigma1_X, dtype=torch.float32)
        # self.sigma1_A = torch.tensor(sigma1_A, dtype=torch.float32)

        k_c_X, k_l_X, k_r_X = self.get_k_params(bins_X)
        k_c_A, k_l_A, k_r_A = self.get_k_params(bins_A)

        # 【关键修改】使用 register_buffer
        self.register_buffer('K_c_X', torch.tensor(k_c_X))
        self.register_buffer('k_l_X', torch.tensor(k_l_X))
        self.register_buffer('k_r_X', torch.tensor(k_r_X))

        self.register_buffer('K_c_A', torch.tensor(k_c_A))
        self.register_buffer('k_l_A', torch.tensor(k_l_A))
        self.register_buffer('k_r_A', torch.tensor(k_r_A))

        self.register_buffer('sigma1_X', torch.tensor(sigma1_X, dtype=torch.float32))
        self.register_buffer('sigma1_A', torch.tensor(sigma1_A, dtype=torch.float32))

        self.bins_X = bins_X
        self.bins_A = bins_A

        # Probability stabilisation (align with pure VBFN)
        self.eps_prob = float(kwargs.get('eps_prob', 1e-9))
        # Whether to mask diagonal edges during training/sampling
        self.mask_diag_edges = bool(kwargs.get('mask_diag_edges', True))

        # ---------------- QW / Sinkhorn-OT regularizer (training only; optional) ----------------
        # Adds an OT-style structural regularizer between predicted node-type distributions and GT labels
        # using graph shortest-path distance as the ground cost. This does NOT affect sampling.
        self.qw_enable   = bool(kwargs.get('use_qw_loss', False))
        self.qw_weight   = float(kwargs.get('qw_weight', 0.0))
        self.qw_eps      = float(kwargs.get('qw_eps', 0.2))
        self.qw_iters    = int(kwargs.get('qw_iters', 30))
        self.qw_on_edges = bool(kwargs.get('qw_on_edges', False))


        # pdb.set_trace()
        self.K_X = bins_X # Number of atomic types
        self.K_A = bins_A # Number of bond types

        start, end = -2, 2
        in_dim_X = bins_X
        in_dim_A = bins_A
        # self.width_X = (end - start) / self.K_X
        # self.width_A = (end - start) / self.K_A
        self.width_X = (end - start) / in_dim_X
        self.width_A = (end - start) / in_dim_A
        # self.centers_X = torch.linspace(start, end, in_dim_X, device=self._device)  # [feature_num]
        # self.centers_A = torch.linspace(start, end, in_dim_A, device=self._device)  # [feature_num]
        # 创建时先放在CPU上，不需要指定device
        centers_X = torch.linspace(start, end, in_dim_X)
        centers_A = torch.linspace(start, end, in_dim_A)
        self.register_buffer('centers_X', centers_X)
        self.register_buffer('centers_A', centers_A)

        # Define dimensions for GraphTransformer
        # input_dims = {'X': self.K_X, 'E': self.K_A, 'y': 1} # X and E are continuous scalars, y is time embedding
        input_dims = {'X': 4, 'E': 2, 'y': 1}
        hidden_mlp_dims = {'X': hidden_nf * 4, 'E': hidden_nf * 4, 'y': hidden_nf * 2} # Example, adjust as needed
        hidden_dims = {'dx': hidden_nf * 4, 'de': hidden_nf * 2, 'dy': hidden_nf,
                       'n_head': n_head, 'dim_ffX': dim_ffX, 'dim_ffE': dim_ffE, 'dim_ffy': dim_ffy}
        # output_dims = {'X': self.K_X, 'E': self.K_A, 'y': 1} # Output logits for X and E, y remains hidden_nf
        output_dims = {'X': 4, 'E': 2, 'y': 0} # Output logits for X and E, y remains hidden_nf

        self.graph_transformer = GraphTransformer_Mol(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            scale=1.0
        )

        self.mu_head_X = nn.Linear(output_dims['X'], 1)    # 专门负责mu的阀门
        self.sigma_head_X = nn.Linear(output_dims['X'], 1) # 专门负责sigma的阀门
        self.mu_head_A = nn.Linear(output_dims['E'], 1)    # 专门负责mu的阀门
        self.sigma_head_A = nn.Linear(output_dims['E'], 1) # 专门负责sigma的阀门

        # self.time_embed = nn.Sequential(
        #     nn.Linear(1, hidden_nf),
        #     nn.SiLU(),
        #     nn.Linear(hidden_nf, hidden_nf)
        # )

    def get_k_params(self, bins):
        """
        function to get the k parameters for the discretised variable
        """
        # k = torch.ones_like(mu)
        # ones_ = torch.ones((mu.size()[1:])).cuda()
        # ones_ = ones_.unsqueeze(0)
        list_c = []
        list_l = []
        list_r = []
        for k in range(bins):
            # k = torch.cat([k,torch.ones_like(mu)*(i+1)],dim=1
            k_c = (2 * k + 1) / bins - 1
            k_l = k_c - 1 / bins
            k_r = k_c + 1 / bins
            list_c.append(k_c)
            list_l.append(k_l)
            list_r.append(k_r)
        # k_c = torch.cat(list_c,dim=0)
        # k_l = torch.cat(list_l,dim=0)
        # k_r = torch.cat(list_r,dim=0)

        return list_c, list_l, list_r
    
    def gaussian_basis(self, x, centers, width):
        """x: [batch_size, ...]"""
        # x = torch.unsqueeze(x, dim=-1)  # [batch_size, ..., 1]
        out = (x - centers) / width
        ret = torch.exp(-0.5 * out**2)

        return F.normalize(ret, dim=-1, p=1) * 2 - 1  # [batch_size, ..., feature_num]
    
    def discretised_cdf(self, mu, sigma, x):
        """
        cdf function for the discretised variable
        """
        # print("msx",mu,sigma,x)
        # in this case we use the discretised cdf for the discretised output function
        # 只有当 x 的维度比 mu 多（多出类别维度）时，才对 mu 和 sigma 进行 unsqueeze
        if x.dim() > mu.dim():
            mu = mu.unsqueeze(-2)
            sigma = sigma.unsqueeze(-2)
        # pdb.set_trace()
        # print(sigma.min(),sigma.max())
        # print(mu.min(),mu.max())

        f_ = 0.5 * (1 + torch.erf((x - mu) / ((sigma) * np.sqrt(2))))
        flag_upper = torch.ge(x, 1)
        flag_lower = torch.le(x, -1)
        f_ = torch.where(flag_upper, torch.ones_like(f_), f_)
        f_ = torch.where(flag_lower, torch.zeros_like(f_), f_)
        # if not torch.all(f_.isfinite()):
        #     print("f_", f_.min(), f_.max())
        #     print("mu", mu.min(), mu.max())
        #     print("sigma", sigma.min(), sigma.max())
        #     print("x", x.min(), x.max())
        #     print("flag_upper", flag_upper.min(), flag_upper.max())
        #     print("flag_lower", flag_lower.min(), flag_lower.max())
        #     raise ValueError("f_ is not finite")
        return f_
    
    @staticmethod
    def _masked_mean_per_sample(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        value: [B, ...]
        mask : same shape (bool)
        return: [B] per-sample mean over masked entries
        """
        v = torch.where(mask, value, torch.zeros_like(value))
        denom = mask.float().sum(dim=tuple(range(1, mask.dim()))).clamp_min(1.0).to(v.dtype)
        return v.sum(dim=tuple(range(1, v.dim()))) / denom

    def ctime4discreteised_loss(self, t, sigma1, x_pred, x, mask=None):
        # pdb.set_trace()
        if mask is None:
            loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).mean(dim=1)
        else:
            # print("loss with mask")
            # =================== xyd 2025/12/15 ==================
            # loss = ((x_pred - x) * mask.unsqueeze(-1)).view(x.shape[0], -1).pow(2).sum(dim=-1) / mask.view(x.shape[0], -1).sum(dim=-1).clamp_min(1.0)
            loss = ((x_pred - x) * mask.unsqueeze(-1)).view(x.shape[0], -1).pow(2).mean(dim=-1)
            # =================== ============== ==================

        return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))


    # ---------------- QW / Sinkhorn-OT regularizer helpers ----------------
    def _graph_shortest_path_cost(self, A_idx: torch.Tensor, node_mask: torch.Tensor, large: float = 1e3) -> torch.Tensor:
        """Batched Floyd–Warshall on small graphs (QM9 max_nodes is small).

        Returns: D in R^{B×N×N}, where D[b,i,j] is shortest-path length within the GT graph.
        """
        if A_idx.dtype != torch.long:
            A_idx = A_idx.long()
        B, N = A_idx.shape[:2]
        device = A_idx.device
        dist = torch.full((B, N, N), float(large), device=device, dtype=torch.float32)
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        dist[eye] = 0.0

        # adjacency: any bond type > 0
        adj = (A_idx > 0)
        adj = adj | adj.transpose(1, 2)
        adj = adj & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        dist[adj] = 1.0

        # Floyd–Warshall
        for k in range(N):
            dist = torch.minimum(dist, dist[:, :, k:k+1] + dist[:, k:k+1, :])
        return dist

    def _sinkhorn_ot_cost(self, a: torch.Tensor, b: torch.Tensor, C: torch.Tensor, eps: float, iters: int) -> torch.Tensor:
        """Entropic OT (Sinkhorn) cost in log-domain.

        a, b: [B,N] distributions; C: [B,N,N] ground cost.
        Returns: cost per sample [B].
        """
        eps = float(max(eps, 1e-6))
        logK = -C / eps
        loga = torch.log(a.clamp_min(self.eps_prob))
        logb = torch.log(b.clamp_min(self.eps_prob))
        logu = torch.zeros_like(loga)
        logv = torch.zeros_like(logb)
        for _ in range(int(max(iters, 1))):
            logu = loga - torch.logsumexp(logK + logv.unsqueeze(1), dim=2)
            logv = logb - torch.logsumexp(logK + logu.unsqueeze(2), dim=1)
        logP = logK + logu.unsqueeze(2) + logv.unsqueeze(1)
        P = torch.exp(logP)
        return (P * C).sum(dim=(1, 2))

    def _qw_loss_nodes(self, p_X: torch.Tensor, X_idx: torch.Tensor, A_idx: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Sinkhorn-OT loss between predicted and GT node-type distributions over nodes.

        p_X: [B,N,K], X_idx: [B,N], A_idx: [B,N,N] (int), node_mask: [B,N] (bool)
        Returns: [B] (averaged over present classes)
        """
        B, N, K = p_X.shape
        C = self._graph_shortest_path_cost(A_idx, node_mask)  # [B,N,N]
        y = F.one_hot(X_idx.clamp(min=0), num_classes=K).to(p_X.dtype)
        nm = node_mask.to(p_X.dtype)

        nm_sum = nm.sum(dim=1, keepdim=True)  # [B,1]
        valid_nodes = (nm_sum.squeeze(1) > 0).to(p_X.dtype)   # [B]

        costs = []
        present = []
        for k in range(K):
            a_raw = p_X[:, :, k] * nm
            b_raw = y[:, :, k] * nm

            a_sum = a_raw.sum(dim=1, keepdim=True)  # [B,1]
            # If prediction for this class is exactly zero, fall back to uniform over valid nodes (avoid degenerate OT)
            a = torch.where(
                a_sum > 0,
                a_raw / a_sum.clamp_min(self.eps_prob),
                nm / nm_sum.clamp_min(self.eps_prob),
            )

            b_sum = b_raw.sum(dim=1, keepdim=True)
            is_present = (b_sum.squeeze(1) > 0).to(p_X.dtype)
            # If class absent in GT, do not push it: set b=a (cost 0) and mask it out via `present`
            b = torch.where(b_sum > 0, b_raw / b_sum.clamp_min(self.eps_prob), a)

            costs.append(self._sinkhorn_ot_cost(a, b, C, self.qw_eps, self.qw_iters))
            present.append(is_present)

        costs = torch.stack(costs, dim=1)      # [B,K]
        present = torch.stack(present, dim=1)  # [B,K]
        present = present * valid_nodes.unsqueeze(1)
        denom = present.sum(dim=1).clamp_min(1.0)
        return (costs * present).sum(dim=1) / denom

    def _qw_loss_edges(self, p_A: torch.Tensor, A_idx: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Optional Sinkhorn-OT loss over bond-type distributions on edges (upper triangle).

        p_A: [B,N,N,K], A_idx: [B,N,N] (int), node_mask: [B,N] (bool)
        Returns: [B]
        """
        if p_A.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            p_A = p_A.float()
        if A_idx.dtype != torch.long:
            A_idx = A_idx.long()

        B, N, _, K = p_A.shape
        device = p_A.device

        u, v = torch.triu_indices(N, N, offset=1, device=device)
        M = u.numel()
        if M == 0:
            return torch.zeros((B,), device=device, dtype=p_A.dtype)

        # edge mask over upper-triangle edges
        em = (node_mask[:, u] & node_mask[:, v]).to(p_A.dtype)  # [B,M]
        em_sum = em.sum(dim=1, keepdim=True)                    # [B,1]
        valid_edges = (em_sum.squeeze(1) > 0).to(p_A.dtype)      # [B]

        A_e = A_idx[:, u, v].long().clamp(min=0, max=K - 1)      # [B,M]
        p_e = p_A[:, u, v, :]                                    # [B,M,K]

        # cost between edges via min endpoint-to-endpoint shortest-path distance (cheap line-graph proxy)
        D = self._graph_shortest_path_cost(A_idx, node_mask)      # [B,N,N]
        duu = D[:, u[:, None], u[None, :]]
        duv = D[:, u[:, None], v[None, :]]
        dvu = D[:, v[:, None], u[None, :]]
        dvv = D[:, v[:, None], v[None, :]]
        C = torch.minimum(torch.minimum(duu, duv), torch.minimum(dvu, dvv))  # [B,M,M]

        costs = []
        present = []
        # Usually ignore type-0 (no-bond) to avoid dominating the loss.
        for k in range(1, K):
            a_raw = p_e[:, :, k] * em
            b_raw = (A_e == k).to(p_A.dtype) * em

            a_sum = a_raw.sum(dim=1, keepdim=True)
            # IMPORTANT: domain is EDGES, so the uniform fallback must be over `em`, not node-mask `nm`
            a = torch.where(
                a_sum > 0,
                a_raw / a_sum.clamp_min(self.eps_prob),
                em / em_sum.clamp_min(self.eps_prob),
            )

            b_sum = b_raw.sum(dim=1, keepdim=True)
            is_present = (b_sum.squeeze(1) > 0).to(p_A.dtype)
            b = torch.where(b_sum > 0, b_raw / b_sum.clamp_min(self.eps_prob), a)

            costs.append(self._sinkhorn_ot_cost(a, b, C, self.qw_eps, self.qw_iters))
            present.append(is_present)

        if len(costs) == 0:
            return torch.zeros((B,), device=device, dtype=p_A.dtype)

        costs = torch.stack(costs, dim=1)      # [B,K-1]
        present = torch.stack(present, dim=1)  # [B,K-1]
        present = present * valid_edges.unsqueeze(1)
        denom = present.sum(dim=1).clamp_min(1.0)
        return (costs * present).sum(dim=1) / denom

    def interdependency_modeling(
        self,
        mu_X_t: torch.Tensor,
        mu_A_t: torch.Tensor,
        t: torch.Tensor,
        gamma_X: torch.Tensor,
        gamma_A: torch.Tensor,
        # true_labels_X: torch.Tensor,
        # true_labels_A: torch.Tensor,
        node_mask: Optional[torch.Tensor]=None,
    ):
        mu_X_in = self.gaussian_basis(mu_X_t, self.centers_X, self.width_X) # X和A的width需要不一样吗？
        mu_A_in = mu_A_t
        # mu_A_in = self.gaussian_basis(mu_A_t, self.centers_A, self.width_A)
        # pdb.set_trace()
        # t = self.time_embed(t)

        X, E = self.graph_transformer(
            X=mu_X_in, # mu_X_t is already (B, N, 1)
            E=mu_A_in, # mu_A_t is already (B, N, N, 1)
            y=t,
            node_mask=node_mask
        )
        # pdb.set_trace()
        
        mu_X_eps = self.mu_head_X(X)
        sigma_X_eps = self.sigma_head_X(X) # 预测 log(sigma) 更稳定
        mu_A_eps = self.mu_head_A(E)
        sigma_A_eps = self.sigma_head_A(E)
        # pdb.set_trace()

        mu_X_eps = torch.clamp(mu_X_eps, min=-10, max=10)
        sigma_X_eps = torch.clamp(sigma_X_eps, min=-10, max=10)
        mu_A_eps = torch.clamp(mu_A_eps, min=-10, max=10)
        sigma_A_eps = torch.clamp(sigma_A_eps, min=-10, max=10)

        sigma_X_eps = torch.exp(sigma_X_eps)
        sigma_A_eps = torch.exp(sigma_A_eps)
        
        mu_X_x = mu_X_t / gamma_X.unsqueeze(1) - torch.sqrt((1 - gamma_X) / gamma_X).unsqueeze(1) * mu_X_eps
        sigma_X_x = torch.sqrt((1 - gamma_X) / gamma_X).unsqueeze(1) * sigma_X_eps
        mu_A_x = mu_A_t / gamma_A.unsqueeze(1) - torch.sqrt((1 - gamma_A) / gamma_A).unsqueeze(1) * mu_A_eps.squeeze()
        sigma_A_x = torch.sqrt((1 - gamma_A) / gamma_A).unsqueeze(1) * sigma_A_eps.squeeze()

        mu_X_x = torch.clamp(mu_X_x, min=-2, max=2)
        sigma_X_x = torch.clamp(sigma_X_x, min=1e-6, max=4)
        mu_A_x = torch.clamp(mu_A_x, min=-2, max=2)
        sigma_A_x = torch.clamp(sigma_A_x, min=1e-6, max=4)

        # pdb.set_trace()

        # k_r_X = torch.tensor(self.k_r_X).to(self._device).unsqueeze(-1).unsqueeze(0).unsqueeze(1)
        # k_l_X = torch.tensor(self.k_l_X).to(self._device).unsqueeze(-1).unsqueeze(0).unsqueeze(1)
        # p_o = self.discretised_cdf(mu_X_x, sigma_X_x, k_r_X) - self.discretised_cdf(mu_X_x, sigma_X_x, k_l_X)
        # k_hat_X = p_o

        # k_r_A = torch.tensor(self.k_r_A).to(self._device).unsqueeze(-1).unsqueeze(0).unsqueeze(1).unsqueeze(2)
        # k_l_A = torch.tensor(self.k_l_A).to(self._device).unsqueeze(-1).unsqueeze(0).unsqueeze(1).unsqueeze(2)
        # p_o = self.discretised_cdf(mu_A_x, sigma_A_x, k_r_A) - self.discretised_cdf(mu_A_x, sigma_A_x, k_l_A)
        # k_hat_A = p_o
        # pdb.set_trace()


        # X_hat_continuous = mu_X_t / gamma_X.unsqueeze(1) - torch.sqrt((1 - gamma_X) / gamma_X).unsqueeze(1) * eps_X_hat_continuous
        # A_hat_continuous = mu_A_t / gamma_A.unsqueeze(1).unsqueeze(2) - torch.sqrt((1 - gamma_A) / gamma_A).unsqueeze(1).unsqueeze(2) * eps_A_hat_continuous


        return mu_X_x, sigma_X_x, mu_A_x.unsqueeze(-1), sigma_A_x.unsqueeze(-1)

    def training_step(self, data: Batch):
        """TopBF training step aligned with pure VBFN:
        - compute discretised probs p_X/p_A
        - apply mask + clamp + renormalise
        - regress expected centers with masked MSE flow loss
        """
        X1, A_raw = data[0], data[1]
        device = X1.device
        B, N = A_raw.shape[:2]

        # masks
        node_mask = node_flags(A_raw).bool()
        edge_mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        if self.mask_diag_edges:
            diag = ~torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
            edge_mask = edge_mask & diag

        # discrete -> continuous centers in [-1,1]
        X_idx = X1.argmax(dim=-1)  # [B,N]
        X1c = map_discrete_to_continuous(X_idx, self.K_X).unsqueeze(-1).float()
        X1c = X1c * node_mask.unsqueeze(-1)

        # A_raw is expected to be integer adjacency in {0..K_A-1}
        A1c = map_discrete_to_continuous(A_raw, self.K_A).float()
        A1c = A1c * edge_mask.float()

        # time
        t = torch.rand(B, 1, device=device).clamp_min(self.t_min)

        # gamma schedule (kept consistent with the original TopBF parameterisation)
        gamma_X = 1 - torch.pow(self.sigma1_X, 2 * t)
        gamma_A = 1 - torch.pow(self.sigma1_A, 2 * t)

        # sample noisy messages mu
        mu_X = gamma_X.unsqueeze(1) * X1c + torch.randn_like(X1c) * torch.sqrt(gamma_X.unsqueeze(1) * (1 - gamma_X.unsqueeze(1)))
        mu_A = gamma_A.unsqueeze(1) * A1c + torch.randn_like(A1c) * torch.sqrt(gamma_A.unsqueeze(1) * (1 - gamma_A.unsqueeze(1)))

        mu_X = torch.clamp(mu_X, min=-10, max=10) * node_mask.unsqueeze(-1)
        mu_A = torch.clamp(mu_A, min=-10, max=10) * edge_mask.float()

        # predict clean distribution params
        mu_X_x, sigma_X_x, mu_A_x, sigma_A_x = self.interdependency_modeling(
            mu_X_t=mu_X,
            mu_A_t=mu_A,
            t=t,
            gamma_X=gamma_X,
            gamma_A=gamma_A,
            node_mask=node_mask,
        )

        # discretised probs
        p_X = (self.discretised_cdf(mu_X_x, sigma_X_x, self.k_r_X.view(1, 1, -1, 1)) -
               self.discretised_cdf(mu_X_x, sigma_X_x, self.k_l_X.view(1, 1, -1, 1))).squeeze(-1)  # [B,N,K]
        p_A = (self.discretised_cdf(mu_A_x, sigma_A_x, self.k_r_A.view(1, 1, 1, -1, 1)) -
               self.discretised_cdf(mu_A_x, sigma_A_x, self.k_l_A.view(1, 1, 1, -1, 1))).squeeze(-1)  # [B,N,N,K]

        # mask + clamp + renorm (align with VBFN)
        p_X = p_X * node_mask.unsqueeze(-1)
        p_A = p_A * edge_mask.unsqueeze(-1)

        p_X = p_X.clamp_min(self.eps_prob)
        p_X = p_X / p_X.sum(dim=-1, keepdim=True).clamp_min(self.eps_prob)

        p_A = p_A.clamp_min(self.eps_prob)
        p_A = p_A / p_A.sum(dim=-1, keepdim=True).clamp_min(self.eps_prob)

        # expected centers for regression
        kcx = self.K_c_X.to(p_X.dtype).view(1, 1, -1)
        kca = self.K_c_A.to(p_A.dtype).view(1, 1, 1, -1)
        x_pred_c = (p_X * kcx).sum(dim=-1, keepdim=True)  # [B,N,1]
        a_pred_c = (p_A * kca).sum(dim=-1, keepdim=True)  # [B,N,N,1]

        # masked flow loss
        X_loss = self.ctime4discreteised_loss(t=t, sigma1=self.sigma1_X, x_pred=x_pred_c, x=X1c, mask=node_mask)
        A_loss = self.ctime4discreteised_loss(t=t, sigma1=self.sigma1_A, x_pred=a_pred_c, x=A1c.unsqueeze(-1), mask=edge_mask)

        # optional QW/Sinkhorn-OT regularizer (training only)
        qw_reg = torch.zeros((B,), device=device, dtype=X_loss.dtype)
        if self.qw_enable and (self.qw_weight > 0):
            qw_reg = self._qw_loss_nodes(p_X, X_idx, A_raw, node_mask)
            if self.qw_on_edges:
                qw_reg_edges= self._qw_loss_edges(p_A, A_raw, node_mask)
                qw_reg = qw_reg + qw_reg_edges

        # print(
        #     (X_loss + A_loss).mean().item(),
        #     qw_reg.mean().item(),
        #     qw_reg_edges.mean().item() if qw_reg_edges is not None else None,
        #     (self.qw_weight * qw_reg.mean()).item()
        #     )

        # print((X_loss + A_loss).mean(), self.qw_weight, qw_reg.mean())
        total_loss = (X_loss + A_loss).mean() + self.qw_weight * qw_reg.mean()
        return total_loss


    @torch.no_grad()
    def sampling(self, num_samples, max_nodes, T_steps, K_X, K_A, batch_size_sampling=10):
        """Sampling aligned with pure VBFN (prob clamp+renorm, beta-diff alpha schedule, no per-step snapping)."""
        device = self.k_l_X.device
        num_batches = (num_samples + batch_size_sampling - 1) // batch_size_sampling

        all_x, all_adj, all_nm = [], [], []
        n_node_hist = np.array(self.n_node_histogram / np.sum(self.n_node_histogram))

        done = 0
        for bidx in tqdm(range(num_batches), desc="Sampling batches"):
            B = min(batch_size_sampling, num_samples - done)
            if B <= 0:
                break
            done += B

            # node counts / masks
            node_mask = torch.zeros(B, self.max_nodes, dtype=torch.bool, device=device)
            n_nodes_list = np.random.choice(n_node_hist.shape[0], size=B, p=n_node_hist)
            for i, n in enumerate(n_nodes_list):
                node_mask[i, :n] = True

            edge_mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
            if self.mask_diag_edges:
                diag = ~torch.eye(self.max_nodes, device=device, dtype=torch.bool).unsqueeze(0)
                edge_mask = edge_mask & diag

            # posterior mean and precision (scalar / iid observation)
            mu_X_t = torch.zeros(B, self.max_nodes, 1, device=device, dtype=torch.float32)
            mu_A_t = torch.zeros(B, self.max_nodes, self.max_nodes, device=device, dtype=torch.float32)
            ro_X = torch.ones_like(mu_X_t)
            ro_A = torch.ones_like(mu_A_t)

            beta_prev_X = torch.zeros((B, 1), device=device, dtype=torch.float32)
            beta_prev_A = torch.zeros((B, 1), device=device, dtype=torch.float32)

            for step in tqdm(range(1, T_steps + 1)):
                t_val = (step - 1) / T_steps
                t = torch.full((B, 1), float(t_val), device=device, dtype=torch.float32).clamp_min(self.t_min)

                # alpha schedule aligned with pure VBFN
                beta_X = (torch.pow(self.sigma1_X, -2 * t) - 1.0)  # [B,1]
                beta_A = (torch.pow(self.sigma1_A, -2 * t) - 1.0)
                alpha_X = (beta_X - beta_prev_X).clamp_min(0.0)  # [B,1]
                alpha_A = (beta_A - beta_prev_A).clamp_min(0.0)
                beta_prev_X, beta_prev_A = beta_X, beta_A

                gamma_X = 1 - torch.pow(self.sigma1_X, 2 * t)
                gamma_A = 1 - torch.pow(self.sigma1_A, 2 * t)

                mu_X_t = torch.clamp(mu_X_t, min=-10, max=10) * node_mask.unsqueeze(-1)
                mu_A_t = torch.clamp(mu_A_t, min=-10, max=10) * edge_mask.float()

                mu_X_x, sigma_X_x, mu_A_x, sigma_A_x = self.interdependency_modeling(
                    mu_X_t=mu_X_t,
                    mu_A_t=mu_A_t,
                    t=t,
                    gamma_X=gamma_X,
                    gamma_A=gamma_A,
                    node_mask=node_mask
                )

                p_X = (self.discretised_cdf(mu_X_x, sigma_X_x, self.k_r_X.view(1, 1, -1, 1)) -
                       self.discretised_cdf(mu_X_x, sigma_X_x, self.k_l_X.view(1, 1, -1, 1))).squeeze(-1)  # [B,N,K]
                p_A = (self.discretised_cdf(mu_A_x, sigma_A_x, self.k_r_A.view(1, 1, 1, -1, 1)) -
                       self.discretised_cdf(mu_A_x, sigma_A_x, self.k_l_A.view(1, 1, 1, -1, 1))).squeeze(-1)  # [B,N,N,K]

                # mask + clamp + renorm
                p_X = p_X * node_mask.unsqueeze(-1)
                p_X = p_X.clamp_min(self.eps_prob)
                p_X = p_X / p_X.sum(dim=-1, keepdim=True).clamp_min(self.eps_prob)

                p_A = p_A * edge_mask.unsqueeze(-1)
                p_A = p_A.clamp_min(self.eps_prob)
                p_A = p_A / p_A.sum(dim=-1, keepdim=True).clamp_min(self.eps_prob)

                # expected centers
                kcx = self.K_c_X.to(p_X.dtype).view(1, 1, -1)
                x_mean = (p_X * kcx).sum(dim=-1, keepdim=True) * node_mask.unsqueeze(-1)  # [B,N,1]

                kca = self.K_c_A.to(p_A.dtype).view(1, 1, 1, -1)
                a_mean = (p_A * kca).sum(dim=-1) * edge_mask.float()  # [B,N,N]

                # sample message y: N(mean, (alpha I)^-1)
                if float(alpha_X.max()) > 0.0:
                    y_X = x_mean + torch.randn_like(x_mean) / torch.sqrt(alpha_X.view(B, 1, 1).clamp_min(1e-12))
                else:
                    y_X = x_mean

                if float(alpha_A.max()) > 0.0:
                    y_A = a_mean + torch.randn_like(a_mean) / torch.sqrt(alpha_A.view(B, 1, 1).clamp_min(1e-12))
                else:
                    y_A = a_mean

                # Bayes fusion (iid)
                aX = alpha_X.view(B, 1, 1)
                aA = alpha_A.view(B, 1, 1)

                mu_X_t = (ro_X * mu_X_t + aX * y_X) / (ro_X + aX + 1e-12)
                mu_A_t = (ro_A * mu_A_t + aA * y_A) / (ro_A + aA + 1e-12)
                ro_X = ro_X + aX
                ro_A = ro_A + aA

                # keep masks + symmetry
                mu_X_t = mu_X_t * node_mask.unsqueeze(-1)
                mu_A_t = mu_A_t * edge_mask.float()
                mu_A_t = 0.5 * (mu_A_t + mu_A_t.transpose(1, 2))

            # final decode at t=1
            final_t = torch.ones((B, 1), device=device, dtype=torch.float32).clamp_min(self.t_min)
            gamma_X_final = torch.full_like(final_t, 1 - self.sigma1_X ** 2)
            gamma_A_final = torch.full_like(final_t, 1 - self.sigma1_A ** 2)

            mu_X_x, sigma_X_x, mu_A_x, sigma_A_x = self.interdependency_modeling(
                mu_X_t=mu_X_t,
                mu_A_t=mu_A_t,
                t=final_t,
                gamma_X=gamma_X_final,
                gamma_A=gamma_A_final,
                node_mask=node_mask
            )

            p_X = (self.discretised_cdf(mu_X_x, sigma_X_x, self.k_r_X.view(1, 1, -1, 1)) -
                   self.discretised_cdf(mu_X_x, sigma_X_x, self.k_l_X.view(1, 1, -1, 1))).squeeze(-1)  # [B,N,K]
            p_A = (self.discretised_cdf(mu_A_x, sigma_A_x, self.k_r_A.view(1, 1, 1, -1, 1)) -
                   self.discretised_cdf(mu_A_x, sigma_A_x, self.k_l_A.view(1, 1, 1, -1, 1))).squeeze(-1)  # [B,N,N,K]

            p_X = p_X * node_mask.unsqueeze(-1)
            p_X = p_X.clamp_min(self.eps_prob)
            p_X = p_X / p_X.sum(dim=-1, keepdim=True).clamp_min(self.eps_prob)

            p_A = p_A * edge_mask.unsqueeze(-1)
            p_A = p_A.clamp_min(self.eps_prob)
            p_A = p_A / p_A.sum(dim=-1, keepdim=True).clamp_min(self.eps_prob)

            # adjacency discrete types by expected center -> nearest discrete
            kca = self.K_c_A.view(1, 1, 1, -1).to(p_A.dtype)
            a_center = (p_A * kca).sum(dim=-1)  # [B,N,N] in [-1,1]
            a_idx = map_continuous_to_discrete(a_center, self.K_A)

            # enforce symmetry + diag=0
            a_idx = torch.maximum(a_idx, a_idx.transpose(1, 2))
            a_idx = a_idx * edge_mask.long()
            a_idx = a_idx * (~torch.eye(self.max_nodes, device=device, dtype=torch.bool).unsqueeze(0)).long()

            all_x.append(p_X.detach().cpu())
            all_adj.append(a_idx.detach().cpu())
            all_nm.append(node_mask.detach().cpu())

        x = torch.cat(all_x, dim=0)[:num_samples]
        adj = torch.cat(all_adj, dim=0)[:num_samples]
        nm = torch.cat(all_nm, dim=0)[:num_samples]
        return x, adj, nm
