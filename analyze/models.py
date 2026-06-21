import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj: torch.Tensor, eps=1e-8):
    """D^{-1/2} A D^{-1/2}"""
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt

class ChebNet(nn.Module):
    """
    Chebyshev Graph Convolution (K=2)
    T0(x) = x
    T1(x) = L_tilde x
    T2(x) = 2 L_tilde T1 - T0
    """
    def __init__(self, in_dim, out_dim, K=2):
        super().__init__()
        
        self.W0 = nn.Linear(in_dim, out_dim, bias=False)
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        x: (N, F)
        adj: (N, N)
        return: (N, out_dim)
        """
        N = adj.size(0)
        device = x.device

        # 构造对称归一化拉普拉斯 L̃ = I - D^{-1/2} A D^{-1/2}
        A_norm = normalize_adj(adj)
        I = torch.eye(N, device=device)
        L = I - A_norm

        # Chebyshev 多项式
        T0 = x                               # (N, F)
        T1 = L @ x                           # (N, F)
        T2 = 2 * (L @ T1) - T0               # (N, F)

        out = (
            self.W0(T0) +
            self.W1(T1) +
            self.W2(T2)
        )
        return out


class MTAEncoder(nn.Module):
    def __init__(self, in_dim, k=2, hidden_dim=32, num_heads=4, dropout=0.1, concat=True):
        """
        in_dim: 输入维度
        hidden_dim: 每个 head 的输出维度
        num_heads: 多头数
        concat: True=拼接输出，False=平均（一般用于最后一层）
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)

        # 1) Chebyshev 卷积
        self.cheb = ChebNet(in_dim=in_dim, out_dim=hidden_dim, K=k)

        # 2) 多头 GAT：每个 head 有独立参数
        self.attn_src = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_heads)
        ])
        self.attn_dst = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_heads)
        ])
        self.attn_scorer = nn.ModuleList([
            nn.Linear(2 * hidden_dim, 1, bias=False) for _ in range(num_heads)
        ])

        # 输出投影
        out_dim = hidden_dim * num_heads if concat else hidden_dim
        self.out_proj = nn.Linear(out_dim, hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        N = adj.size(0)
        A = adj.float()

        # -------------------------
        # 1. Chebyshev 卷积
        # -------------------------
        h = F.relu(self.cheb(x, A))       # (N, hidden_dim)
        h = self.dropout(h)

        # -------------------------
        # 2. 多头 GAT 注意力
        # -------------------------
        head_outputs = []

        for k in range(self.num_heads):
            src_k = self.attn_src[k](h)       # (N, hidden_dim)
            dst_k = self.attn_dst[k](h)

            e_k = torch.zeros((N, N), device=x.device)

            # 计算所有边 (i,j) 的注意力分数
            for i in range(N):
                concat_ij = torch.cat([
                    src_k[i].repeat(N, 1), 
                    dst_k
                ], dim=-1)                   # (N, 2*hidden_dim)

                e_k[i] = self.attn_scorer[k](concat_ij).squeeze(-1)

            e_k = F.leaky_relu(e_k)

            # mask 非邻居
            neg_inf = -1e9
            e_masked = torch.where(adj.bool(), e_k, torch.full_like(e_k, neg_inf))

            # softmax
            attn_k = torch.softmax(e_masked, dim=1)
            attn_k = self.dropout(attn_k)

            # head_k 输出: (N, hidden_dim)
            h_k = attn_k @ h
            head_outputs.append(h_k)

        # -------------------------
        # 3. 多头输出整合
        # -------------------------
        if self.concat:
            # (N, num_heads*hidden_dim)
            h_cat = torch.cat(head_outputs, dim=-1)
        else:
            # 平均（(N, hidden_dim)）
            h_cat = torch.stack(head_outputs, dim=0).mean(dim=0)

        # -------------------------
        # 4. 输出投影
        # -------------------------
        z = self.out_proj(h_cat)
        return z


class GATEncoder(nn.Module):
    """Pure multi-head graph attention encoder without the ChebNet front-end."""

    def __init__(self, in_dim, hidden_dim=32, num_heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.attn_src = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_heads)
        ])
        self.attn_dst = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_heads)
        ])
        self.attn_scorer = nn.ModuleList([
            nn.Linear(2 * hidden_dim, 1, bias=False) for _ in range(num_heads)
        ])
        out_dim = hidden_dim * num_heads if concat else hidden_dim
        self.out_proj = nn.Linear(out_dim, hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        N = adj.size(0)
        h = F.relu(self.input_proj(x))
        h = self.dropout(h)
        head_outputs = []

        for k in range(self.num_heads):
            src_k = self.attn_src[k](h)
            dst_k = self.attn_dst[k](h)
            e_k = torch.zeros((N, N), device=x.device)
            for i in range(N):
                concat_ij = torch.cat([src_k[i].repeat(N, 1), dst_k], dim=-1)
                e_k[i] = self.attn_scorer[k](concat_ij).squeeze(-1)

            e_k = F.leaky_relu(e_k)
            self_mask = torch.eye(N, dtype=torch.bool, device=x.device)
            mask = adj.bool() | self_mask
            e_masked = torch.where(mask, e_k, torch.full_like(e_k, -1e9))
            attn_k = torch.softmax(e_masked, dim=1)
            attn_k = self.dropout(attn_k)
            head_outputs.append(attn_k @ h)

        if self.concat:
            h_cat = torch.cat(head_outputs, dim=-1)
        else:
            h_cat = torch.stack(head_outputs, dim=0).mean(dim=0)
        return self.out_proj(h_cat)


class GCNEncoder(nn.Module):
    """Two-layer GCN encoder without attention."""

    def __init__(self, in_dim, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        N = adj.size(0)
        A = adj.float() + torch.eye(N, device=x.device)
        A_norm = normalize_adj(A)
        h = A_norm @ x
        h = F.relu(self.lin1(h))
        h = self.dropout(h)
        h = A_norm @ h
        return self.lin2(h)


def build_graph_encoder(encoder_type: str, in_dim: int, hidden_dim: int = 32, dropout: float = 0.1) -> nn.Module:
    encoder_type = (encoder_type or "mta").lower()
    if encoder_type == "mta":
        return MTAEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    if encoder_type == "gat":
        return GATEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    if encoder_type == "gcn":
        return GCNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unknown analyzer encoder_type: {encoder_type}")


class Informer(nn.Module):
    """
    Informer-like encoder-only model (one-shot prediction).
    - 在 encoder 输出之上使用 one-shot 预测头（flatten + linear）直接预测未来 H 步
    返回 (node_preds, edge_preds)
    """
    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        horizon: int = 6,
        use_causal_mask: bool = True,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.horizon = horizon
        self.use_causal_mask = use_causal_mask

        # projection from encoder d_model to prediction space
        # node head will produce H * N * 2 values when flattened per-batch; we'll shape later
        self.node_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # edge projection
        self.edge_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # final linear heads: map per-node d_model -> H * 2, per-edge d_model -> H
        # registered here so parameters are tracked by the module/optimizer
        self.node_out_head = nn.Linear(d_model, self.horizon * 2)
        self.edge_out_head = nn.Linear(d_model, self.horizon)

    @staticmethod
    def _generate_causal_mask(sz: int, device: torch.device):
        # 下三角为0，上三角为 -inf（阻止注意）
        mask = torch.full((sz, sz), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask


    def forward(
        self,
        seq_x: torch.Tensor,      # (T, N, d_model)
        edge_repr: torch.Tensor,  # (N, d_model) 兼容参数，当前实现由编码得到
        edge_index: torch.Tensor, # (2, E) long
    ):
        """
        返回：
        - node_preds: (H, N, 2)
        - edge_preds: (H, E, 1)
        """
        # 编码器：输入 (T, N, d_model) 直接进入
        memory = self.encoder(seq_x)        # (T, N, d_model)

        src = edge_index[0]
        dst = edge_index[1]

        H = self.horizon
        device = seq_x.device

        # 使用最后一个时间步的表示（保留时间信息）作为每个节点的表示
        # memory: (T, N, d_model) -> last step repr: (N, d_model)
        node_repr = memory[-1]  # (N, d_model)
        node_feat = self.node_proj(node_repr)  # (N, d_model)

        # 对节点进行one-shot预测：为每个节点输出 H * 2 值
        N = node_feat.size(0)
        node_flat = self.node_out_head(node_feat)  # (N, H*2)
        # reshape -> (H, N, 2)
        node_preds = node_flat.view(N, H, 2).permute(1, 0, 2).contiguous()

        # 边的预测：若没有边返回空
        if edge_index.size(1) > 0:
            E = edge_index.size(1)
            # construct edge representations by averaging src/dst node repr
            edge_repr_calc = 0.5 * (node_repr[src] + node_repr[dst])  # (E, d_model)
            edge_feat = self.edge_proj(edge_repr_calc)  # (E, d_model)
            edge_flat = self.edge_out_head(edge_feat)  # (E, H)
            edge_preds = edge_flat.view(E, H, 1).permute(1, 0, 2).contiguous()  # (H, E, 1)
        else:
            edge_preds = torch.zeros(H, 0, 1, device=device)

        return node_preds, edge_preds
