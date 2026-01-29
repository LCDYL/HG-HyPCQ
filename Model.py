import faiss
import faiss.contrib.torch_utils
import warnings
from sklearn.cluster import KMeans
from torch_geometric.nn import HypergraphConv, LayerNorm
from Hyperbolic_Encoder import *

warnings.filterwarnings("ignore")

class Classification_Head(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.Layer1 = nn.Linear(in_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.Layer2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.Layer1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.Layer2(x)
        return x

def Make_Mask(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """
    -------- input
    x: [B, C, Np]
    mask_ratio: [0, 1)
    -------- output
    mask: [B, C, Np] bool，True = masked
    """
    B, C, Np = x.shape
    mask = torch.zeros((B, C, Np), dtype=torch.bool, device=x.device)

    k_per = int(round(mask_ratio * Np))
    if k_per == 0:
        return mask

    for b in range(B):
        for c in range(C):
            idx = torch.randperm(Np, device=x.device)[:k_per]
            mask[b, c, idx] = True
    return mask

def Make_Patch(C: int, T: int, win: int, stride: int):
    """
    -------- input
    C : int
    T : int
    win : int
    stride : int
    -------- output
    P_index : torch.Tensor
        [C, Np, 2]， [start, end)。
    """
    # 0, stride, 2*stride, ... <= T - win
    dtype = torch.long
    starts = torch.arange(0, T - win + 1, stride, dtype=dtype)  # [Np]
    Np = int(starts.numel())
    ends = starts + win  # [Np]
    # [1,Np,1] -> [C,Np,1]
    s = starts.view(1, Np, 1).expand(C, Np, 1)  # [C,Np,1]
    e = ends.view(1, Np, 1).expand(C, Np, 1)    # [C,Np,1]
    P_index = torch.cat([s, e], dim=-1).contiguous()  # [C,Np,2]

    return P_index

class MyHyperPQEncoder(nn.Module):
    """
    HyperPQ patch encoder
    -------- input
        x: [in_dim] or [N, in_dim] or [..., in_dim]
    -------- output
        emb:  [..., out_dim]
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        M: int,
        K: int,
        softmax_temp: float = 1.0,
        quant_method: str = "softmax",
        init_neg_curvs: float = 1.0,
        clip_r: float = 1.0,
        use_alpha: bool = False,
    ):
        super().__init__()
        assert out_dim % M == 0, "out_dim must = M * D"
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.M = int(M)
        self.D = self.out_dim // self.M

        self.input_proj = nn.Linear(self.in_dim, self.out_dim)

        self.hyper_pq_head = FullHyperPQhead(
            feat_dim=self.out_dim,
            M=self.M,
            K=K,
            softmax_temp=softmax_temp,
            quant_method=quant_method,
            init_neg_curvs=init_neg_curvs,
            clip_r=clip_r,
            use_alpha=use_alpha,
            writer=None,
        )

    def _hyper_to_tangent_batch(self, x_hat: torch.Tensor) -> torch.Tensor:
        """
        -------- input
            x_hat: [N, M, D]
        -------- output
            return: [N, out_dim]
        """
        assert x_hat.dim() == 3 and x_hat.size(1) == self.M and x_hat.size(2) == self.D, \
            f"looking for x_hat=[N,{self.M},{self.D}]，actrually {tuple(x_hat.shape)}"

        v_list = []
        for i in range(self.M):
            neg_c_i = torch.clamp(self.hyper_pq_head.neg_curvs[i], min=1e-6, max=1e6)
            k_i = 1.0 / neg_c_i
            # xi_hat: [N, D]
            xi_hat = x_hat[:, i, :]
            vi = logmap0(xi_hat, k=k_i)  # [N, D]
            v_list.append(vi)

        v = torch.stack(v_list, dim=1)          # [N, M, D]
        emb = v.reshape(v.size(0), self.out_dim) # [N, out_dim]
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        -------- input
            x: [in_dim] / [N,in_dim] / [...,in_dim]
        -------- output
            ret: [..., out_dim]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, in_dim]

        assert x.size(-1) == self.in_dim, f"期望 x 最后一维={self.in_dim}, 实际 {tuple(x.shape)}"

        orig_shape = x.shape[:-1]         # [...]
        x2d = x.reshape(-1, self.in_dim)  # [N, in_dim]

        feat = self.input_proj(x2d)       # [N, out_dim]

        # [N, out_dim]，
        hyper_x, x_hat, codes, soft_codes, quant_err = self.hyper_pq_head(feat)

        # x_hat -> [N, M, D]
        emb2d = self._hyper_to_tangent_batch(x_hat)  # [N, out_dim]
        # emb2d = self._hyper_to_tangent_batch(hyper_x)  # [N, out_dim]

        emb = emb2d.reshape(*orig_shape, self.out_dim)  # [..., out_dim]
        return emb

class Original_EEG_Encoder(nn.Module):
    """
    EEG Encoder
    -------- intput
        X: [B, C, T]
    -------- output
        E: [B, C, Np, emb_dim]
    -------- others
        stride == patch_win
        T % patch_win == 0
    """
    def __init__(
        self,
        patch_win: int,
        out_dim: int,
        M: int,
        K: int,
        softmax_temp: float = 1.0,
        quant_method: str = "softmax",
        init_neg_curvs: float = 1.0,
        clip_r: float = 1.0,
        use_alpha: bool = False,
    ):
        super().__init__()
        self.patch_win = int(patch_win)
        self.emb_dim = int(out_dim)

        self.encoder = MyHyperPQEncoder(
            in_dim=self.patch_win,
            out_dim=self.emb_dim,
            M=int(M),
            K=int(K),
            softmax_temp=softmax_temp,
            quant_method=quant_method,
            init_neg_curvs=init_neg_curvs,
            clip_r=clip_r,
            use_alpha=use_alpha,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B,C,T]
        B, C, T = X.shape
        if T % self.patch_win != 0:
            print(f"[Original_EEG_Encoder] illegal：T={T} can't be divided patch_win={self.patch_win}")
            exit(20)

        # Np = T/patch_win，
        patches = X.unfold(dimension=-1, size=self.patch_win, step=self.patch_win)  # [B,C,Np,patch_win]
        patches = patches.contiguous()
        B, C, Np, Pw = patches.shape  # Pw==patch_win

        flat = patches.view(B * C * Np, Pw)           # [N, patch_win]
        emb = self.encoder(flat)                      # [N, emb_dim]
        E = emb.view(B, C, Np, self.emb_dim)          # [B,C,Np,D]
        return E

class EEG_Position_Coder(nn.Module):
    """
    -------- input
        E: [B, C, Np, D]
    -------- output
        E_prime: [B, C, Np, D]
    """

    def __init__(self, emb_dim: int, n: int, x1: int, x2: int):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.n = int(n)
        self.x1 = int(x1)
        self.x2 = int(x2)

        # depthwise 2D conv: input [N, D, C, n] -> output [N, D, C, 1]
        self.time_dwconv = nn.Conv2d(
            in_channels=self.emb_dim,
            out_channels=self.emb_dim,
            kernel_size=(1, self.n),
            padding=0,
            groups=self.emb_dim,
            bias=True
        )

        self.conv1 = HypergraphConv(self.emb_dim, self.emb_dim)
        self.ln1   = LayerNorm(self.emb_dim)
        self.conv2 = HypergraphConv(self.emb_dim, self.emb_dim)
        self.ln2   = LayerNorm(self.emb_dim)

        m = self.x2 - self.x1
        if m < 0:
            raise ValueError(f"[EEG_Position_Coder] invalid x1/x2: x1={self.x1}, x2={self.x2}")

        self.m = m 
        self.N_nodes = 1 + m

        if m == 0:
            self.register_buffer("_hyperedge_local", torch.empty((2, 0), dtype=torch.long), persistent=False)
            self._conn_local = 0
        else:
            node_list = []
            edge_list = []
            # edge e includes nodes 0..(e+1)
            for e in range(m):
                nodes = torch.arange(0, e + 2, dtype=torch.long)  # [0,1,...,e+1]
                edges = torch.full((nodes.numel(),), e, dtype=torch.long)
                node_list.append(nodes)
                edge_list.append(edges)

            node_idx = torch.cat(node_list, dim=0)
            edge_idx = torch.cat(edge_list, dim=0)
            hyperedge_local = torch.stack([node_idx, edge_idx], dim=0)  # [2, conn]
            self.register_buffer("_hyperedge_local", hyperedge_local, persistent=False)
            self._conn_local = int(hyperedge_local.size(1))

        self._hyperedge_cache = {}

    def _build_time_window_indices(self, Np: int, device: torch.device) -> torch.Tensor:
        """
        -------- input
            start = clamp(p-half, 0, Np-n)
            idx[p, :] = start + [0..n-1]
        -------- output
            idx: [Np, n] long
        """
        if Np < self.n:
            raise ValueError(
                f"[EEG_Position_Coder] Np < n, Np={Np}, n={self.n}. "
                f"please decrease Time_Area_n or increase Np（decrease patch_length 或increase win）"
            )

        half = (self.n - 1) // 2
        p = torch.arange(Np, device=device, dtype=torch.long)          # [Np]
        start = p - half
        start = torch.clamp(start, min=0, max=(Np - self.n))           # [Np]
        offs = torch.arange(self.n, device=device, dtype=torch.long)   # [n]
        idx = start[:, None] + offs[None, :]                           # [Np, n]
        return idx

    def _compute_Et(self, E: torch.Tensor) -> torch.Tensor:
        """
        -------- input
            E: [B, C, Np, D]
        -------- output
            Et: [B, Np, C, D]
        """
        B, C, Np, D = E.shape
        device = E.device

        idx = self._build_time_window_indices(Np, device)  # [Np, n]

        E_bcdp = E.permute(0, 1, 3, 2).contiguous()  # [B,C,D,Np]

        idx_exp = idx.view(1, 1, 1, Np, self.n).expand(B, C, D, Np, self.n)  # [B,C,D,Np,n]
        E_bcdp5 = E_bcdp.unsqueeze(-1).expand(B, C, D, Np, self.n)  # [B,C,D,Np,n]
        near = torch.gather(E_bcdp5, dim=3, index=idx_exp)  # [B,C,D,Np,n]

        near = near.permute(0, 3, 2, 1, 4).contiguous()  # [B,Np,D,C,n]
        near2 = near.view(B * Np, D, C, self.n)          # [B*Np,D,C,n]

        y = self.time_dwconv(near2)                      # [B*Np,D,C,1]
        y = y.squeeze(-1)                                # [B*Np,D,C]
        Et = y.permute(0, 2, 1).contiguous()             # [B*Np,C,D]
        Et = Et.view(B, Np, C, D)                        # [B,Np,C,D]
        return Et

    def _get_big_hyperedge_index(self, Q: int, device: torch.device) -> torch.Tensor:
        """
        --------output
            hyperedge_index: [2, Q*conn_local]
        """
        if self.m == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        key = (Q, str(device))
        if key in self._hyperedge_cache:
            return self._hyperedge_cache[key]

        local = self._hyperedge_local.to(device)   # [2, conn_local]
        conn = self._conn_local
        N = self.N_nodes
        m = self.m

        # offsets: [Q,1]
        g = torch.arange(Q, device=device, dtype=torch.long).view(Q, 1)
        node_off = g * N
        edge_off = g * m

        # broadcast add: [Q, conn]
        node_idx = local[0].view(1, conn) + node_off
        edge_idx = local[1].view(1, conn) + edge_off

        hyperedge_index = torch.stack([node_idx.reshape(-1), edge_idx.reshape(-1)], dim=0)  # [2, Q*conn]
        self._hyperedge_cache[key] = hyperedge_index
        return hyperedge_index

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        # E: [B,C,Np,D]
        B, C, Np, D = E.shape

        if self.m == 0:
            return E

        if self.x2 > C:
            raise ValueError(f"[EEG_Position_Coder] x2 must be <= C. Got x2={self.x2}, C={C}")
        if self.x2 - self.x1 <= 0:
            return E

        # Et: [B,Np,C,D]
        Et = self._compute_Et(E)

        cur = E.permute(0, 2, 1, 3).contiguous()  # [B,Np,C,D]
        cur_n = F.normalize(cur, dim=-1)
        et_n  = F.normalize(Et,  dim=-1)

        sims = torch.einsum("bpcd,bpkd->bpck", cur_n, et_n)  # [B,Np,C,C]

        eye = torch.eye(C, device=E.device, dtype=torch.bool).view(1, 1, C, C)
        sims = sims.masked_fill(eye, -1e9)

        idx_top = torch.topk(sims, k=self.x2, dim=-1, largest=True, sorted=True).indices  # [B,Np,C,x2]
        idx_pick = idx_top[..., self.x1:self.x2]                                           # [B,Np,C,m]

        # neighbors: [B,Np,C,m,D]
        Et_expand = Et.unsqueeze(2).expand(B, Np, C, C, D)  # [B,Np,C,C,D]
        idx_exp = idx_pick.unsqueeze(-1).expand(B, Np, C, self.m, D)
        neighbors = torch.take_along_dim(Et_expand, idx_exp, dim=3)  # [B,Np,C,m,D]

        # Xn: [B,Np,C,N_nodes,D] where N_nodes=1+m
        Xn = torch.cat([cur.unsqueeze(3), neighbors], dim=3)  # [B,Np,C,1+m,D]

        Q = B * Np * C
        N = self.N_nodes

        X_flat = Xn.view(Q * N, D)  # [num_nodes_total, D]
        hyperedge_index = self._get_big_hyperedge_index(Q, device=E.device)  # [2, num_conn_total]

        Z1 = self.conv1(X_flat, hyperedge_index)
        Z1 = self.ln1(Z1)
        Z2 = self.conv2(Z1, hyperedge_index)
        Z2 = self.ln2(Z2)

        Z2 = Z2.view(Q, N, D)
        emb_TS = Z2[:, 0, :]  # [Q,D]

        emb_TS = emb_TS.view(B, Np, C, D).permute(0, 2, 1, 3).contiguous()  # [B,C,Np,D]

        E_prime = E + emb_TS
        return E_prime

class EEG_Mask_Reconstruc_Task(nn.Module):
    """
    -------- input
      X:       [B, C, T]
      E_prime: [B, C, Np, emb_dim]
      P_index: [C, Np, 2]
      M_index: [B, C, Np]  True=mask
    -------- output
      recon: [B, C, T]
      loss:  batch mean
    """
    def __init__(self, emb_dim: int, T: int, Np: int, mask_ratio: float = None):
        super().__init__()
        k_per = int(round((mask_ratio if mask_ratio is not None else 0.0) * Np))
        num_keep = Np - k_per

        self.emb_dim  = int(emb_dim)
        self.T        = int(T)
        self.Np       = int(Np)
        self.num_keep = int(num_keep)

        in_feats = self.num_keep * self.emb_dim
        self.to_signal = nn.Linear(in_feats, self.T)

    @staticmethod
    def _sorted_keep_indices(m_row: torch.Tensor):
        keep = (~m_row).nonzero(as_tuple=False).view(-1)
        return keep

    def forward(self, X, E_prime, P_index, M_index):
        # X: [B,C,T], E_prime:[B,C,Np,D], P_index:[C,Np,2], M_index:[B,C,Np]
        B, C, T = X.shape
        recon = X.new_zeros((B, C, T))

        loss_sum = X.new_tensor(0.0)

        for b in range(B):
            for c in range(C):
                keep_idx = self._sorted_keep_indices(M_index[b, c])      # [num_keep]
                keep_emb = E_prime[b, c, keep_idx, :]                    # [num_keep, D]
                ch_vec = keep_emb.reshape(-1)                             # [num_keep*D]
                recon[b, c] = self.to_signal(ch_vec)                      # [T]

            total_err = X.new_tensor(0.0)
            total_count = 0

            for c in range(C):
                masked_idx = (M_index[b, c].nonzero(as_tuple=False).view(-1))  # [n_masked]
                if masked_idx.numel() == 0:
                    continue

                for p in masked_idx:
                    start, end = P_index[c, p]
                    s = int(start.item())
                    e = int(end.item())
                    x_seg = X[b, c, s:e]
                    rec_seg = recon[b, c, s:e]
                    diff = rec_seg - x_seg
                    total_err += (diff * diff).sum()
                    total_count += (e - s)

            if total_count > 0:
                loss_b = total_err / float(total_count)
            else:
                loss_b = X.new_tensor(0.0)

            loss_sum = loss_sum + loss_b

        loss = loss_sum / B
        return recon, loss

class Hierarchical_Clustering_Task(nn.Module):
    """
    -------- input
      E_prime: [C, Np, D] 或 [B, C, Np, D]
    """
    def __init__(self, margin_M: float = 1.0, seed: int = 0, w_min: float = 0.2, w_max: float = 0.8):
        super().__init__()
        self.margin_M = float(margin_M)
        self.seed = int(seed)
        self.w_min = float(w_min)
        self.w_max = float(w_max)

    @staticmethod
    def _compute_centers(X: torch.Tensor, labels: torch.Tensor, k: int):
        d = X.size(1)
        centers = X.new_zeros((k, d))
        for j in range(k):
            sel = (labels == j)
            if sel.any():
                centers[j] = X[sel].mean(dim=0)
        return centers

    @staticmethod
    def _lc_loss(X: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(X, centers, p=2) ** 2
        return dists[torch.arange(X.size(0), device=X.device), labels].mean()

    def _ld_loss(self, centers: torch.Tensor) -> torch.Tensor:
        k = centers.size(0)
        if k <= 1:
            return centers.new_tensor(0.0)
        C2 = torch.cdist(centers, centers, p=2) ** 2
        C2 = C2 + torch.eye(k, device=C2.device) * 1e9
        min_pair = C2.min()
        return torch.clamp(self.margin_M - min_pair, min=0.0)

    def _u_shape_weights(self, L: int) -> torch.Tensor:
        if L == 1:
            return torch.tensor([self.w_max])
        idx = torch.arange(L, dtype=torch.float32)
        w = self.w_min + (self.w_max - self.w_min) * torch.abs(torch.cos(torch.pi * idx / (L - 1)))
        return w

    def _faiss_kmeans_gpu(self, X_np: np.ndarray, k: int, niter: int = 20, nredo: int = 1):
        """
        X_np: float32, shape [N, D]
        return labels_np:[N], centers_np:[k,D]
        """
        N, D = X_np.shape
        X_np = np.ascontiguousarray(X_np.astype(np.float32))

        kmeans = faiss.Kmeans(
            d=D,
            k=k,
            niter=niter,
            nredo=nredo,
            gpu=True,
            seed=self.seed,
            verbose=False,
        )
        kmeans.train(X_np)
        _, I = kmeans.index.search(X_np, 1)   # I: [N,1]
        labels_np = I.reshape(-1).astype(np.int64)
        centers_np = kmeans.centroids.reshape(k, D).astype(np.float32)
        return labels_np, centers_np

    def _forward_single(self, E_prime_single: torch.Tensor) -> torch.Tensor:
        C, Np, D = E_prime_single.shape
        X = E_prime_single.reshape(C * Np, D)

        if not torch.isfinite(X).all():
            print("[Hierarchical_Clustering_Task] non-finite detected in X, applying nan_to_num()")
            exit(0)
            X = torch.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)

        X = torch.clamp(X, min=-1e4, max=1e4)

        k0 = C
        if k0 < 2:
            return X.new_tensor(0.0)

        ks = list(range(k0, 1, -1))
        L = len(ks)
        w = self._u_shape_weights(L).to(X.device)

        X_np = X.detach().float().cpu().numpy()
        labels_np, centers_np = self._faiss_kmeans_gpu(X_np, k0, niter=20, nredo=1)

        labels = torch.from_numpy(labels_np).to(X.device, dtype=torch.long)
        centers = torch.from_numpy(centers_np).to(X.device, dtype=X.dtype)

        loss_total = X.new_tensor(0.0)
        current_k = k0

        for layer_idx, _ in enumerate(ks):
            LC = self._lc_loss(X, labels, centers)
            LD = self._ld_loss(centers)
            loss_total = loss_total + w[layer_idx] * (LC + LD)

            if current_k == 2:
                break

            C2 = torch.cdist(centers, centers, p=2) ** 2
            C2 = C2 + torch.eye(current_k, device=C2.device) * 1e9
            flat = torch.argmin(C2)
            a = int(flat // current_k)
            b = int(flat % current_k)
            if b < a:
                a, b = b, a

            labels = labels.clone()
            labels[labels == b] = a
            labels[labels > b] = labels[labels > b] - 1

            current_k = current_k - 1
            centers = self._compute_centers(X, labels, current_k)

        return loss_total

    def forward(self, E_prime: torch.Tensor) -> torch.Tensor:
        if E_prime.dim() == 3:
            return self._forward_single(E_prime)

        B = E_prime.size(0)
        loss_sum = E_prime.new_tensor(0.0)
        for b in range(B):
            loss_sum = loss_sum + self._forward_single(E_prime[b])
        return loss_sum / B

class MySuperEEG(nn.Module):
    """
    -------- input
        X:[B, C, T]
    -------- output
      E_prime_batch: [B, C, Np, D]
      loss_total: batch mean
    """
    def __init__(
        self,
        win: int,
        patch_length: int,
        sfrq: int = 500,

        Spatial_Area_x1=None,
        Spatial_Area_x2=None,
        Time_Area_n=None,

        enc_out_dim: int = 128,
        pq_M: int = 8,
        pq_K: int = 256,
        pq_softmax_temp: float = 1.0,
        pq_quant_method: str = "softmax",
        pq_init_neg_curvs: float = 1.0,
        pq_clip_r: float = 1.0,
        pq_use_alpha: bool = False,

        mask_ratio: float = 0.5,

        margin_M: float = 1.0,
        w_min: float = 0.2,
        w_max: float = 0.8,

        recon_weight: float = 1.0,
        cluster_weight: float = 1.0,
    ):
        super().__init__()

        self.win = int(win)
        self.sfrq = int(sfrq)
        # self.patch_length = int(patch_length)
        self.patch_length = patch_length
        self.patch_win = int(self.patch_length * self.sfrq)
        self.stride = self.patch_win

        if self.patch_win <= 0:
            print(f"[MySuperEEG] patch_win 非法: patch_length={self.patch_length}, sfrq={self.sfrq}")
            exit(1)

        if (self.win % self.patch_win) != 0:
            print(f"[MySuperEEG] 配置错误：win(={self.win}) 不是 patch_win(={self.patch_win}) 的整倍数。")
            print(f"Please adjust win or patch_length/sfrq to make win % patch_win == 0")
            exit(2)

        self.Np = self.win // self.patch_win

        self.emb_dim = int(enc_out_dim)
        self.mask_ratio = float(mask_ratio)
        self.recon_weight = float(recon_weight)
        self.cluster_weight = float(cluster_weight)

        self.encoder = Original_EEG_Encoder(
            patch_win=self.patch_win,
            out_dim=self.emb_dim,
            M=pq_M,
            K=pq_K,
            softmax_temp=pq_softmax_temp,
            quant_method=pq_quant_method,
            init_neg_curvs=pq_init_neg_curvs,
            clip_r=pq_clip_r,
            use_alpha=pq_use_alpha,
        )

        self.eeg_position_coder = EEG_Position_Coder(
            emb_dim=self.emb_dim,
            x1=Spatial_Area_x1,
            x2=Spatial_Area_x2,
            n=Time_Area_n
        )

        self.reconstructor = EEG_Mask_Reconstruc_Task(
            emb_dim=self.emb_dim,
            T=self.win,
            Np=self.Np,
            mask_ratio=self.mask_ratio
        )

        self.cluster_task = Hierarchical_Clustering_Task(
            margin_M=margin_M, seed=0, w_min=w_min, w_max=w_max
        )

    def forward(self, X: torch.Tensor):
        # X: [B,C,T]
        B, C, T = X.shape

        if T != self.win:
            raise SystemExit(f"[MySuperEEG] 输入 T={T} 与配置 win={self.win} 不一致（请统一）")

        P_index = Make_Patch(C=C, T=T, win=self.patch_win, stride=self.stride).to(X.device)  # [C,Np,2]
        Np = P_index.shape[1]

        x_ref = X.new_empty((B, C, Np))
        M_index = Make_Mask(x_ref, self.mask_ratio)  # [B,C,Np]

        E = self.encoder(X)  # [B,C,Np,D]

        E_prime_batch = self.eeg_position_coder(E)

        _, loss_recon = self.reconstructor(X, E_prime_batch, P_index, M_index)

        loss_cluster = self.cluster_task(E_prime_batch)

        loss_total = self.recon_weight * loss_recon + self.cluster_weight * loss_cluster

        return E_prime_batch, loss_total

