import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit

EXP_MAX_NORM = 10

# @torch.jit.script
def arcosh(x: torch.Tensor):
    dtype = x.dtype
    z = torch.sqrt(torch.clamp_min(x.double().pow(2) - 1.0, 1e-15))
    return torch.log(x + z).to(dtype)


def inner(u, v, *, keepdim=False, dim=-1):
    r"""
    Minkowski inner product.

    .. math::
        \langle\mathbf{u}, \mathbf{v}\rangle_{\mathcal{L}}:=-u_{0} v_{0}+u_{1} v_{1}+\ldots+u_{d} v_{d}

    Parameters
    ----------
    u : tensor
        vector in ambient space
    v : tensor
        vector in ambient space
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        inner product
    """
    return _inner(u, v, keepdim=keepdim, dim=dim)


# @torch.jit.script
def _inner(u, v, keepdim: bool = False, dim: int = -1):
    d = u.size(dim) - 1
    uv = u * v
    if keepdim is False:
        return -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + uv.narrow(
            dim, 1, d
        ).sum(dim=dim, keepdim=False)
    else:
        return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
            dim=dim, keepdim=True
        )


def inner0(v, *, k, keepdim=False, dim=-1):
    r"""
    Minkowski inner product with zero vector.

    Parameters
    ----------
    v : tensor
        vector in ambient space
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        inner product
    """
    return _inner0(v, k=k, keepdim=keepdim, dim=dim)


# @torch.jit.script
def _inner0(v, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    res = -v.narrow(dim, 0, 1) * torch.sqrt(k)
    if keepdim is False:
        res = res.squeeze(dim)
    return res


def dist(x, y, *, k, keepdim=False, dim=-1):
    r"""
    Compute geodesic distance on the Hyperboloid.

    .. math::

        d_{\mathcal{L}}^{k}(\mathbf{x}, \mathbf{y})=\sqrt{k} \operatorname{arcosh}\left(-\frac{\langle\mathbf{x}, \mathbf{y}\rangle_{\mathcal{L}}}{k}\right)

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    y : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`
    """
    return _dist(x, y, k=k, keepdim=keepdim, dim=dim)


# @torch.jit.script
def _dist(x, y, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    d = -_inner(x, y, dim=dim, keepdim=keepdim)
    return torch.sqrt(k) * arcosh(d / k)


def dist0(x, *, k, keepdim=False, dim=-1):
    r"""
    Compute geodesic distance on the Hyperboloid to zero point.

    .. math::

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and zero point
    """
    return _dist0(x, k=k, keepdim=keepdim, dim=dim)


# @torch.jit.script
def _dist0(x, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    d = -_inner0(x, k=k, dim=dim, keepdim=keepdim)
    return torch.sqrt(k) * arcosh(d / k)


def project(x, *, k, dim=-1):
    r"""
    Projection on the Hyperboloid.

    .. math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathbb{H}^{d, 1}}(\mathbf{x}):=\left(\sqrt{k+\left\|\mathbf{x}_{1: d}\right\|_{2}^{2}}, \mathbf{x}_{1: d}\right)

    Parameters
    ----------
    x: tensor
        point in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project(x, k=k, dim=dim)


# @torch.jit.script
def _project(x, k: torch.Tensor, dim: int = -1):
    dn = x.size(dim) - 1
    left_ = torch.sqrt(
        k + torch.norm(x.narrow(dim, 1, dn), p=2, dim=dim) ** 2
    ).unsqueeze(dim)
    right_ = x.narrow(dim, 1, dn)
    proj = torch.cat((left_, right_), dim=dim)
    return proj


def project_polar(x, *, k, dim=-1):
    r"""
    Projection on the Hyperboloid from polar coordinates.

    ... math::
        \pi((\mathbf{d}, r))=(\sqrt{k} \sinh (r/\sqrt{k}) \mathbf{d}, \cosh (r / \sqrt{k}))

    Parameters
    ----------
    x: tensor
        point in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project_polar(x, k=k, dim=dim)


# @torch.jit.script
def _project_polar(x, k: torch.Tensor, dim: int = -1):
    dn = x.size(dim) - 1
    d = x.narrow(dim, 0, dn)
    r = x.narrow(dim, -1, 1)
    res = torch.cat(
        (
            torch.cosh(r / torch.sqrt(k)),
            torch.sqrt(k) * torch.sinh(r / torch.sqrt(k)) * d,
        ),
        dim=dim,
    )
    return res


def project_u(x, v, *, k, dim=-1):
    r"""
    Projection of the vector on the tangent space.

    ... math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathcal{T}_{\mathbf{x}} \mathbb{H}^{d, 1}(\mathbf{v})}:=\mathbf{v}+\langle\mathbf{x}, \mathbf{v}\rangle_{\mathcal{L}} \mathbf{x} / k

    Parameters
    ----------
    x: tensor
        point on the Hyperboloid
    v: tensor
        vector in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project_u(x, v, k=k, dim=dim)


# @torch.jit.script
def _project_u(x, v, k: torch.Tensor, dim: int = -1):
    return v.addcmul(_inner(x, v, dim=dim, keepdim=True), x / k)


def norm(u, *, keepdim=False, dim=-1):
    r"""
    Compute vector norm on the tangent space w.r.t Riemannian metric on the Hyperboloid.

    .. math::

        \|\mathbf{v}\|_{\mathcal{L}}=\sqrt{\langle\mathbf{v}, \mathbf{v}\rangle_{\mathcal{L}}}

    Parameters
    ----------
    u : tensor
        tangent vector on Hyperboloid
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        norm of vector
    """
    return _norm(u, keepdim=keepdim, dim=dim)


# @torch.jit.script
def _norm(u, keepdim: bool = False, dim: int = -1):
    return torch.sqrt(torch.clamp_min(_inner(u, u, keepdim=keepdim), 1e-8))


def expmap(x, u, *, k, dim=-1):
    r"""
    Compute exponential map on the Hyperboloid.

    .. math::

        \exp _{\mathbf{x}}^{k}(\mathbf{v})=\cosh \left(\frac{\|\mathbf{v}\|_{\mathcal{L}}}{\sqrt{k}}\right) \mathbf{x}+\sqrt{k} \sinh \left(\frac{\|\mathbf{v}\|_{\mathcal{L}}}{\sqrt{k}}\right) \frac{\mathbf{v}}{\|\mathbf{v}\|_{\mathcal{L}}}


    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    u : tensor
        unit speed vector on Hyperboloid
    k: tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{x, u}(1)` end point
    """
    return _expmap(x, u, k=k, dim=dim)


# @torch.jit.script
def _expmap(x, u, k: torch.Tensor, dim: int = -1):
    nomin = _norm(u, keepdim=True, dim=dim)

    u = u / nomin

    nomin = (nomin / torch.sqrt(k)).clamp_max(EXP_MAX_NORM)

    p = (
        torch.cosh(nomin) * x
        + torch.sqrt(k) * torch.sinh(nomin) * u
    )
    return p


def expmap0(u, *, k, dim=-1):
    r"""
    Compute exponential map for Hyperboloid from :math:`0`.

    Parameters
    ----------
    u : tensor
        speed vector on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    """
    return _expmap0(u, k, dim=dim)


# @torch.jit.script
def _expmap0(u, k: torch.Tensor, dim: int = -1):
    nomin = _norm(u, keepdim=True, dim=dim)
    u = u / nomin
    nomin = (nomin / torch.sqrt(k)).clamp_max(EXP_MAX_NORM)
    l_v = torch.cosh(nomin) * torch.sqrt(k)
    r_v = torch.sqrt(k) * torch.sinh(nomin) * u
    dn = r_v.size(dim) - 1
    p = torch.cat((l_v + r_v.narrow(dim, 0, 1), r_v.narrow(dim, 1, dn)), dim)
    return p


def logmap(x, y, *, k, dim=-1):
    r"""
    Compute logarithmic map for two points :math:`x` and :math:`y` on the manifold.

    .. math::

        \log _{\mathbf{x}}^{k}(\mathbf{y})=d_{\mathcal{L}}^{k}(\mathbf{x}, \mathbf{y})
            \frac{\mathbf{y}+\frac{1}{k}\langle\mathbf{x},
            \mathbf{y}\rangle_{\mathcal{L}} \mathbf{x}}{\left\|
            \mathbf{y}+\frac{1}{k}\langle\mathbf{x},
            \mathbf{y}\rangle_{\mathcal{L}} \mathbf{x}\right\|_{\mathcal{L}}}

    The result of Logarithmic map is a vector such that

    .. math::

        y = \operatorname{Exp}^c_x(\operatorname{Log}^c_x(y))


    Parameters
    ----------
    x : tensor
        starting point on Hyperboloid
    y : tensor
        target point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`x` to :math:`y`
    """
    return _logmap(x, y, k=k, dim=dim)


# @torch.jit.script
def _logmap(x, y, k, dim: int = -1):
    dist_ = _dist(x, y, k=k, dim=dim, keepdim=True)
    nomin = y + 1.0 / k * _inner(x, y, keepdim=True) * x
    denom = _norm(nomin, keepdim=True)
    return dist_ * nomin / denom


def logmap0(y, *, k, dim=-1):
    r"""
    Compute logarithmic map for :math:`y` from :math:`0` on the manifold.

    Parameters
    ----------
    y : tensor
        target point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`0` to :math:`y`
    """
    return _logmap0(y, k=k, dim=dim)


# @torch.jit.script
def _logmap0(y, k, dim: int = -1):
    dist_ = _dist0(y, k=k, dim=dim, keepdim=True)
    nomin_ = 1.0 / k * _inner0(y, k=k, keepdim=True) * torch.sqrt(k)
    dn = y.size(dim) - 1
    nomin = torch.cat((nomin_ + y.narrow(dim, 0, 1), y.narrow(dim, 1, dn)), dim)
    denom = _norm(nomin, keepdim=True)
    return dist_ * nomin / denom


def logmap0back(x, *, k, dim=-1):
    r"""
    Compute logarithmic map for :math:`0` from :math:`x` on the manifold.

    Parameters
    ----------
    x : tensor
        target point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`0` to :math:`y`
    """
    return _logmap0back(x, k=k, dim=dim)


# @torch.jit.script
def _logmap0back(x, k, dim: int = -1):
    dist_ = _dist0(x, k=k, dim=dim, keepdim=True)
    nomin_ = 1.0 / k * _inner0(x, k=k, keepdim=True) * x
    dn = nomin_.size(dim) - 1
    nomin = torch.cat(
        (nomin_.narrow(dim, 0, 1) + torch.sqrt(k), nomin_.narrow(dim, 1, dn)), dim
    )
    denom = _norm(nomin, keepdim=True)
    return dist_ * nomin / denom


def egrad2rgrad(x, grad, *, k, dim=-1):
    r"""
    Translate Euclidean gradient to Riemannian gradient on tangent space of :math:`x`.

    .. math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathcal{T}_{\mathbf{x}} \mathbb{H}^{d, k}(\mathbf{v})}:=\mathbf{v}+\langle\mathbf{x}, \mathbf{v}\rangle_{\mathcal{L}} \frac{\mathbf{x}}{k}

    Parameters
    ----------
    x : tensor
        point on the Hyperboloid
    grad : tensor
        Euclidean gradient for :math:`x`
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Riemannian gradient :math:`u\in `
    """
    return _egrad2rgrad(x, grad, k=k, dim=dim)


# @torch.jit.script
def _egrad2rgrad(x, grad, k, dim: int = -1):
    grad.narrow(-1, 0, 1).mul_(-1)
    grad = grad.addcmul(_inner(x, grad, dim=dim, keepdim=True), x / k)
    return grad


def parallel_transport(x, y, v, *, k, dim=-1):
    r"""
    Perform parallel transport on the Hyperboloid.

    Parameters
    ----------
    x : tensor
        starting point
    y : tensor
        end point
    v : tensor
        tangent vector to be transported
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        transported vector
    """
    return _parallel_transport(x, y, v, k=k, dim=dim)


# @torch.jit.script
def _parallel_transport(x, y, v, k, dim: int = -1):
    lmap = _logmap(x, y, k=k, dim=dim)
    nom = _inner(lmap, v, keepdim=True)
    denom = _dist(x, y, k=k, dim=dim, keepdim=True) ** 2
    p = v - nom / denom * (lmap + _logmap(y, x, k=k, dim=dim))
    return p


def parallel_transport0(y, v, *, k, dim=-1):
    r"""
    Perform parallel transport from zero point.

    Parameters
    ----------
    y : tensor
        end point
    v : tensor
        tangent vector to be transported
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        transported vector
    """
    return _parallel_transport0(y, v, k=k, dim=dim)


# @torch.jit.script
def _parallel_transport0(y, v, k, dim: int = -1):
    lmap = _logmap0(y, k=k, dim=dim)
    nom = _inner(lmap, v, keepdim=True)
    denom = _dist0(y, k=k, dim=dim, keepdim=True) ** 2
    p = v - nom / denom * (lmap + _logmap0back(y, k=k, dim=dim))
    return p


def parallel_transport0back(x, v, *, k, dim: int = -1):
    r"""
    Perform parallel transport to the zero point.

    Special case parallel transport with last point at zero that
    can be computed more efficiently and numerically stable

    Parameters
    ----------
    x : tensor
        target point
    v : tensor
        vector to be transported
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
    """
    return _parallel_transport0back(x, v, k=k, dim=dim)


# @torch.jit.script
def _parallel_transport0back(x, v, k, dim: int = -1):
    lmap = _logmap0back(x, k=k, dim=dim)
    nom = _inner(lmap, v, keepdim=True)
    denom = _dist0(x, k=k, dim=dim, keepdim=True) ** 2
    p = v - nom / denom * (lmap + _logmap0(x, k=k, dim=dim))
    return p


def geodesic_unit(t, x, u, *, k):
    r"""
    Compute unit speed geodesic at time :math:`t` starting from :math:`x` with direction :math:`u/\|u\|_x`.

    .. math::

        \gamma_{\mathbf{x} \rightarrow \mathbf{u}}^{k}(t)=\cosh \left(\frac{t}{\sqrt{k}}\right) \mathbf{x}+\sqrt{k} \sinh \left(\frac{t}{\sqrt{k}}\right) \mathbf{u}

    Parameters
    ----------
    t : tensor
        travelling time
    x : tensor
        initial point
    u : tensor
        unit direction vector
    k : tensor
        manifold negative curvature

    Returns
    -------
    tensor
        the point on geodesic line
    """
    return _geodesic_unit(t, x, u, k=k)


# @torch.jit.script
def _geodesic_unit(t, x, u, k):
    return (
        torch.cosh(t / torch.sqrt(k)) * x
        + torch.sqrt(k) * torch.sinh(t / torch.sqrt(k)) * u
    )


def lorentz_to_poincare(x, k, dim=-1):
    r"""
    Diffeomorphism that maps from Hyperboloid to Poincare disk.

    .. math::

        \Pi_{\mathbb{H}^{d, 1} \rightarrow \mathbb{D}^{d, 1}\left(x_{0}, \ldots, x_{d}\right)}=\frac{\left(x_{1}, \ldots, x_{d}\right)}{x_{0}+\sqrt{k}}

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        points on the Poincare disk
    """
    dn = x.size(dim) - 1
    return x.narrow(dim, 1, dn) / (x.narrow(-dim, 0, 1) + torch.sqrt(k))


def poincare_to_lorentz(x, k, dim=-1, eps=1e-6):
    r"""
    Diffeomorphism that maps from Poincare disk to Hyperboloid.

    .. math::

        \Pi_{\mathbb{D}^{d, k} \rightarrow \mathbb{H}^{d d, 1}}\left(x_{1}, \ldots, x_{d}\right)=\frac{\sqrt{k} \left(1+|| \mathbf{x}||_{2}^{2}, 2 x_{1}, \ldots, 2 x_{d}\right)}{1-\|\mathbf{x}\|_{2}^{2}}

    Parameters
    ----------
    x : tensor
        point on Poincare ball
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        points on the Hyperboloid
    """
    x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
    res = (
        torch.sqrt(k)
        * torch.cat((1 + x_norm_square, 2 * x), dim=dim)
        / (1.0 - x_norm_square + eps)
    )
    return res


class LorentzCalculation:
    def __init__(self):
        self.name = "Hyperboloid Network"

    def proj_tan0(self, u):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

    def expmap0(self, v, clip_r, c, alpha_scaler=None):
        if alpha_scaler is not None:
            v = v * torch.exp(alpha_scaler)
        k = 1. / c
        # v is in the tangent space
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        v_clipped = torch.minimum(torch.ones_like(v_norm), clip_r / v_norm) * v  # clipped embeddings v_norm
        x = expmap0(v_clipped, k=k)
        return x

    def lorentz_dot(self, x, y):
        # BD, KD -> BK
        # minkowski product, one2all
        result_space = torch.matmul(x[:, 1:], y[:, 1:].T)  # BD,KD->BK
        result_time = torch.matmul(x[:, 0].view(-1, 1), y[:, 0].view(1, -1))  # B1,1K->BK
        result = result_space - result_time
        return result

    def lorentz_dot_o2o(self, x, y):
        # BD, BD -> B1
        m = x * y
        result = m[:, 1:].sum(dim=1) - m[:, 0]
        return result.reshape(-1, 1)

    def lorentz_dist(self, x, y, c):
        # BD,KD -> BK
        k = 1. / c
        prod = self.lorentz_dot(x, y)
        assert not torch.any(-prod / k < 0.99999)
        # dist = torch.sqrt(k) * math_util.arcosh(-prod/k)
        return torch.sqrt(k) * arcosh(-prod / k)

    def sqrt_lorentz_dist(self, x, y, c):
        # BD, KD -> BK
        k = 1. / c
        prod = self.lorentz_dot(x, y)
        return -2 * k - 2 * prod

    def mid_point(self, prob, x, c):
        # BK, KD -> BD
        k = 1. / c
        avg = torch.matmul(prob, x)  # BD
        denom = -self.lorentz_dot_o2o(avg, avg)  # B1
        denom = denom.abs().clamp_min(1e-8).sqrt()
        centroid = torch.sqrt(k) * avg / denom
        return centroid

    def mid_point_for_softsort(self, P_hat, x, c):
        # BKK, KD -> BKD
        k = 1. / c
        avg = torch.einsum("bhk,kd->bhd", P_hat, x)  # BKD
        dot_avg = avg * avg  # BKD
        denom = (dot_avg[:, :, 1:].sum(dim=-1) - dot_avg[:, :, 0])  # BK
        denom = denom.abs().clamp_min_(1e-8).sqrt()
        centroid = torch.sqrt(k) * avg / (denom.unsqueeze(-1))
        return centroid  # BKD

    def lorentz_dist_for_softsort(self, z_i, z_j_cwd, c):
        # z_i: [B,D]
        # z_j_cwd: [B,K,D]
        # return [B,K]
        def lorentz_dot_for_softsort(x, y):
            # BD, BKD -> BK
            x_space = x[:, 1:].unsqueeze(dim=-1)  # [B,1, D-1]
            y_space = y[:, :, 1:]  # [B,K, D-1]
            result_space = torch.bmm(y_space, x_space).squeeze()  # [B,K]
            x_time = x[:, 0].unsqueeze(-1)  # [B,1]
            y_time = y[:, :, 0]  # [B,K]
            result_time = x_time * y_time  # broadcast
            result = result_space - result_time
            return result

        k = 1. / c
        prod = lorentz_dot_for_softsort(z_i, z_j_cwd)
        assert not torch.any(-prod / k < 0.99999)
        # dist = torch.sqrt(k) * math_util.arcosh(-prod/k)
        return torch.sqrt(k) * arcosh(-prod / k)


class FullHyperPQhead(nn.Module):
    def __init__(self, feat_dim, M, K, softmax_temp, quant_method,
                 init_neg_curvs=1.0, clip_r=1.0, use_alpha=False, writer=None):
        # MlogK bits
        # M = M; number of codebooks
        # K = K; number of codewords
        # D: D; dimensions of the codewords
        super(FullHyperPQhead, self).__init__()
        self.feat_dim, self.M, self.K, self.D = feat_dim, M, K, feat_dim // M
        self.softmax_temp, self.quant_method = softmax_temp, quant_method
        self.lorentz_calculator = LorentzCalculation()
        self.C = nn.Parameter(torch.empty((self.K, self.M * self.D)), requires_grad=True)  # in Ambient Space Originally
        self.clip_r = nn.Parameter(torch.tensor(clip_r), requires_grad=False)
        nn.init.xavier_uniform_(self.C.data)
        self.writer = writer
        self.global_step = 0

        if use_alpha:
            self.alpha_scaler = nn.Parameter(torch.log(torch.sqrt(torch.Tensor([1. / self.D] * M))),
                                             requires_grad=False)
        else:
            self.alpha_scaler = None
        # self.use_soft_sort = use_soft_sort
        # if self.use_soft_sort:
        #     raise NotImplementedError("Not implemented")
        #     self.soft_sort = SoftSort(tau=tau)

        if isinstance(init_neg_curvs, float):
            tmp_neg_curvs = torch.Tensor([init_neg_curvs] * M)
            self.neg_curvs = nn.Parameter(tmp_neg_curvs, requires_grad=True)
        elif isinstance(init_neg_curvs, list):
            tmp_neg_curvs = torch.Tensor(init_neg_curvs)
            self.neg_curvs = nn.Parameter(tmp_neg_curvs, requires_grad=True)

    def quant(self, i, xi, ci, i_neg_curvs):
        # x[i] is in the lorentz sapce now
        # c[i] is in the tangent space now

        # logits: [bsz, K] unnormalized log weights in lorentz space
        # logits = self.lorentz_calculator.lorentz_simlarity(xi, ci, i_neg_curvs)
        # calcualte the squared lorentzian distance
        sqrt_dist = self.lorentz_calculator.sqrt_lorentz_dist(xi, ci, i_neg_curvs)
        logits = -sqrt_dist

        if self.quant_method == "softmax":
            # soft_prob: [bsz, K]
            soft_prob = F.softmax(logits * self.softmax_temp, dim=1)
            xi_hat = self.lorentz_calculator.mid_point(soft_prob, ci, i_neg_curvs)
            if self.writer is not None:
                max_val, _ = torch.max(logits, dim=1)
                self.writer.add_scalar('max_val_of_logits_%d' % i, torch.mean(max_val), self.global_step)
                max_val_prob, _ = torch.max(soft_prob, dim=1)
                self.writer.add_scalar('max_val_of_prob_%d' % i, torch.mean(max_val_prob), self.global_step)
        else:
            raise NotImplementedError("Wrong Quantization Method.")

        return xi_hat, logits

    def encode_hyper_feats(self, x):
        x = torch.split(x, self.D, dim=1)  # [[B, D]] * M
        tan_x = [self.lorentz_calculator.proj_tan0(xi) for xi in x]
        hyper_x = [
            self.lorentz_calculator.expmap0(v=tan_x[i], clip_r=self.clip_r, c=self.neg_curvs[i], alpha_scaler=None) for
            i in range(len(tan_x))]
        hyper_x = torch.stack(hyper_x, dim=-1)
        hyper_x = torch.transpose(hyper_x, 1, 2)  # [b, M, D]
        return hyper_x

    def encode_tangent_feats(self, x):
        x = torch.split(x, self.D, dim=1)  # [[B, D]] * M
        tan_x = [self.lorentz_calculator.proj_tan0(xi) for xi in x]
        tan_x = torch.stack(tan_x, dim=-1)
        tan_x = torch.transpose(tan_x, 1, 2)  # [b, M, D]
        tan_x = tan_x.reshape(tan_x.shape[0], - 1)  # [b, M*D]
        return tan_x

    def tangent_to_hyper(self, x):
        # x: [b, M*D] in tangent space
        tan_x = torch.split(x, self.D, dim=1)  # [[B, D]] * M
        hyper_x = [
            self.lorentz_calculator.expmap0(v=tan_x[i], clip_r=self.clip_r, c=self.neg_curvs[i], alpha_scaler=None) for
            i in range(len(tan_x))]
        hyper_x = torch.stack(hyper_x, dim=-1)
        hyper_x = torch.transpose(hyper_x, 1, 2)  # [b, M, D]
        return hyper_x

    @torch.no_grad()
    def _codebook_normalization(self):
        # normalize the codewords
        codewords = self.C.data.clone()
        codewords = codewords.view(self.K, self.M, self.D)
        codewords = F.normalize(codewords, dim=-1)
        codewords = codewords.view(self.K, self.M * self.D)
        self.C.copy_(codewords)

    def forward(self, x):
        # tuple; ele of the tuple has the shape like (bsz, D), M element
        x = torch.split(x, self.D, dim=1)
        # tuple; ele of the tuple has the shape like (K, D), M elements.
        c = torch.split(self.C, self.D, dim=1)
        self.global_step += 1

        x_hat = []
        codes = []
        soft_codes = []
        quant_err = []
        hyper_x = []  # test
        # v_norm_all = []

        # if self.use_soft_sort:
        #     softsort_cwd = []
        for i in range(self.M):
            # first transform xi and ci into tangent space
            if self.writer is not None:
                self.writer.add_scalar("curvature_%d" % i, self.neg_curvs[i], self.global_step)
            xi = x[i]
            ci = c[i]
            xi_tan0 = self.lorentz_calculator.proj_tan0(xi)
            xi_hyper = self.lorentz_calculator.expmap0(v=xi_tan0, clip_r=self.clip_r, c=self.neg_curvs[i],
                                                       alpha_scaler=None)
            ci_tan0 = self.lorentz_calculator.proj_tan0(ci)
            ci_hyper = self.lorentz_calculator.expmap0(v=ci_tan0, clip_r=self.clip_r, c=self.neg_curvs[i],
                                                       alpha_scaler=None)

            # v_norm_i = torch.norm(xi_tan0, dim=-1, keepdim=True)
            # #v_norm_i = torch.minimum(torch.ones_like(v_norm_i), self.clip_r/v_norm_i)*v_norm_i # clipped embeddings
            # v_norm_all.append(v_norm_i)
            xi_hat_hyper, logits_i, = self.quant(i, xi_hyper, ci_hyper, self.neg_curvs[i])
            codes_i = logits_i.argmax(dim=1, keepdim=True)
            soft_codes_i = F.softmax(logits_i * self.softmax_temp, dim=1)

            # TODO. Calculate the quantization error.
            quanti_err = self.lorentz_calculator.lorentz_dist(xi_hyper, xi_hat_hyper, self.neg_curvs[i])  # BD,BD->BB
            quanti_err = torch.diag(quanti_err)  # B

            x_hat.append(xi_hat_hyper)
            codes.append(codes_i)
            soft_codes.append(soft_codes_i)
            quant_err.append(quanti_err)
            hyper_x.append(xi_hyper)
            # if softsort_cwd_i != None:
            #     softsort_cwd.append(softsort_cwd_i)

        hyper_x = torch.stack(hyper_x, dim=-1)
        hyper_x = torch.transpose(hyper_x, 1, 2)
        x_hat = torch.stack(x_hat, dim=-1)  # [b, D, M]
        x_hat = torch.transpose(x_hat, 1, 2)  # [b, M, D]
        codes = torch.cat(codes, dim=1)  # [b, M]
        soft_codes = torch.stack(soft_codes, dim=-1)  # [b, D, M]
        soft_codes = torch.transpose(soft_codes, 1, 2)  # [b, M, D]
        quant_err = torch.mean(torch.stack(quant_err, 0)).detach()
        # v_norm_all = torch.cat(v_norm_all, dim=1) # [b,M]

        return hyper_x, x_hat, codes, soft_codes, quant_err

    def hyper_codebooks(self):
        C = torch.split(self.C, self.D, dim=1)
        hyper_C = []
        for i in range(len(C)):
            ci = C[i]
            ci = self.lorentz_calculator.proj_tan0(ci)
            hyper_ci = self.lorentz_calculator.expmap0(v=ci, clip_r=self.clip_r, c=self.neg_curvs[i], alpha_scaler=None)
            hyper_C.append(hyper_ci)
        hyper_C = torch.stack(hyper_C, dim=1)
        hyper_C = hyper_C.view(hyper_C.shape[0], -1)  # (K, M*D)
        return hyper_C.detach()

    def save_codebooks(self, path):
        # first transform the codewords to hyperbolic space, then save
        C = torch.split(self.C, self.D, dim=1)
        hyper_C = []
        for i in range(C):
            ci = C[i]
            ci = self.lorentz_calculator.proj_tan0(ci)
            hyper_ci = self.lorentz_calculator.expmap0(v=ci, clip_r=self.clip_r, c=self.neg_curvs[i], alpha_scaler=None)
            hyper_C.append(hyper_ci)
        with open(path, 'wb') as f:

            np.save(f, hyper_C.detach().cpu().numpy())
