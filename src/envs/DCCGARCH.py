from scipy.stats import norm, t as t_dist
from scipy.special import gammaln
import numpy as np
from scipy.optimize import minimize
import torch

class DCCGARCHSimulator:
    def __init__(self, params, Q_bar, dcc_alpha, dcc_beta, nu_Q, r_daily=0.04/252, device='cuda'):
        """
        Args:
            params: Dict of GARCH parameters for each asset
            Q_bar: Unconditional correlation matrix (will be converted to tensor)
            dcc_alpha: DCC parameter α (will be converted to tensor)
            dcc_beta: DCC parameter β (will be converted to tensor)
            nu_Q: Student-t degrees of freedom (will be converted to tensor)
            r_daily: Daily risk-free rate (will be converted to tensor)
            device: 'cuda' or 'cpu'
        """
        self.params = params
        self.n_assets = len(params)
        self.device = device

        self.Q_bar = torch.tensor(Q_bar, dtype=torch.float32, device=device)
        self.dcc_alpha = torch.tensor(dcc_alpha, dtype=torch.float32, device=device)
        self.dcc_beta = torch.tensor(dcc_beta, dtype=torch.float32, device=device)
        self.nu_Q = torch.tensor(nu_Q, dtype=torch.float32, device=device)
        self._build_t_cdf_lut(float(nu_Q))
        self.r_daily = torch.tensor(r_daily, dtype=torch.float32, device=device)


        self.omega = torch.tensor([params[i]['omega'] for i in range(self.n_assets)],
                                   dtype=torch.float32, device=device)
        self.alpha_garch = torch.tensor([params[i]['alpha'] for i in range(self.n_assets)],
                                        dtype=torch.float32, device=device)
        self.beta_garch = torch.tensor([params[i]['beta'] for i in range(self.n_assets)],
                                       dtype=torch.float32, device=device)
        self.gamma = torch.tensor([params[i]['gamma'] for i in range(self.n_assets)],
                                  dtype=torch.float32, device=device)
        self.lambda_ = torch.tensor([params[i]['lambda'] for i in range(self.n_assets)],
                                    dtype=torch.float32, device=device)
        self.h_unconditional = torch.tensor(
            [params[i]['h_unconditional'] for i in range(self.n_assets)],
            dtype=torch.float32, device=device
        )
        pred = (self.omega + self.alpha_garch) / (
            1 - self.beta_garch - self.alpha_garch * self.gamma**2
        )


        print("predicted h_unconditional")
        for x in pred:
            print(f"{x.item():.10f}")

        print("true h_unconditional")
        for x in self.h_unconditional:
            print(f"{x.item():.10f}")
    def _sample_t_copula_gaussian_margins(self, B, n, L_chol, device, generator=None):
        L_chol = L_chol.to(device)
        nu = float(self.nu_Q.item())

        g = torch.randn(B, n, device=device, generator=generator)
        g = torch.bmm(L_chol, g.unsqueeze(-1)).squeeze(-1)

        half_nu = torch.tensor(nu / 2.0, device=device)
        chi2 = 2.0 * torch._standard_gamma(half_nu.expand(B), generator=generator)
        X = g / torch.sqrt(chi2 / nu).unsqueeze(-1)

        U = self._t_cdf_gpu(X).clamp(1e-6, 1 - 1e-6)   # GPU-only, no CPU roundtrip
        Z = torch.distributions.Normal(0., 1.).icdf(U)

        return Z, X
    def _build_t_cdf_lut(self, nu, grid_size=50_000, clip=1e-6, margin=1.0):
        """
        One-time (~10ms CPU) table build. Range comes from the inverse CDF. 
        Anything past [x_lo, x_hi] gets hard-clipped to
        clip / 1-clip
        """
        x_lo = float(t_dist.ppf(clip, df=nu))
        x_hi = float(t_dist.ppf(1 - clip, df=nu))
        x_grid = np.linspace(x_lo - margin, x_hi + margin, grid_size)
        cdf_grid = t_dist.cdf(x_grid, df=nu)
        self._t_lut_x = torch.tensor(x_grid, dtype=torch.float32, device=self.device)
        self._t_lut_cdf = torch.tensor(cdf_grid, dtype=torch.float32, device=self.device)
        self._t_lut_xlo, self._t_lut_xhi = x_lo, x_hi
        self._t_clip_lo, self._t_clip_hi = clip, 1 - clip

    def _t_cdf_gpu(self, X):
        """GPU-only replacement for scipy.stats.t.cdf(X, df=nu)."""
        flat = X.reshape(-1)
        xc = flat.clamp(self._t_lut_xlo, self._t_lut_xhi)
        idx = torch.searchsorted(self._t_lut_x, xc).clamp(1, self._t_lut_x.numel() - 1)
        x0, x1 = self._t_lut_x[idx - 1], self._t_lut_x[idx]
        y0, y1 = self._t_lut_cdf[idx - 1], self._t_lut_cdf[idx]
        U = y0 + (xc - x0) / (x1 - x0) * (y1 - y0)
        U = torch.where(flat <= self._t_lut_xlo, torch.full_like(U, self._t_clip_lo), U)
        U = torch.where(flat >= self._t_lut_xhi, torch.full_like(U, self._t_clip_hi), U)
        return U.reshape(X.shape)
