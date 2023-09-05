import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from modules.stats import diag_gaussian_log_prob
from modules.nn_utils import MLP


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.mu_net = MLP(input_dim, hidden_dims, output_dim)
        self.logvar_net = MLP(input_dim, hidden_dims, output_dim)

    def forward(self, *args):
        return self.mu_net(*args), F.softplus(self.logvar_net(*args))


class Vae(nn.Module):
    def __init__(self, x_dim, h_dims, z_dim, y_dim, n_components, n_samples, log_p_y_xz_fn):
        super().__init__()
        self.n_samples = n_samples
        self.log_p_y_xz_fn = log_p_y_xz_fn
        self.q_z_xy_net = GaussianMLP(x_dim + y_dim, h_dims, z_dim)
        self.p_y_xz_net = MLP(x_dim + z_dim, h_dims, y_dim)
        self.logits_c = nn.Parameter(torch.ones(n_components))
        self.mu_z_c = nn.Parameter(torch.zeros(n_components, z_dim))
        self.logvar_z_c = nn.Parameter(torch.zeros(n_components, z_dim))
        nn.init.xavier_normal_(self.mu_z_c)
        nn.init.xavier_normal_(self.logvar_z_c)


    def sample_z(self, mu, var):
        sd = var.sqrt()
        eps = torch.randn(self.n_samples, *sd.shape).to(sd.get_device())
        return mu + eps * sd


    def forward(self, x, y_true):
        batch_size = len(x) # For assertions
        # z ~ q(z|x,y)
        mu_z_xy, var_z_xy = self.q_z_xy_net(x, y_true)
        z = self.sample_z(mu_z_xy, var_z_xy)
        log_q_z_xy = diag_gaussian_log_prob(z, mu_z_xy, var_z_xy).view(-1)
        assert log_q_z_xy.shape == (self.n_samples * batch_size,)
        # E_q(z|x,y)[log p(y|x,z)]
        x = torch.repeat_interleave(x[None], repeats=self.n_samples, dim=0)
        y_true = torch.repeat_interleave(y_true[None], repeats=self.n_samples, dim=0)
        x, y_true, z = x.view(-1, x.shape[-1]), y_true.view(-1, y_true.shape[-1]), z.view(-1, z.shape[-1])
        logits_y_xz = self.p_y_xz_net(x, z)
        log_p_y_xz = self.log_p_y_xz_fn(logits_y_xz, y_true)
        assert log_p_y_xz.shape == (self.n_samples * batch_size,)
        # KL(q(z|x,y) || p(z))
        dist_c = D.Categorical(logits=self.logits_c)
        var_z_c = F.softplus(self.logvar_z_c)
        dist_z_c = D.Independent(D.Normal(self.mu_z_c, var_z_c.sqrt()), 1)
        dist_z = D.MixtureSameFamily(dist_c, dist_z_c)
        log_p_z = dist_z.log_prob(z)
        assert log_p_z.shape == (self.n_samples * batch_size,)
        kl = (log_q_z_xy - log_p_z).mean()
        elbo = log_p_y_xz.mean() - kl
        logits_y_xz = logits_y_xz.view((self.n_samples, batch_size, -1))
        return {
            "loss": -elbo,
            "kl": kl,
            "logits": logits_y_xz[0]
        }


class VanillaVAE(nn.Module):
    def __init__(self, x_dim, h_dims, z_dim, y_dim, n_samples, log_p_y_xz_fn):
        super().__init__()
        self.n_samples = n_samples
        self.log_p_y_xz_fn = log_p_y_xz_fn
        self.q_z_xy_net = GaussianMLP(x_dim + y_dim, h_dims, z_dim)
        self.p_y_xz_net = MLP(x_dim + z_dim, h_dims, y_dim, nn.Sigmoid)


    def sample_z(self, mu, var):
        sd = var.sqrt()
        eps = torch.randn(self.n_samples, *sd.shape).to(sd.get_device())
        return mu + eps * sd


    def forward(self, x, y):
        batch_size = len(x) # For assertions
        # z ~ q(z|x,y)
        mu_z_xy, var_z_xy = self.q_z_xy_net(x, y)
        z = self.sample_z(mu_z_xy, var_z_xy)
        log_q_z_xy = diag_gaussian_log_prob(z, mu_z_xy, var_z_xy).view(-1)
        assert log_q_z_xy.shape == (self.n_samples * batch_size,)
        # E_q(z|x,y)[log p(y|x,z)]
        x = torch.repeat_interleave(x[None], repeats=self.n_samples, dim=0)
        y = torch.repeat_interleave(y[None], repeats=self.n_samples, dim=0)
        x, y, z = x.view(-1, x.shape[-1]), y.view(-1, y.shape[-1]), z.view(-1, z.shape[-1])
        logits_y_xz = self.p_y_xz_net(x, z)
        log_p_y_xz = self.log_p_y_xz_fn(logits_y_xz, y)
        assert log_p_y_xz.shape == (self.n_samples * batch_size,)
        # KL(q(z|x,y) || p(z))
        kl = (-0.5 * torch.sum(1 + var_z_xy.log() - mu_z_xy.pow(2) - var_z_xy, dim=1)).mean()
        elbo = log_p_y_xz.mean() - kl
        logits_y_xz = logits_y_xz.view((self.n_samples, batch_size, -1))
        return {
            "loss": -elbo,
            "kl": kl,
            "logits": logits_y_xz[0]
        }