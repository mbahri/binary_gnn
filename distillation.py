import torch
import torch.nn.functional as F
import torch_scatter as ts


def hamming_relaxed(z_i, z_j):
    "Continuous relaxation of the Hamming distance for vectors valued in F{-1,1}"
    return 0.5 * ( -z_i * z_j + 1).sum(-1)


def rbf_hamming(z_i, z_j):
    return torch.exp(-hamming_relaxed(z_i, z_j))


def rbf_hamming_sq(z_i, z_j):
    return torch.exp(-hamming_relaxed(z_i, z_j)**2)


def l2(z_i, z_j):
    return torch.cdist(z_i.unsqueeze(1), z_j.unsqueeze(1), p=2).squeeze()**2


def rbf_l2(z_i, z_j):
    return torch.exp(-l2(z_i, z_j))


def linear(z_i, z_j):
    return torch.sum(z_i * z_j, dim=-1)


def poly(z_i, z_j, c=0, d=2):
    return (linear(z_i, z_j) + c)**d


class StructuralSimilarity(torch.nn.Module):
    def __init__(self, kernel_s=lambda x: x, kernel_t=lambda x: x, same_ei=False):
        super().__init__()
        self.kernel_s = kernel_s
        self.kernel_t = kernel_t
        self.same_ei = same_ei

    def compute_ls_static(self, z, e, student=True):
        if student:
            kernel = self.kernel_s
        else:
            kernel = self.kernel_t

        sims = kernel(z[e[0]], z[e[1]])
        ls_flat = ts.scatter_log_softmax(sims, e[0])

        return ls_flat

    def compute_ls_both(self, z_s, z_t, e_s, e_t):
        if not self.same_ei:
            e_u = torch.unique(torch.cat((e_s, e_t), dim=1), dim=1)
        else:
            e_u = e_t

        ls_s = self.compute_ls_static(z_s, e_u, student=True)
        ls_t = self.compute_ls_static(z_t, e_u, student=False)

        return ls_s, ls_t, e_u

    def forward(self, z_s, z_t, e_s, e_t):
        ls_s, ls_t, e_u = self.compute_ls_both(z_s, z_t, e_s, e_t)

        return ts.scatter_sum(
            F.kl_div(ls_s, ls_t, reduction='none', log_target=True),
            e_u[0]
        ).mean()


class CrossEntropyWithTemperature(torch.nn.Module):
    def __init__(self, T=1.0):
        super().__init__()
        self.T = T

    def forward(self, z_s, z_t):
        "z_s: logits of student, z_t: logits of teacher"
        ls_s = F.log_softmax(z_s / self.T, dim=1)
        ls_t = F.log_softmax(z_t / self.T, dim=1)

        return self.T**2 * F.kl_div(ls_s, ls_t, log_target=True, reduction='batchmean')


def get_att_vector(x):
        x_vol = torch.norm(x, dim=-1)**2
        x_vol = x_vol / torch.norm(x_vol, dim=-1, keepdim=True)
        return x_vol