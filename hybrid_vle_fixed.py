# hybrid_vle_fixed.py (shape-robust version with extra debug)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ---------------- Config ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.manual_seed(0)
np.random.seed(0)

n_comp = 13
state_dim = 1 + 2 * n_comp
res_dim = 1 + n_comp
batch_size = 16

# ---------------- Synthetic dataset ----------------
class VLE_Synth(Dataset):
    def __init__(self, n_samples=256, n_comp=n_comp):
        super().__init__()
        self.n_comp = n_comp
        N = n_samples
        T = np.random.uniform(300.0, 500.0, size=(N,1))
        P = np.random.uniform(1.0, 20.0, size=(N,1))
        raw_z = np.random.rand(N, n_comp)
        z = raw_z / raw_z.sum(axis=1, keepdims=True)
        logK = np.random.randn(N, n_comp) * 0.5
        K = np.exp(logK)
        beta = 0.4 + 0.2 * np.random.rand(N,1)
        denom = beta + (1.0 - beta) * K
        x = z / denom
        x = x / x.sum(axis=1, keepdims=True)
        y = K * x
        y = y / y.sum(axis=1, keepdims=True)

        self.T = torch.tensor(T, dtype=dtype, device=device)
        self.P = torch.tensor(P, dtype=dtype, device=device)
        self.z = torch.tensor(z, dtype=dtype, device=device)
        self.beta = torch.tensor(beta, dtype=dtype, device=device)
        self.x = torch.tensor(x, dtype=dtype, device=device)
        self.y = torch.tensor(y, dtype=dtype, device=device)

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        return {
            "T": self.T[idx],   # shape (1,)
            "P": self.P[idx],   # shape (1,)
            "z": self.z[idx],   # shape (n_comp,)
            "beta": self.beta[idx], # shape (1,)
            "x": self.x[idx],   # shape (n_comp,)
            "y": self.y[idx],   # shape (n_comp,)
        }

# ---------------- Surrogate ----------------
class CompactSurrogate(nn.Module):
    def __init__(self, n_comp, hidden=128):
        super().__init__()
        in_dim = n_comp + 2
        out_dim = 1 + 2 * n_comp
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
        self.n_comp = n_comp

    def forward(self, z, T, P):
        # Expect z:(B,n), T:(B,1) or (B,), P same
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if T.ndim == 1:
            T = T.unsqueeze(0)
        if P.ndim == 1:
            P = P.unsqueeze(0)
        inp = torch.cat([z, T, P], dim=-1)  # (B, n+2)
        out = self.net(inp)                 # (B, 1+2n)
        beta = out[:, 0:1]                  # keep shape (B,1)
        comps = out[:, 1:].view(out.shape[0], 2, self.n_comp)  # (B,2,n)
        x_logits = comps[:,0,:]             # (B,n)
        y_logits = comps[:,1,:]             # (B,n)
        x = F.softmax(x_logits, dim=-1)     # (B,n)
        y = F.softmax(y_logits, dim=-1)     # (B,n)
        return beta, x, y

# ---------------- Residual (safe) ----------------
def vle_residual_state_from_tuple(s_tuple, z):
    # s_tuple: (beta (B,1), x (B,n), y (B,n))
    beta, x, y = s_tuple
    # enforce shapes
    assert beta.ndim == 2 and beta.shape[1] == 1, f"beta must be (B,1), got {beta.shape}"
    assert x.ndim == 2 and y.ndim == 2, f"x,y must be (B,n), got {x.shape}, {y.shape}"
    B = x.shape[0]
    n = x.shape[1]
    assert z.ndim == 2 and z.shape[1] == n, f"z must be (B,n) with same n, got {z.shape}"

    # For numerical robustness, clamp small values
    x_safe = x.clamp(min=1e-12)
    y_safe = y.clamp(min=1e-12)
    K = (y_safe / x_safe).clamp(min=1e-12)            # (B,n)
    denom = (1.0 + beta * (K - 1.0)).clamp(min=1e-12) # beta shape (B,1) broadcasts to (B,n)
    rr = (z * (K - 1.0) / denom).sum(dim=-1, keepdim=True)  # (B,1)
    zrec = beta * x + (1.0 - beta) * y                  # (B,n) uses broadcasting
    mb = zrec - z                                       # (B,n)
    # flatten residual: first rr then mb components
    res = torch.cat([rr, mb], dim=-1)  # (B, 1+n)
    return res

# ---------------- Preconditioner ----------------
class LearnedPreconditioner(nn.Module):
    def __init__(self, res_dim, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(res_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )
    def forward(self, r):
        # r: (B, res_dim)
        return self.net(r)

# ---------------- Solve linear system J * delta = -r (naive autograd) ----------------
def solve_linear_system_J(s, z, r):
    # s: (B, nstate), z: (B,n), r: (B, m)
    # Naive autograd jacobian per sample (ok for small dims / demo)
    s_var = s.clone().detach().requires_grad_(True)
    r_s = vle_residual_state_from_tuple((s_var[:,0:1], s_var[:,1:1+n_comp], s_var[:,1+n_comp:1+2*n_comp]), z)
    B, m = r_s.shape
    n = s_var.shape[1]
    deltas = []
    for i in range(B):
        si = s_var[i:i+1, :]  # (1,n)
        ri = r_s[i]           # (m,)
        J = torch.zeros((m, n), dtype=s_var.dtype, device=s_var.device)
        for j in range(m):
            grad_j = torch.autograd.grad(ri[j], si, retain_graph=True, create_graph=False, allow_unused=True)[0]
            if grad_j is None:
                grad_j = torch.zeros_like(si)
            J[j, :] = grad_j.view(-1)
        rhs = -r[i].unsqueeze(-1)  # (m,1)
        # solve least squares J delta = rhs
        try:
            sol = torch.linalg.lstsq(J, rhs).solution.squeeze(-1)
        except Exception:
            sol_np = np.linalg.lstsq(J.cpu().numpy(), rhs.cpu().numpy(), rcond=None)[0].squeeze(-1)
            sol = torch.tensor(sol_np, dtype=s_var.dtype, device=s_var.device)
        deltas.append(sol)
    return torch.stack(deltas, dim=0)  # (B, n)

# ---------------- Hybrid solver using learned preconditioner ----------------
def hybrid_solve_case(surrogate, precond, z, T, P, max_iters=6, tol=1e-6, verbose=False):
    # Ensure inputs have batch dim
    if z.ndim == 1:
        z = z.unsqueeze(0)
    if T.ndim == 1:
        T = T.unsqueeze(0)
    if P.ndim == 1:
        P = P.unsqueeze(0)

    beta, x, y = surrogate(z, T, P)  # beta: (B,1), x/y: (B,n)
    if verbose:
        print("Initial shapes -> beta:", beta.shape, "x:", x.shape, "y:", y.shape, "z:", z.shape)

    s_beta = beta  # (B,1)
    s_x = x        # (B,n)
    s_y = y        # (B,n)

    for it in range(max_iters):
        s_tuple = (s_beta, s_x, s_y)
        r = vle_residual_state_from_tuple(s_tuple, z)  # (B, 1+n)
        res_norm = r.abs().mean().item()
        if verbose:
            print(f"Iter {it}: residual mean {res_norm:.6e}; shapes r:{r.shape}")
        if res_norm < tol:
            return (s_beta, s_x, s_y), True, it

        # Preconditioner expects residual vector r (B, m) -> predict state-delta (B, state_dim)
        delta = precond(r)  # (B, state_dim)
        # enforce shapes
        assert delta.ndim == 2 and delta.shape[0] == s_beta.shape[0] and delta.shape[1] == state_dim, \
            f"delta shape unexpected: {delta.shape}"

        # Apply update: s_new = s + delta
        s_vec = torch.cat([s_beta, s_x, s_y], dim=-1)  # (B, state_dim)
        s_vec = s_vec + delta
        # Unpack and reproject to valid ranges
        s_beta = s_vec[:, 0:1]            # (B,1)
        s_x = F.softmax(s_vec[:, 1:1+n_comp], dim=-1)  # (B,n)
        s_y = F.softmax(s_vec[:, 1+n_comp:1+2*n_comp], dim=-1)  # (B,n)

        # Debug shape check
        if verbose:
            print(f" After update shapes -> beta: {s_beta.shape}, x: {s_x.shape}, y: {s_y.shape}")

    # final residual check
    r = vle_residual_state_from_tuple((s_beta, s_x, s_y), z)
    return (s_beta, s_x, s_y), (r.abs().mean().item() < tol), max_iters

# ---------------- Full Newton baseline (autograd) ----------------
def full_newton_solve(surrogate, z, T, P, max_iters=10, tol=1e-6, verbose=False):
    if z.ndim == 1:
        z = z.unsqueeze(0)
    if T.ndim == 1:
        T = T.unsqueeze(0)
    if P.ndim == 1:
        P = P.unsqueeze(0)

    beta, x, y = surrogate(z, T, P)
    s_vec = torch.cat([beta, x, y], dim=-1)  # (B, state_dim)

    for it in range(max_iters):
        s_tuple = (s_vec[:,0:1], s_vec[:,1:1+n_comp], s_vec[:,1+n_comp:1+2*n_comp])
        r = vle_residual_state_from_tuple(s_tuple, z)
        res_norm = r.abs().mean().item()
        if verbose:
            print(f"FullNewton Iter {it}: residual mean {res_norm:.6e}")
        if res_norm < tol:
            return (s_vec[:,0:1], s_vec[:,1:1+n_comp], s_vec[:,1+n_comp:1+2*n_comp]), True, it

        # solve for delta
        delta = solve_linear_system_J(s_vec, z, r)  # (B, state_dim)
        s_vec = s_vec + delta
        # projection
        s_vec[:,1:1+n_comp] = F.softmax(s_vec[:,1:1+n_comp], dim=-1)
        s_vec[:,1+n_comp:1+2*n_comp] = F.softmax(s_vec[:,1+n_comp:1+2*n_comp], dim=-1)
        s_vec[:,0:1] = s_vec[:,0:1].clamp(1e-8, 1-1e-8)

    r = vle_residual_state_from_tuple((s_vec[:,0:1], s_vec[:,1:1+n_comp], s_vec[:,1+n_comp:1+2*n_comp]), z)
    return (s_vec[:,0:1], s_vec[:,1:1+n_comp], s_vec[:,1+n_comp:1+2*n_comp]), (r.abs().mean().item() < tol), max_iters

# ---------------- Training pipeline (surrogate + precond) ----------------
def train_surrogate_and_precond():
    ds = VLE_Synth(n_samples=512, n_comp=n_comp)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    surrogate = CompactSurrogate(n_comp=n_comp, hidden=128).to(device)
    precond = LearnedPreconditioner(res_dim, state_dim, hidden=256).to(device)

    opt_s = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
    opt_p = torch.optim.Adam(precond.parameters(), lr=1e-3)

    print("Training surrogate (quick)...")
    for epoch in range(4):
        tot = 0.0
        nstep = 0
        for batch in dl:
            z = batch["z"]
            T = batch["T"]
            P = batch["P"]
            beta_t = batch["beta"]
            x_t = batch["x"]
            y_t = batch["y"]
            # reshape to (B,n) etc
            beta_pred, x_pred, y_pred = surrogate(z, T, P)
            loss = F.mse_loss(beta_pred, beta_t) + F.mse_loss(x_pred, x_t) + F.mse_loss(y_pred, y_t)
            opt_s.zero_grad(); loss.backward(); opt_s.step()
            tot += loss.item(); nstep += 1
        print(f"Surrogate epoch {epoch}, loss {tot/max(1,nstep):.4e}")

    # Build preconditioner training data (r -> delta_ref)
    print("Preparing preconditioner targets (may be slow on CPU).")
    Rs = []
    Deltas = []
    for batch in dl:
        z = batch["z"]
        beta_t = batch["beta"]
        x_t = batch["x"]
        y_t = batch["y"]
        s_true = torch.cat([beta_t, x_t, y_t], dim=-1)  # (B, state_dim)
        # create perturbed start
        s0 = s_true + 0.01 * torch.randn_like(s_true)
        # compute residual r0 (requires tuple form)
        s0_tuple = (s0[:,0:1], s0[:,1:1+n_comp], s0[:,1+n_comp:1+2*n_comp])
        r0 = vle_residual_state_from_tuple(s0_tuple, z).detach()
        # compute delta_ref solved from J(s0) delta = -r0 (naive)
        delta_ref = solve_linear_system_J(s0, z, r0).detach()
        Rs.append(r0.cpu()); Deltas.append(delta_ref.cpu())
    Rs = torch.cat(Rs, dim=0).to(device)
    Deltas = torch.cat(Deltas, dim=0).to(device)

    # train precond on (Rs -> Deltas)
    print("Training preconditioner...")
    p_ds = torch.utils.data.TensorDataset(Rs, Deltas)
    p_loader = DataLoader(p_ds, batch_size=64, shuffle=True)
    for epoch in range(40):
        tot = 0.0
        for r_batch, d_batch in p_loader:
            r_batch = r_batch.to(device); d_batch = d_batch.to(device)
            opt_p.zero_grad()
            pred = precond(r_batch)
            loss_p = F.mse_loss(pred, d_batch)
            loss_p.backward(); opt_p.step()
            tot += loss_p.item()
        if epoch % 10 == 0:
            print(f"Precond epoch {epoch}, loss {tot/max(1,len(p_loader)):.4e}")
    return surrogate, precond, ds

# ---------------- Main ----------------
if __name__ == "__main__":
    surrogate, precond, ds = train_surrogate_and_precond()
    print("Benchmarking on test subset...")
    successes = {"hybrid":0, "fullnewton":0}
    iters = {"hybrid":[], "fullnewton":[]}
    for idx in range(10):
        sample = ds[idx]
        z = sample["z"].unsqueeze(0)   # (1,n)
        T = sample["T"].unsqueeze(0)   # (1,1)
        P = sample["P"].unsqueeze(0)
        (sb, sx, sy), ok_h, it_h = hybrid_solve_case(surrogate, precond, z, T, P, max_iters=6, tol=1e-6, verbose=True)
        (fb, fx, fy), ok_f, it_f = full_newton_solve(surrogate, z, T, P, max_iters=10, tol=1e-6, verbose=False)
        successes["hybrid"] += int(ok_h); successes["fullnewton"] += int(ok_f)
        iters["hybrid"].append(it_h); iters["fullnewton"].append(it_f)
        print(f"Sample {idx}: hybrid_ok={ok_h} iters={it_h}, fullnewton_ok={ok_f} iters={it_f}")
    import numpy as _np
    print("Summary:", successes, "Iters mean:", {k: float(_np.mean(v)) for k,v in iters.items()})
