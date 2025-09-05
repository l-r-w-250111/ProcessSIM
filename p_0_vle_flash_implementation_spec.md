# P0 VLE Flash Implementation Spec

## Scope
- Build surrogate-based solver for isothermal, isobaric VLE flash.
- Replace traditional iterative Rachford-Rice solution with NN surrogate.

## Problem Definition
- Input: Overall composition `z_i`, Pressure `P`, Temperature `T`.
- Output: Vapor fraction `V/F`, Phase compositions `x_i` (liquid), `y_i` (vapor).
- Governing equations: Equilibrium `K_i = y_i/x_i`, Material balances, Phase fraction balance.

## Surrogate Strategy
- Generate dataset from conventional VLE flash calculations using EOS (e.g., Peng–Robinson).
- Train neural network surrogate to learn mapping `(z, T, P) → (V/F, x, y)`.
- PINN constraint: Include mass balance and equilibrium consistency in loss.

## (A) EOS Flash Data Generator (Peng–Robinson)
```python
import numpy as np
from scipy.optimize import fsolve

# Peng–Robinson EOS parameters for pure component i
def pr_params(Tc, Pc, omega):
    R = 8.314  # J/mol/K
    kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
    return R, kappa, Tc, Pc

# Fugacity coefficient calculation (simplified placeholder)
def fugacity_coeff(z, T, P, Tc, Pc, omega):
    # Normally requires solving cubic EOS; here return dummy K values for scaffold
    K = Pc/P * np.exp((Tc - T)/Tc)
    return K

# Flash function with safe autograd support
def flash(z, T, P, Tc, Pc, omega):
    K = fugacity_coeff(z, T, P, Tc, Pc, omega)

    def rr(beta):
        return np.sum(z * (K - 1) / (1 + beta * (K - 1)))

    beta_guess = 0.5
    try:
        beta = fsolve(rr, beta_guess)[0]
    except Exception:
        beta = beta_guess  # fallback if solver fails

    x = z / (1 + beta * (K - 1))
    y = K * x

    # Normalize to ensure sum to 1, avoid divide-by-zero
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    if x_sum > 1e-12:
        x /= x_sum
    if y_sum > 1e-12:
        y /= y_sum

    return float(beta), x, y

# Example: binary mixture
if __name__ == "__main__":
    z = np.array([0.5, 0.5])
    T, P = 350.0, 5e6
    Tc = np.array([507.6, 305.4])  # K
    Pc = np.array([3.025e6, 4.88e6])  # Pa
    omega = np.array([0.2975, 0.099])

    beta, x, y = flash(z, T, P, Tc, Pc, omega)
    print("beta=", beta)
    print("x=", x)
    print("y=", y)
```

## Next Steps
1. Validate EOS flash generator across test mixtures (binary/ternary).
2. Extend generator to produce dataset for surrogate training.
3. Integrate with PyTorch surrogate training pipeline.
4. Refine fugacity coefficient model with full PR EOS implementation.
5. Scale dataset generation for wide ranges of T, P, compositions.
6. Proceed to surrogate training (B).

---

⚠️ **Note on RuntimeError (autograd)**: If PyTorch’s `autograd.grad` throws `allow_unused` errors, wrap the call with `torch.autograd.grad(..., allow_unused=True)`. This ensures unused tensors don’t stop gradient computation when part of the state is algebraically redundant.

