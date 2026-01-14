import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

def bs_call_put_price(S, K, T, sigma, r=0.0, q=0.0):
    if T <= 0:
        call = max(S - K, 0.0)
        put = max(K - S, 0.0)
        return call, put

    vol_sqrt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt
    d2 = d1 - vol_sqrt

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    call = disc_q * S * norm.cdf(d1) - disc_r * K * norm.cdf(d2)
    put  = disc_r * K * norm.cdf(-d2) - disc_q * S * norm.cdf(-d1)
    return call, put

def bs_straddle_value(S, K, T, sigma, r=0.0, q=0.0):
    c, p = bs_call_put_price(S, K, T, sigma, r=r, q=q)
    return c + p

def bs_straddle_delta_gamma(S, K, T, sigma, r=0.0, q=0.0):
    if T <= 0:
        raise ValueError("Need T>0 for smooth greeks.")

    vol_sqrt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt
    disc_q = math.exp(-q * T)

    delta_call = disc_q * norm.cdf(d1)
    delta_put  = disc_q * (norm.cdf(d1) - 1.0)

    gamma_one = disc_q * norm.pdf(d1) / (S * vol_sqrt)  # call gamma == put gamma in BS
    delta0 = delta_call + delta_put
    gamma0 = gamma_one + gamma_one
    return delta0, gamma0

def _bisect_root(f, a, b, tol=1e-12, max_iter=200):
    fa = f(a)
    fb = f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise ValueError("root not bracketed")
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or 0.5 * (b - a) < tol:
            return m
        if fa * fm <= 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    return 0.5 * (a + b)

def _bracket_and_solve_positive_root(f, x0=1e-9, x1=1.0, x_max=1e6):
    a = x0
    b = x1
    fa = f(a)
    fb = f(b)
    while fa * fb > 0 and b < x_max:
        b *= 2.0
        fb = f(b)
    if fa * fb > 0:
        raise ValueError("failed to bracket root up to x_max")
    return _bisect_root(f, a, b)

def piecewise_breakpoints_straddle(S0, K, T, sigma, method, r=0.0, q=0.0, delta_L=-1.0, delta_R=1.0):
    """
    Returns (S_L, S_R, gamma_L, gamma_R) for the piecewise quadratic/linear straddle PnL approximation.

    method:
      - "greek_clip": breakpoints where delta0 + gamma0*x hits delta_L/delta_R, using BS gamma0 at S0.
      - "fit_bs": choose breakpoints so that the piecewise model matches BS PnL at S_L and S_R,
                  while enforcing slope continuity with tail slopes delta_L/delta_R.
                  This yields (in general) different effective gammas on left and right.
    """
    V0 = bs_straddle_value(S0, K, T, sigma, r=r, q=q)
    delta0, gamma0 = bs_straddle_delta_gamma(S0, K, T, sigma, r=r, q=q)

    if method == "greek_clip":
        x_L = (delta_L - delta0) / gamma0
        x_R = (delta_R - delta0) / gamma0
        if x_L > x_R:
            x_L, x_R = x_R, x_L
        return S0 + x_L, S0 + x_R, gamma0, gamma0

    if method == "fit_bs":
        # Right: solve BS_PnL(S0 + x) = 0.5*(delta0 + delta_R)*x, x>0
        def f_right(x):
            pnl_bs = bs_straddle_value(S0 + x, K, T, sigma, r=r, q=q) - V0
            pnl_at_join = 0.5 * (delta0 + delta_R) * x
            return pnl_bs - pnl_at_join

        # Left: solve BS_PnL(S0 - y) = 0.5*(delta0 + delta_L)*(-y), y>0
        def f_left(y):
            pnl_bs = bs_straddle_value(S0 - y, K, T, sigma, r=r, q=q) - V0
            pnl_at_join = 0.5 * (delta0 + delta_L) * (-y)
            return pnl_bs - pnl_at_join

        x_R = _bracket_and_solve_positive_root(f_right, x0=1e-9, x1=1.0, x_max=10.0 * S0)
        y_L = _bracket_and_solve_positive_root(f_left,  x0=1e-9, x1=1.0, x_max=10.0 * S0)
        x_L = -y_L

        gamma_L = (delta_L - delta0) / x_L
        gamma_R = (delta_R - delta0) / x_R
        return S0 + x_L, S0 + x_R, gamma_L, gamma_R

    raise ValueError("unknown method")

def pnl_piecewise_quad_linear_two_sided(x, delta0, delta_L, delta_R, x_L, x_R, gamma_L, gamma_R):
    # piecewise PnL, with potentially different gammas on left/right quadratics
    if x <= x_L:
        pnl_at_xL = delta0 * x_L + 0.5 * gamma_L * x_L * x_L
        return pnl_at_xL + delta_L * (x - x_L)
    if x >= x_R:
        pnl_at_xR = delta0 * x_R + 0.5 * gamma_R * x_R * x_R
        return pnl_at_xR + delta_R * (x - x_R)
    if x < 0.0:
        return delta0 * x + 0.5 * gamma_L * x * x
    return delta0 * x + 0.5 * gamma_R * x * x

# ---- parameters ----
S0 = 100.0
K  = 100.0
T  = 0.25
sigma = 0.20
r = 0.0
q = 0.0

PLOT = True

delta_L = -1.0
delta_R = 1.0

V0 = bs_straddle_value(S0, K, T, sigma, r=r, q=q)
delta0, gamma0 = bs_straddle_delta_gamma(S0, K, T, sigma, r=r, q=q)

# breakpoints for both methods (keep both in code/dataframe)
S_L_clip, S_R_clip, gamma_L_clip, gamma_R_clip = piecewise_breakpoints_straddle(
    S0, K, T, sigma, method="greek_clip", r=r, q=q, delta_L=delta_L, delta_R=delta_R
)
S_L_fit, S_R_fit, gamma_L_fit, gamma_R_fit = piecewise_breakpoints_straddle(
    S0, K, T, sigma, method="fit_bs", r=r, q=q, delta_L=delta_L, delta_R=delta_R
)

x_L_clip = S_L_clip - S0
x_R_clip = S_R_clip - S0
x_L_fit  = S_L_fit  - S0
x_R_fit  = S_R_fit  - S0

# grid
S = np.linspace(60.0, 140.0, 161)
x = S - S0

# exact BS PnL
pnl_exact = np.array([bs_straddle_value(float(Si), K, T, sigma, r=r, q=q) - V0 for Si in S])

# delta-gamma (Taylor around S0)
pnl_dg = delta0 * x + 0.5 * gamma0 * x * x

# piecewise: clip rule
pnl_pw_clip = np.array([
    pnl_piecewise_quad_linear_two_sided(float(xi), delta0, delta_L, delta_R, x_L_clip, x_R_clip, gamma_L_clip, gamma_R_clip)
    for xi in x
])

# piecewise: fit-to-BS-at-joins rule
pnl_pw_fit = np.array([
    pnl_piecewise_quad_linear_two_sided(float(xi), delta0, delta_L, delta_R, x_L_fit, x_R_fit, gamma_L_fit, gamma_R_fit)
    for xi in x
])

df = pd.DataFrame({
    "S": S,
    "dS": x,
    "pnl_delta_gamma": pnl_dg,
    "pnl_piecewise_clip": pnl_pw_clip,
    "pnl_piecewise_fit_bs": pnl_pw_fit,
    "pnl_exact_bs": pnl_exact
})

print("V0 =", V0, "delta0 =", delta0, "gamma0 =", gamma0)
print("clip breakpoints: S_L =", S_L_clip, "S_R =", S_R_clip)
print("fit  breakpoints: S_L =", S_L_fit,  "S_R =", S_R_fit)
print(df.to_string())

if PLOT:
    plt.figure()
    plt.plot(df["S"], df["pnl_exact_bs"], label="exact_bs")
    plt.plot(df["S"], df["pnl_delta_gamma"], label="delta_gamma")
    plt.plot(df["S"], df["pnl_piecewise_clip"], label="piecewise_clip")

    # show only the clip linearization points
    plt.axvline(S_L_clip, linestyle=":", linewidth=1.5, label="S_L_clip")
    plt.axvline(S_R_clip, linestyle=":", linewidth=1.5, label="S_R_clip")

    plt.axhline(0.0)
    plt.legend()
    plt.xlabel("S")
    plt.ylabel("profit vs S0")
    plt.title("ATM straddle PnL: delta-gamma vs piecewise_clip")
    plt.show()
