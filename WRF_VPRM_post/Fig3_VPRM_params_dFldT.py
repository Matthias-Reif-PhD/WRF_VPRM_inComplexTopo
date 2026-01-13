"""Fig3: VPRM parameter visualisations.

This module plots:
- GPP and R_eco curves per PFT
- derivatives dGPP/dT and dR_eco/dT per PFT

Refactored into functions for clarity and reuse.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import matplotlib

# use non-interactive backend for script execution (headless servers)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import traceback


# ---------------- PFT parameters (clean table) ----------------
pft_parameters: Dict[str, Dict[str, float]] = {
    "Evergreen Forest": {
        "PAR_0": 207.69,
        "alpha": 0.18627740222039335,
        "beta": 1.6453495029934293,
        "lambda": 0.467,
        "Topt": 14.25,
    },
    "Deciduous Forest": {
        "PAR_0": 183.80,
        "alpha": 0.2323879681886519,
        "beta": 0.8678708197370003,
        "lambda": 0.361,
        "Topt": 23.58,
    },
    "Mixed Forest": {
        "PAR_0": 240.39,
        "alpha": 0.20136626041893155,
        "beta": 2.811577670752306,
        "lambda": 0.248,
        "Topt": 17.44,
    },
    "Cropland": {
        "PAR_0": 364.15,
        "alpha": 0.24582294447901698,
        "beta": 1.1525179918498774,
        "lambda": 0.230,
        "Topt": 22.00,
    },
    "Grassland": {
        "PAR_0": 284.87,
        "alpha": 0.37784002921977233,
        "beta": 1.5010602215726705,
        "lambda": 0.771,
        "Topt": 15.88,
    },
}

# Colors for plotting (same order as dict iteration)
PLOT_COLORS = ["#006400", "#228B22", "#8FBC8F", "#FFA07A", "#7CFC00"]


# ---------------- Constants ----------------
TMIN = 0
TMAX = 41
PAR_CONST = 500.0
P_SCALE = 0.5
LSWI_X_PERCENT = 0.4
LSWI_MAX = 0.8
W_SCALE = (1 + LSWI_X_PERCENT) / (1 + LSWI_MAX)
EVI_CONST = 1.0


def Tscale_array(T: np.ndarray, Tmin: float, Tmax: float, Topt: float) -> np.ndarray:
    """Compute Tscale for an array of temperatures using the original formula.

    Tscale = ((T-Tmin)*(T-Tmax)) / ((T-Tmin)*(T-Tmax) - (T-Topt)**2)
    Stable numerics: where denominator == 0, set Tscale to 0.
    """
    a1 = T - Tmin
    a2 = T - Tmax
    a3 = T - Topt
    denom = a1 * a2 - a3**2
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = np.where(denom == 0.0, 0.0, (a1 * a2) / denom)
    # where resulting ts is negative or NaN, set to zero to avoid nonsensical GPP
    ts = np.where(np.isfinite(ts) & (ts > 0.0), ts, 0.0)
    return ts


def GPP_from_Tscale(Tscale: np.ndarray, lam: float, PAR_0: float) -> np.ndarray:
    """Compute GPP from Tscale using the PAR formulation used previously."""
    return (lam * Tscale * P_SCALE * W_SCALE * EVI_CONST * PAR_CONST) / (
        1.0 + (PAR_CONST / PAR_0)
    )


def RECO_from_T(T: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Simple linear R_eco model used previously: alpha * T + beta"""
    return alpha * T + beta


def dTscale_dT_array(
    T: np.ndarray, Tmin: float, Tmax: float, Topt: float
) -> np.ndarray:
    """Compute derivative dTscale/dT vectorized for an array T.

    Uses analytical derivative from notebook; returns zeros where undefined.
    """
    a1 = T - Tmin
    a2 = T - Tmax
    a3 = T - Topt
    denom = a1 * a2 - a3**2
    numer = (a1 + a2) * denom - a1 * a2 * ((a1 + a2) - 2 * a3)
    with np.errstate(divide="ignore", invalid="ignore"):
        deriv = np.where(denom == 0.0, 0.0, numer / (denom**2))
    deriv = np.where(np.isfinite(deriv), deriv, 0.0)
    return deriv


def plot_gpp_and_reco(
    pft_params: Dict[str, Dict[str, float]], outpath: str = "VPRM_pft_GPP_RECO.pdf"
) -> None:
    """Plot GPP and R_eco curves for each PFT and save to `outpath`."""
    T = np.arange(TMIN, TMAX)
    plt.figure(figsize=(10, 6))
    for (pft, params), color in zip(pft_params.items(), PLOT_COLORS):
        PAR_0 = params["PAR_0"]
        alpha = params["alpha"]
        beta = params["beta"]
        lam = params["lambda"]
        Topt = params["Topt"]

        Tscale = Tscale_array(T, TMIN, TMAX, Topt)
        GPP = GPP_from_Tscale(Tscale, lam, PAR_0)
        RECO = RECO_from_T(T, alpha, beta)

        plt.plot(T, GPP, label=pft, color=color, linewidth=3)
        plt.plot(T, RECO, "--", color=color, linewidth=3)
        plt.axvline(Topt, color=color, linestyle=":", linewidth=3)

    plt.xlabel(r"T$_\mathrm{2m}$ [°C]", fontsize=20)
    plt.ylabel(r"GPP and R$_{\mathrm{eco}}$ [$\mu$mol m$^{-2}$ s$^{-1}$]", fontsize=20)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tick_params(labelsize=20)
    plt.legend(ncol=1, fontsize=20)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def plot_dGPP_dT_and_dRECO_dT(
    pft_params: Dict[str, Dict[str, float]], outpath: str = "VPRM_pft_dGPPdT.pdf"
) -> None:
    """Plot dGPP/dT and dR_eco/dT for each PFT and save to `outpath`."""
    T = np.arange(TMIN, TMAX)
    T_trim = T[5:]
    plt.figure(figsize=(10, 6))
    for (pft, params), color in zip(pft_params.items(), PLOT_COLORS):
        PAR_0 = params["PAR_0"]
        alpha = params["alpha"]
        lam = params["lambda"]
        Topt = params["Topt"]

        dTscale = dTscale_dT_array(T, TMIN, TMAX, Topt)
        dGPPdT = GPP_from_Tscale(dTscale, lam, PAR_0)

        # dRECO/dT is simply alpha (constant). For comparability with discrete plotting, we plot alpha as dashed.
        plt.plot(T_trim, dGPPdT[5:], label=pft, color=color, linewidth=3)
        plt.plot(
            T_trim,
            np.full_like(T_trim, alpha, dtype=float),
            linestyle="--",
            color=color,
            linewidth=3,
        )
        plt.axvline(Topt, color=color, linestyle=":", linewidth=3)

    plt.xlabel(r"T$_\mathrm{2m}$ [°C]", fontsize=20)
    plt.ylabel(
        r"$\partial$GPP/$\partial$T$\,$ and $\partial$R$_{\mathrm{eco}}$/$\partial$T$\,$ [$\mu$mol m$^{-2}$ s$^{-1}$ $^\circ$C$^{-1}$]",
        fontsize=20,
    )
    plt.tick_params(labelsize=20)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Run both figures and save outputs."""
    OUTFOLDER = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"
    print(f"W-scale = {W_SCALE:.3f}")
    try:
        plot_gpp_and_reco(pft_parameters, outpath=f"{OUTFOLDER}/VPRM_pft_GPP_RECO.pdf")
        print("Saved VPRM_pft_GPP_RECO.pdf")
    except Exception:
        print("Error while creating VPRM_pft_GPP_RECO.pdf:")
        traceback.print_exc()

    try:
        plot_dGPP_dT_and_dRECO_dT(
            pft_parameters, outpath=f"{OUTFOLDER}/VPRM_pft_dGPPdT.pdf"
        )
        print("Saved VPRM_pft_dGPPdT.pdf")
    except Exception:
        print("Error while creating VPRM_pft_dGPPdT.pdf:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
